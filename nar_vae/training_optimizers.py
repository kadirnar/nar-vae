"""Architecture-aware optimizers for NAR-VAE training.

Muon is intentionally used only for hidden two-dimensional linear weights.  A
standard AdamW optimizer owns every other trainable parameter, as recommended
by the Muon algorithm and the PyTorch implementation.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nar_vae.configuration import resolve_adamw_settings, resolve_muon_settings
from nar_vae.models.dit import DiTBlock, EchoDiT, EncoderTransformerBlock
from nar_vae.models.duration import DurationResidualBlock

_MUON_HIDDEN_CONTAINERS = (DiTBlock, EncoderTransformerBlock, DurationResidualBlock)
_HYBRID_STATE_VERSION = 1


@dataclass(frozen=True)
class MuonParameterPartition:
    """A complete, deterministic partition of trainable model parameters."""

    muon: tuple[nn.Parameter, ...]
    adamw_decay: tuple[nn.Parameter, ...]
    adamw_no_decay: tuple[nn.Parameter, ...]
    muon_names: tuple[str, ...]
    adamw_decay_names: tuple[str, ...]
    adamw_no_decay_names: tuple[str, ...]

    @property
    def trainable_names(self) -> tuple[str, ...]:
        return self.muon_names + self.adamw_decay_names + self.adamw_no_decay_names


def _muon_linear_parameter_ids(model: nn.Module) -> set[int]:
    """Return weights declared to be hidden matrices by the model topology."""
    eligible: set[int] = set()
    for module in model.modules():
        if isinstance(module, _MUON_HIDDEN_CONTAINERS):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    eligible.add(id(child.weight))
        elif isinstance(module, EchoDiT):
            # The middle timestep-conditioning projection is a hidden transform.
            # The first input projection and fused final AdaLN projection remain
            # on AdamW because they represent different input/output subspaces.
            hidden = module.cond_module[2]
            if isinstance(hidden, nn.Linear):
                eligible.add(id(hidden.weight))
    return eligible


def _uses_weight_decay(owner: nn.Module, parameter_name: str, parameter: nn.Parameter) -> bool:
    if parameter_name == "bias" or parameter.ndim < 2:
        return False
    if isinstance(
        owner,
        (
            nn.modules.batchnorm._BatchNorm,
            nn.GroupNorm,
            nn.LayerNorm,
            nn.modules.instancenorm._InstanceNorm,
        ),
    ):
        return False
    # NAR-VAE uses custom RMSNorm modules, including 2-D per-head gains.
    return not owner.__class__.__name__.lower().endswith("norm")


def partition_muon_parameters(model: nn.Module) -> MuonParameterPartition:
    """Partition every trainable parameter exactly once in stable name order.

    Eligibility is based on concrete EchoDiT block types rather than a blanket
    ``parameter.ndim == 2`` rule.  The latter would incorrectly route the
    model's two-dimensional per-head RMSNorm gains through Muon.
    """
    modules = dict(model.named_modules())
    eligible_ids = _muon_linear_parameter_ids(model)
    muon: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    muon_names: list[str] = []
    adamw_decay_names: list[str] = []
    adamw_no_decay_names: list[str] = []
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise ValueError(f"Trainable parameter {name!r} is registered more than once.")
        seen.add(parameter_id)
        module_name, _, parameter_name = name.rpartition(".")
        owner = modules.get(module_name, model)

        if parameter_id in eligible_ids:
            if (
                parameter.ndim != 2
                or parameter_name != "weight"
                or not isinstance(owner, nn.Linear)
            ):
                raise ValueError(f"Muon parameter {name!r} is not a 2-D Linear weight.")
            muon.append(parameter)
            muon_names.append(name)
        elif _uses_weight_decay(owner, parameter_name, parameter):
            adamw_decay.append(parameter)
            adamw_decay_names.append(name)
        else:
            adamw_no_decay.append(parameter)
            adamw_no_decay_names.append(name)

    partition = MuonParameterPartition(
        muon=tuple(muon),
        adamw_decay=tuple(adamw_decay),
        adamw_no_decay=tuple(adamw_no_decay),
        muon_names=tuple(muon_names),
        adamw_decay_names=tuple(adamw_decay_names),
        adamw_no_decay_names=tuple(adamw_no_decay_names),
    )
    if len(seen) != len(partition.trainable_names):
        raise AssertionError("Muon parameter routing did not cover each trainable parameter once.")
    return partition


class HybridMuonAdamW(torch.optim.Optimizer):
    """One Trainer-compatible optimizer backed by native Muon and AdamW.

    Exposing the sub-optimizers' actual parameter-group dictionaries lets one
    Transformers scheduler control all groups.  The nested state dictionary is
    saved as the run's single ``optimizer.pt`` artifact.
    """

    def __init__(
        self,
        *,
        muon_parameters: Sequence[nn.Parameter],
        adamw_parameter_groups: Sequence[dict[str, Any]],
        muon_learning_rate: float,
        muon_weight_decay: float,
        adamw_learning_rate: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        epsilon: float,
        adjust_lr_fn: str,
        adam_betas: tuple[float, float],
        adam_epsilon: float,
    ) -> None:
        muon_parameters = list(muon_parameters)
        if not muon_parameters:
            raise ValueError(
                "optimizer: muon found no eligible hidden 2-D Linear weights after freezing."
            )
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError(
                "optimizer: muon requires PyTorch 2.9 or newer. Reinstall the current nar-vae "
                "package so its PyTorch requirement is satisfied."
            )

        all_parameters = list(muon_parameters)
        for group in adamw_parameter_groups:
            all_parameters.extend(group["params"])
        # Initialize Optimizer's hook machinery before replacing this temporary
        # group with the sub-optimizers' live groups.
        super().__init__(all_parameters, defaults={"lr": muon_learning_rate})

        self.muon_optimizer = muon_class(
            [{"params": muon_parameters, "optimizer_role": "muon"}],
            lr=muon_learning_rate,
            weight_decay=muon_weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=(3.4445, -4.7750, 2.0315),
            eps=epsilon,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        self.adamw_optimizer = (
            torch.optim.AdamW(
                list(adamw_parameter_groups),
                lr=adamw_learning_rate,
                betas=adam_betas,
                eps=adam_epsilon,
            )
            if adamw_parameter_groups
            else None
        )
        self._refresh_optimizer_views()

    def _refresh_optimizer_views(self) -> None:
        adamw_groups = [] if self.adamw_optimizer is None else self.adamw_optimizer.param_groups
        self.param_groups = [*self.muon_optimizer.param_groups, *adamw_groups]
        states = [self.muon_optimizer.state]
        if self.adamw_optimizer is not None:
            states.append(self.adamw_optimizer.state)
        self.state = ChainMap(*states)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon_optimizer.step()
        if self.adamw_optimizer is not None:
            self.adamw_optimizer.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "hybrid_muon_adamw_version": _HYBRID_STATE_VERSION,
            "muon": self.muon_optimizer.state_dict(),
            "adamw": (None if self.adamw_optimizer is None else self.adamw_optimizer.state_dict()),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("hybrid_muon_adamw_version") != _HYBRID_STATE_VERSION:
            raise ValueError("Unsupported hybrid Muon/AdamW optimizer checkpoint version.")
        self.muon_optimizer.load_state_dict(state_dict["muon"])
        adamw_state = state_dict.get("adamw")
        if self.adamw_optimizer is None:
            if adamw_state is not None:
                raise ValueError("Optimizer checkpoint contains an unexpected AdamW state.")
        elif adamw_state is None:
            raise ValueError("Optimizer checkpoint is missing its auxiliary AdamW state.")
        else:
            self.adamw_optimizer.load_state_dict(adamw_state)
        self._refresh_optimizer_views()


def build_muon_optimizer(model: nn.Module, config: Mapping[str, Any]) -> HybridMuonAdamW:
    """Build the validated hybrid optimizer for pretraining, SFT, or GRPO."""
    muon = resolve_muon_settings(config)
    if muon is None:
        raise ValueError("build_muon_optimizer requires optimizer: muon.")
    adamw = resolve_adamw_settings(config)
    partition = partition_muon_parameters(model)
    adamw_groups: list[dict[str, Any]] = []
    if partition.adamw_decay:
        adamw_groups.append(
            {
                "params": list(partition.adamw_decay),
                "weight_decay": float(config.get("weight_decay", 0.01)),
                "optimizer_role": "adamw_decay",
            }
        )
    if partition.adamw_no_decay:
        adamw_groups.append(
            {
                "params": list(partition.adamw_no_decay),
                "weight_decay": 0.0,
                "optimizer_role": "adamw_no_decay",
            }
        )
    return HybridMuonAdamW(
        muon_parameters=partition.muon,
        adamw_parameter_groups=adamw_groups,
        muon_learning_rate=muon.learning_rate,
        muon_weight_decay=muon.weight_decay,
        adamw_learning_rate=float(config["learning_rate"]),
        momentum=muon.momentum,
        nesterov=muon.nesterov,
        ns_steps=muon.ns_steps,
        epsilon=muon.epsilon,
        adjust_lr_fn=muon.adjust_lr_fn,
        adam_betas=(adamw.beta1, adamw.beta2),
        adam_epsilon=adamw.epsilon,
    )


class MuonTrainerMixin:
    """Select the hybrid optimizer without passing unsupported Muon to Transformers."""

    training_config: Mapping[str, Any]

    def create_optimizer(self):
        optimizer_name = str(self.training_config.get("optimizer", "adamw")).strip().lower()
        if self.optimizer is None and optimizer_name == "muon":
            self.optimizer = build_muon_optimizer(self.model, self.training_config)
        if self.optimizer is not None:
            return self.optimizer
        return super().create_optimizer()

    def log(self, logs, *args, **kwargs):
        """Expose both hybrid learning rates to Trainer's mandatory W&B reporter."""
        optimizer = getattr(self, "optimizer", None)
        visited: set[int] = set()
        while optimizer is not None and not isinstance(optimizer, HybridMuonAdamW):
            if id(optimizer) in visited:
                optimizer = None
                break
            visited.add(id(optimizer))
            optimizer = getattr(optimizer, "optimizer", None)
        if isinstance(optimizer, HybridMuonAdamW):
            logs = dict(logs)
            logs["optimizer/muon_enabled"] = 1
            for group in optimizer.param_groups:
                role = group.get("optimizer_role")
                if role == "muon":
                    logs["learning_rate/muon"] = group["lr"]
                elif role in {"adamw_decay", "adamw_no_decay"}:
                    logs.setdefault("learning_rate/aux_adamw", group["lr"])
        return super().log(logs, *args, **kwargs)


__all__ = [
    "HybridMuonAdamW",
    "MuonParameterPartition",
    "MuonTrainerMixin",
    "build_muon_optimizer",
    "partition_muon_parameters",
]
