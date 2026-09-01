"""Executable, resumable flow-GRPO post-training infrastructure.

The stage is intentionally API-driven.  NAR-VAE does not install console entry points, and speech
reward evaluators must be supplied by the server application with immutable identities in the
configuration.  The loop itself owns prompt-group sharding, DDP, metric reduction, W&B logging,
and hash-bound interruption recovery.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from nar_vae.configuration import load_training_lineage
from nar_vae.dataset.identity import (
    training_dataset_identity_sha256,
    validate_training_dataset_identity,
)
from nar_vae.distributed import (
    DistributedContext,
    propagate_distributed_error,
    resolve_node_consistent_value,
)
from nar_vae.model_manifest import ModelManifest, validate_manifest_weight, write_model_manifest
from nar_vae.post_training.grpo import (
    DecodeFunction,
    FlowGRPOConfig,
    FlowGRPOMetrics,
    FlowGRPOTrainer,
    RewardFunction,
    SupervisedLossFunction,
    VelocityAdapter,
)
from nar_vae.training_optimizers import build_muon_optimizer

GRPO_RUN_MANIFEST_FILENAME = "grpo_run_manifest.json"
GRPO_CHECKPOINT_MANIFEST_FILENAME = "grpo_checkpoint_manifest.json"
GRPO_EXPORT_MANIFEST_FILENAME = "grpo_export_manifest.json"
GRPO_DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
GRPO_REFERENCE_MANIFEST_FILENAME = "reference_manifest.json"
GRPO_RUN_MANIFEST_SCHEMA_VERSION = 1
GRPO_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
GRPO_EXPORT_MANIFEST_SCHEMA_VERSION = 1
GRPO_LIBRARY_NAME = "nar-vae"
GRPO_STAGE_NAME = "grpo"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_NAME = re.compile(r"checkpoint-([1-9][0-9]*)")
_REPORTERS = {"wandb"}
_PRECISIONS = {"fp32", "bf16"}
_SCHEDULERS = {"constant", "linear", "cosine"}
_OPTIMIZERS = {"adamw", "muon"}
_MUON_LR_ADJUSTMENTS = {"original", "match_rms_adamw"}


class GRPOStageError(RuntimeError):
    """Raised when a GRPO run or checkpoint violates its immutable contract."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=os.fspath,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=os.fspath) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=os.fspath) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise GRPOStageError(f"Immutable GRPO identity already exists: {path}.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GRPOStageError(f"Missing regular {description}: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GRPOStageError(f"Could not read {description}: {path}.") from exc
    if not isinstance(payload, dict):
        raise GRPOStageError(f"The {description} must be a JSON object: {path}.")
    return payload


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        qualifier = f" greater than or equal to {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be a finite number{qualifier}.")
    return result


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


def _path(value: Any, *, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value).strip():
        raise ValueError(f"{name} must be a non-empty filesystem path.")
    return Path(value).expanduser().resolve()


def _validate_reward_contract(
    weights: Any,
    evaluators: Any,
) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("reward_weights must be a non-empty mapping.")
    normalized_weights: dict[str, float] = {}
    for name, value in weights.items():
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Reward component names must be normalized non-empty strings.")
        normalized_weights[name] = _finite_float(value, name=f"reward_weights.{name}", minimum=0)
    if not any(value > 0 for value in normalized_weights.values()):
        raise ValueError("At least one reward component must have positive weight.")
    if not isinstance(evaluators, Mapping) or set(evaluators) != set(normalized_weights):
        raise ValueError("reward_evaluators must identify every weighted reward component exactly.")
    normalized_evaluators: dict[str, dict[str, str]] = {}
    expected = {"implementation", "revision", "sha256"}
    for name in normalized_weights:
        value = evaluators[name]
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"reward_evaluators.{name} must contain exactly {sorted(expected)}.")
        implementation = value["implementation"]
        revision = value["revision"]
        checksum = value["sha256"]
        if not isinstance(implementation, str) or not implementation.strip():
            raise ValueError(f"reward_evaluators.{name}.implementation must be non-empty.")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError(f"reward_evaluators.{name}.revision must be non-empty.")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise ValueError(f"reward_evaluators.{name}.sha256 must be a lowercase SHA-256.")
        normalized_evaluators[name] = {
            "implementation": implementation,
            "revision": revision,
            "sha256": checksum,
        }
    return normalized_weights, normalized_evaluators


@dataclass(frozen=True)
class GRPOStageConfig:
    """Validated server-run options for one post-SFT flow-GRPO stage."""

    parent_checkpoint: Path
    prompt_dataset_local: Path
    save_folder: Path
    reward_weights: Mapping[str, float]
    reward_evaluators: Mapping[str, Mapping[str, str]]
    resume_from_checkpoint: bool | Path | None = None
    seed: int = 1337
    data_seed: int = 1337
    epochs: int = 1
    max_steps: int | None = None
    prompt_batch_size: int = 1
    optimizer: Literal["adamw", "muon"] = "adamw"
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    muon_learning_rate: float | None = None
    muon_weight_decay: float | None = None
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_epsilon: float = 1e-7
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] = "match_rms_adamw"
    warmup_steps: int = 0
    lr_scheduler_type: Literal["constant", "linear", "cosine"] = "cosine"
    save_steps: int = 100
    logging_steps: int = 1
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    mixed_precision: Literal["fp32", "bf16"] = "bf16"
    ddp_find_unused_parameters: bool = False
    report_to: Literal["wandb"] = "wandb"
    wandb_project: str = "nar-vae-grpo"
    wandb_run_name: str = "nar-vae-flow-grpo"
    prompt_id_column: str = "utterance_id"
    freeze_text_encoder: bool = False
    freeze_speaker_encoder: bool = False
    freeze_language_embedding: bool = False
    freeze_first_n_layers: int = 0
    num_steps: int = 16
    group_size: int = 4
    sde_window_start: int = 1
    sde_window_size: int = 4
    noise_level: float = 0.7
    clip_ratio: float = 0.2
    kl_beta: float = 0.01
    supervised_replay_weight: float = 0.1
    max_grad_norm: float = 1.0
    advantage_epsilon: float = 1e-6
    log_ratio_clip: float = 20.0
    event_reduction: Literal["mean", "sum"] = "mean"
    policy_update_epochs: int = 2

    def __post_init__(self) -> None:
        for field_name in ("parent_checkpoint", "prompt_dataset_local", "save_folder"):
            value = getattr(self, field_name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field_name} must be an absolute pathlib.Path.")
        if self.resume_from_checkpoint is not None and not isinstance(
            self.resume_from_checkpoint, (bool, Path)
        ):
            raise ValueError("resume_from_checkpoint must be null, true, or a checkpoint path.")
        if self.resume_from_checkpoint is False:
            object.__setattr__(self, "resume_from_checkpoint", None)
        if (
            isinstance(self.resume_from_checkpoint, Path)
            and not self.resume_from_checkpoint.is_absolute()
        ):
            raise ValueError("resume_from_checkpoint must be absolute after configuration loading.")
        for name, minimum in (
            ("seed", 0),
            ("data_seed", 0),
            ("epochs", 1),
            ("prompt_batch_size", 1),
            ("warmup_steps", 0),
            ("save_steps", 1),
            ("logging_steps", 1),
            ("dataloader_num_workers", 0),
            ("freeze_first_n_layers", 0),
            ("policy_update_epochs", 2),
        ):
            _integer(getattr(self, name), name=name, minimum=minimum)
        if self.max_steps is not None:
            _integer(self.max_steps, name="max_steps", minimum=1)
        for name in (
            "dataloader_pin_memory",
            "dataloader_drop_last",
            "ddp_find_unused_parameters",
            "freeze_text_encoder",
            "freeze_speaker_encoder",
            "freeze_language_embedding",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        learning_rate = _finite_float(self.learning_rate, name="learning_rate", minimum=0)
        if learning_rate == 0:
            raise ValueError("learning_rate must be positive.")
        _finite_float(self.weight_decay, name="weight_decay", minimum=0)
        if not isinstance(self.optimizer, str) or self.optimizer not in _OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {sorted(_OPTIMIZERS)}.")
        if self.optimizer != "muon":
            inactive_muon_options = []
            for name, default in (
                ("muon_learning_rate", None),
                ("muon_weight_decay", None),
                ("muon_momentum", 0.95),
                ("muon_nesterov", True),
                ("muon_ns_steps", 5),
                ("muon_epsilon", 1e-7),
                ("muon_adjust_lr_fn", "match_rms_adamw"),
            ):
                if getattr(self, name) != default:
                    inactive_muon_options.append(name)
            if inactive_muon_options:
                raise ValueError(
                    f"Muon-only fields require optimizer: muon: {sorted(inactive_muon_options)}."
                )
        beta1 = _finite_float(self.adam_beta1, name="adam_beta1", minimum=0)
        beta2 = _finite_float(self.adam_beta2, name="adam_beta2", minimum=0)
        if beta1 >= 1 or beta2 >= 1:
            raise ValueError("Adam beta values must satisfy 0 <= beta < 1.")
        epsilon = _finite_float(self.adam_epsilon, name="adam_epsilon", minimum=0)
        if epsilon == 0:
            raise ValueError("adam_epsilon must be positive.")
        if self.muon_learning_rate is not None:
            muon_learning_rate = _finite_float(
                self.muon_learning_rate,
                name="muon_learning_rate",
                minimum=0,
            )
            if muon_learning_rate == 0:
                raise ValueError("muon_learning_rate must be positive.")
        if self.muon_weight_decay is not None:
            _finite_float(
                self.muon_weight_decay,
                name="muon_weight_decay",
                minimum=0,
            )
        muon_momentum = _finite_float(
            self.muon_momentum,
            name="muon_momentum",
            minimum=0,
        )
        if muon_momentum >= 1:
            raise ValueError("muon_momentum must be in [0, 1).")
        if not isinstance(self.muon_nesterov, bool):
            raise ValueError("muon_nesterov must be a boolean.")
        _integer(self.muon_ns_steps, name="muon_ns_steps", minimum=1)
        if self.muon_ns_steps >= 100:
            raise ValueError("muon_ns_steps must be an integer in [1, 100).")
        muon_epsilon = _finite_float(
            self.muon_epsilon,
            name="muon_epsilon",
            minimum=0,
        )
        if muon_epsilon == 0:
            raise ValueError("muon_epsilon must be positive.")
        if (
            not isinstance(self.muon_adjust_lr_fn, str)
            or self.muon_adjust_lr_fn not in _MUON_LR_ADJUSTMENTS
        ):
            raise ValueError("muon_adjust_lr_fn must be 'original' or 'match_rms_adamw'.")
        if self.lr_scheduler_type not in _SCHEDULERS:
            raise ValueError(f"lr_scheduler_type must be one of {sorted(_SCHEDULERS)}.")
        if self.mixed_precision not in _PRECISIONS:
            raise ValueError(
                "GRPO mixed_precision supports fp32 or bf16. fp16 is disabled because this "
                "stage does not silently omit dynamic loss scaling."
            )
        if self.report_to not in _REPORTERS:
            raise ValueError("W&B logging is mandatory; report_to must be 'wandb'.")
        for name in ("wandb_project", "wandb_run_name", "prompt_id_column"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        normalized_weights, normalized_evaluators = _validate_reward_contract(
            self.reward_weights,
            self.reward_evaluators,
        )
        object.__setattr__(self, "reward_weights", normalized_weights)
        object.__setattr__(self, "reward_evaluators", normalized_evaluators)
        # Reuse the numerical primitive's complete validation.
        self.flow_config()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GRPOStageConfig":
        if not isinstance(value, Mapping):
            raise ValueError("GRPO configuration must be a mapping.")
        raw = dict(value)
        stage = raw.pop("training_stage", None)
        if stage != GRPO_STAGE_NAME:
            raise ValueError("GRPO configuration requires training_stage: grpo.")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown GRPO configuration fields: {unknown}.")
        for field_name in ("parent_checkpoint", "prompt_dataset_local", "save_folder"):
            if field_name not in raw:
                raise ValueError(f"GRPO configuration requires {field_name}.")
            raw[field_name] = _path(raw[field_name], name=field_name)
        resume = raw.get("resume_from_checkpoint")
        if resume not in (None, False, True):
            raw["resume_from_checkpoint"] = _path(resume, name="resume_from_checkpoint")
        return cls(**raw)

    def flow_config(self) -> FlowGRPOConfig:
        return FlowGRPOConfig(
            num_steps=self.num_steps,
            group_size=self.group_size,
            sde_window_start=self.sde_window_start,
            sde_window_size=self.sde_window_size,
            noise_level=self.noise_level,
            clip_ratio=self.clip_ratio,
            kl_beta=self.kl_beta,
            supervised_replay_weight=self.supervised_replay_weight,
            max_grad_norm=self.max_grad_norm,
            advantage_epsilon=self.advantage_epsilon,
            log_ratio_clip=self.log_ratio_clip,
            event_reduction=self.event_reduction,
            policy_update_epochs=self.policy_update_epochs,
        )

    def manifest_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training_stage"] = GRPO_STAGE_NAME
        payload.pop("resume_from_checkpoint")
        return payload

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest_payload())


def load_grpo_stage_config(path: str | os.PathLike[str]) -> GRPOStageConfig:
    """Load one YAML GRPO configuration without importing it at package import time."""
    try:
        import yaml
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - dependency failure path
        raise RuntimeError(
            "GRPO training requires PyYAML. Reinstall the single package with "
            "`python -m pip install -e .`."
        ) from exc
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return GRPOStageConfig.from_mapping(value)


@dataclass(frozen=True)
class GRPOPreparedBatch:
    """One prompt batch expanded into candidate groups for the low-level trainer."""

    initial_state: torch.Tensor
    conditioning: Any
    trainer_batch: Any
    event_mask: torch.Tensor | None = None


PrepareBatchFunction = Callable[
    [Any, torch.device, int, torch.Generator],
    GRPOPreparedBatch,
]


@dataclass
class GRPOStageRuntime:
    """Concrete model/data/evaluator adapters consumed by :func:`run_grpo_stage`."""

    policy: nn.Module
    reference_policy: nn.Module
    collate_fn: Callable[[Sequence[Any]], Any]
    prepare_batch: PrepareBatchFunction
    velocity_adapter: VelocityAdapter
    decode: DecodeFunction
    reward: RewardFunction
    model_export_config: Mapping[str, Any]
    parent_model_manifest: ModelManifest
    supervised_loss: SupervisedLossFunction | None = None


class RankPromptSampler(Sampler[int]):
    """Deterministically shard whole prompt groups without cross-rank padding."""

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        self.dataset_size = _integer(dataset_size, name="dataset_size", minimum=1)
        self.world_size = _integer(world_size, name="world_size", minimum=1)
        self.rank = _integer(rank, name="rank", minimum=0)
        self.seed = _integer(seed, name="seed", minimum=0)
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size.")
        if self.dataset_size < self.world_size:
            raise ValueError("The prompt dataset must contain at least one group per rank.")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _integer(epoch, name="epoch", minimum=0)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.dataset_size, generator=generator).tolist()
        usable = (len(order) // self.world_size) * self.world_size
        return iter(order[:usable][self.rank :: self.world_size])

    def __len__(self) -> int:
        return self.dataset_size // self.world_size


def validate_prompt_group_dataset(dataset: Dataset, *, prompt_id_column: str) -> None:
    """Validate the row-level prompt boundary that the sampler keeps rank-local."""
    if len(dataset) <= 0:
        raise ValueError("The GRPO prompt dataset must contain at least one row.")
    columns = getattr(dataset, "column_names", None)
    required = {prompt_id_column, "conditioning_ids", "latents"}
    if not isinstance(columns, (list, tuple)) or not required.issubset(columns):
        missing = sorted(required - set(columns or ()))
        raise ValueError(f"The GRPO prompt dataset is missing required columns: {missing}.")
    try:
        values = dataset[prompt_id_column]
    except (KeyError, TypeError, ValueError):
        fallback_values = []
        for index in range(len(dataset)):
            row = dataset[index]
            if not isinstance(row, Mapping):
                raise ValueError(f"GRPO prompt row {index} must be a mapping.")
            fallback_values.append(row.get(prompt_id_column))
        values = fallback_values
    prompt_ids: set[str] = set()
    for index, prompt_id in enumerate(values):
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"GRPO prompt row {index} has an invalid prompt identifier.")
        if prompt_id in prompt_ids:
            raise ValueError(f"GRPO prompt identifiers must be unique; duplicate {prompt_id!r}.")
        prompt_ids.add(prompt_id)


def grpo_reference_identity(
    checkpoint_path: str | os.PathLike[str],
    model_manifest: ModelManifest,
) -> dict[str, Any]:
    """Bind one SFT checkpoint, its pretraining ancestry, and training lineage."""
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if model_manifest.stage != "sft" or model_manifest.parent is None:
        raise GRPOStageError("The GRPO reference must be a validated SFT model manifest.")
    if model_manifest.parent.get("stage") != "pretrain":
        raise GRPOStageError("The GRPO SFT reference lacks its scratch-pretraining parent.")
    is_ema = checkpoint.name in {"ema_model.bin", "pytorch_model_ema.bin"} or (
        "_ema" in checkpoint.stem
    )
    base = None
    if is_ema:
        base = (
            checkpoint.with_name("pytorch_model.bin")
            if checkpoint.name in {"ema_model.bin", "pytorch_model_ema.bin"}
            else Path(str(checkpoint).replace("_ema", ""))
        )
        validate_manifest_weight(model_manifest, base)
    lineage_checkpoint = base or checkpoint
    lineage = load_training_lineage(lineage_checkpoint)
    if lineage.get("stage") != "sft":
        raise GRPOStageError("The GRPO reference training lineage must be an SFT export.")
    parent = lineage.get("parent")
    if not isinstance(parent, Mapping) or parent.get("stage") != "pretrain":
        raise GRPOStageError("The GRPO reference lineage must retain scratch pretraining.")
    if lineage.get("checkpoint_file") != lineage_checkpoint.name:
        raise GRPOStageError("The SFT training lineage does not select the GRPO parent weight.")
    lineage_checkpoint_hash = _file_sha256(lineage_checkpoint)
    if lineage.get("checkpoint_sha256") != lineage_checkpoint_hash:
        raise GRPOStageError("The GRPO parent weight does not match its SFT training lineage.")
    return {
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": _file_sha256(checkpoint),
        "base_checkpoint_filename": base.name if base is not None else None,
        "base_checkpoint_sha256": lineage_checkpoint_hash if base is not None else None,
        "model_manifest_sha256": model_manifest.sha256,
        "training_lineage_sha256": _canonical_sha256(lineage),
        "stage": "sft",
        "pretraining_parent": dict(model_manifest.parent),
    }


def _validate_reference_identity(value: Any) -> dict[str, Any]:
    expected = {
        "checkpoint_filename",
        "checkpoint_sha256",
        "base_checkpoint_filename",
        "base_checkpoint_sha256",
        "model_manifest_sha256",
        "training_lineage_sha256",
        "stage",
        "pretraining_parent",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise GRPOStageError("The GRPO reference identity has incomplete or unknown fields.")
    identity = dict(value)
    filename = identity["checkpoint_filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise GRPOStageError("The GRPO reference has an invalid selected filename.")
    for name in ("checkpoint_sha256", "model_manifest_sha256", "training_lineage_sha256"):
        if not isinstance(identity[name], str) or not _SHA256.fullmatch(identity[name]):
            raise GRPOStageError(f"The GRPO reference has an invalid {name}.")
    base_name = identity["base_checkpoint_filename"]
    base_hash = identity["base_checkpoint_sha256"]
    if (base_name is None) != (base_hash is None):
        raise GRPOStageError("GRPO reference base filename and hash must be set together.")
    if base_name is not None:
        if not isinstance(base_name, str) or Path(base_name).name != base_name:
            raise GRPOStageError("The GRPO reference base filename is invalid.")
        if not isinstance(base_hash, str) or not _SHA256.fullmatch(base_hash):
            raise GRPOStageError("The GRPO reference base checkpoint hash is invalid.")
    if identity["stage"] != "sft":
        raise GRPOStageError("The GRPO reference identity must select an SFT checkpoint.")
    parent = identity["pretraining_parent"]
    parent_fields = {"manifest_sha256", "stage", "weights_sha256", "representation_sha256"}
    if (
        not isinstance(parent, Mapping)
        or set(parent) != parent_fields
        or parent["stage"] != "pretrain"
    ):
        raise GRPOStageError("The GRPO reference identity lacks its pretraining parent.")
    for name in parent_fields - {"stage"}:
        if not isinstance(parent[name], str) or not _SHA256.fullmatch(parent[name]):
            raise GRPOStageError(f"The GRPO pretraining parent has an invalid {name}.")
    identity["pretraining_parent"] = dict(parent)
    return identity


def _state_dict_fingerprint(module: nn.Module) -> str:
    """Hash tensor semantics independently of a torch.save container."""
    digest = hashlib.sha256()
    for name, value in sorted(_unwrap_module(module).state_dict().items()):
        if not isinstance(value, torch.Tensor):
            raise GRPOStageError(f"Model state {name!r} is not a tensor.")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _unwrap_module(module: nn.Module) -> nn.Module:
    current = module
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        wrapped = getattr(current, "module", None)
        if isinstance(wrapped, nn.Module) and wrapped is not current:
            current = wrapped
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module) and original is not current:
            current = original
            continue
        break
    return current


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu() for name, value in _unwrap_module(module).state_dict().items()
    }


def _capture_rng_state(
    device: torch.device,
    *,
    rollout_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
        "rollout": rollout_generator.get_state(),
        "loader": loader_generator.get_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng_state(
    state: Mapping[str, Any],
    device: torch.device,
    *,
    rollout_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> None:
    expected = {"python", "numpy", "torch_cpu", "torch_cuda", "rollout", "loader"}
    if set(state) != expected:
        raise GRPOStageError("The checkpoint RNG state has an incomplete schema.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        cuda_state = state["torch_cuda"]
        if not isinstance(cuda_state, torch.Tensor):
            raise GRPOStageError("A CUDA GRPO resume requires the saved rank-local CUDA RNG.")
        torch.cuda.set_rng_state(cuda_state, device)
    elif state["torch_cuda"] is not None:
        raise GRPOStageError("A CUDA GRPO checkpoint cannot be exactly resumed on CPU.")
    rollout_generator.set_state(state["rollout"])
    loader_generator.set_state(state["loader"])


def _all_rank_rng_states(
    local_state: Mapping[str, Any],
    *,
    context: DistributedContext,
) -> list[Mapping[str, Any]]:
    if not context.is_distributed:
        return [local_state]
    gathered: list[Mapping[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, dict(local_state))
    if any(value is None for value in gathered):
        raise GRPOStageError("Could not gather every rank-local RNG state.")
    return [value for value in gathered if value is not None]


def reduce_grpo_metrics(
    metrics: FlowGRPOMetrics,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Average rank-local optimizer metrics before rank-zero reporting."""
    names = tuple(FlowGRPOMetrics.__dataclass_fields__)
    values = torch.tensor(
        [float(getattr(metrics, name)) for name in names],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
    return {name: float(value) for name, value in zip(names, values.cpu().tolist(), strict=True)}


def _synchronize_rank_error(
    context: DistributedContext,
    error: Exception | None,
    description: str,
) -> None:
    """Make rank-local callback/startup failures visible before the next collective phase."""
    propagate_distributed_error(context, error, description=description)


def _rank_consistent_call(
    context: DistributedContext,
    operation: Callable[[], Any],
    *,
    description: str,
) -> Any:
    """Run rank-local startup work and agree on success before the next phase."""
    result = None
    error: Exception | None = None
    try:
        result = operation()
    except Exception as exc:
        error = exc
    _synchronize_rank_error(context, error, description)
    return result


def _run_manifest_payload(
    config: GRPOStageConfig,
    *,
    dataset_identity: Mapping[str, Any],
    reference_identity: Mapping[str, Any],
    reference_state_sha256: str,
    world_size: int,
    run_id: str,
) -> dict[str, Any]:
    dataset = validate_training_dataset_identity(dataset_identity)
    if not _SHA256.fullmatch(reference_state_sha256):
        raise GRPOStageError("The reference state fingerprint is invalid.")
    return {
        "schema_version": GRPO_RUN_MANIFEST_SCHEMA_VERSION,
        "library": GRPO_LIBRARY_NAME,
        "stage": GRPO_STAGE_NAME,
        "run_id": run_id,
        "config_sha256": config.sha256,
        "dataset": dataset,
        "dataset_sha256": training_dataset_identity_sha256(dataset),
        "reference": _validate_reference_identity(reference_identity),
        "reference_state_sha256": reference_state_sha256,
        "world_size": world_size,
    }


def _validate_run_manifest(
    value: Any,
    config: GRPOStageConfig,
    *,
    dataset_identity: Mapping[str, Any],
    reference_identity: Mapping[str, Any],
    reference_state_sha256: str,
    world_size: int,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "library",
        "stage",
        "run_id",
        "config_sha256",
        "dataset",
        "dataset_sha256",
        "reference",
        "reference_state_sha256",
        "world_size",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise GRPOStageError("The GRPO run manifest has incomplete or unknown fields.")
    manifest = dict(value)
    if (
        manifest["schema_version"] != GRPO_RUN_MANIFEST_SCHEMA_VERSION
        or manifest["library"] != GRPO_LIBRARY_NAME
        or manifest["stage"] != GRPO_STAGE_NAME
    ):
        raise GRPOStageError("The GRPO run manifest has an unsupported identity.")
    try:
        parsed_id = uuid.UUID(manifest["run_id"])
    except (ValueError, TypeError, AttributeError) as exc:
        raise GRPOStageError("The GRPO run manifest has an invalid run_id.") from exc
    if str(parsed_id) != manifest["run_id"]:
        raise GRPOStageError("The GRPO run manifest has a noncanonical run_id.")
    expected = _run_manifest_payload(
        config,
        dataset_identity=dataset_identity,
        reference_identity=reference_identity,
        reference_state_sha256=reference_state_sha256,
        world_size=world_size,
        run_id=manifest["run_id"],
    )
    if manifest != expected:
        raise GRPOStageError(
            "The GRPO run configuration, prompt dataset, reference policy, or world size "
            "does not match the immutable run manifest."
        )
    return manifest


def _initialize_run_manifest(
    config: GRPOStageConfig,
    *,
    dataset_identity: Mapping[str, Any],
    reference_identity: Mapping[str, Any],
    reference_state_sha256: str,
    context: DistributedContext,
) -> dict[str, Any]:
    output = config.save_folder
    path = output / GRPO_RUN_MANIFEST_FILENAME
    error: BaseException | None = None
    if context.is_main_process:
        try:
            if config.resume_from_checkpoint is None:
                if output.exists() and not output.is_dir():
                    raise GRPOStageError(f"save_folder is not a directory: {output}.")
                if output.exists() and any(output.iterdir()):
                    raise GRPOStageError(f"Fresh GRPO requires an empty save_folder: {output}.")
                output.mkdir(parents=True, exist_ok=True)
                payload = _run_manifest_payload(
                    config,
                    dataset_identity=dataset_identity,
                    reference_identity=reference_identity,
                    reference_state_sha256=reference_state_sha256,
                    world_size=context.world_size,
                    run_id=str(uuid.uuid4()),
                )
                _atomic_create_json(path, payload)
            else:
                _validate_run_manifest(
                    _read_json(path, description="GRPO run manifest"),
                    config,
                    dataset_identity=dataset_identity,
                    reference_identity=reference_identity,
                    reference_state_sha256=reference_state_sha256,
                    world_size=context.world_size,
                )
        except BaseException as exc:  # propagate one rank-zero identity failure to every rank
            error = exc
    if context.is_distributed:
        errors = [repr(error) if error is not None else None]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            if error is not None:
                raise error
            raise GRPOStageError(f"Rank zero could not initialize the GRPO run: {errors[0]}")
    elif error is not None:
        raise error
    manifest = _rank_consistent_call(
        context,
        lambda: _validate_run_manifest(
            _read_json(path, description="GRPO run manifest"),
            config,
            dataset_identity=dataset_identity,
            reference_identity=reference_identity,
            reference_state_sha256=reference_state_sha256,
            world_size=context.world_size,
        ),
        description="GRPO run-manifest local validation",
    )
    assert manifest is not None
    return manifest


def _artifact_inventory(directory: Path, *, manifest_filename: str) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise GRPOStageError(f"GRPO checkpoint artifacts cannot be symlinks: {path}.")
        if not path.is_file() or path.name == manifest_filename:
            continue
        relative = path.relative_to(directory).as_posix()
        artifacts[relative] = path
    return artifacts


def _validate_checkpoint_manifest(
    checkpoint: Path,
    *,
    run_manifest: Mapping[str, Any],
    expected_global_step: int | None = None,
) -> dict[str, Any]:
    value = _read_json(
        checkpoint / GRPO_CHECKPOINT_MANIFEST_FILENAME,
        description="GRPO checkpoint manifest",
    )
    expected_fields = {
        "schema_version",
        "library",
        "stage",
        "kind",
        "run_id",
        "config_sha256",
        "dataset_sha256",
        "reference",
        "world_size",
        "global_step",
        "artifact_sha256",
    }
    if set(value) != expected_fields:
        raise GRPOStageError("The GRPO checkpoint manifest has incomplete or unknown fields.")
    if (
        value["schema_version"] != GRPO_CHECKPOINT_MANIFEST_SCHEMA_VERSION
        or value["library"] != GRPO_LIBRARY_NAME
        or value["stage"] != GRPO_STAGE_NAME
        or value["kind"] != "training_checkpoint"
    ):
        raise GRPOStageError("Unsupported GRPO checkpoint manifest identity.")
    for name in ("run_id", "config_sha256", "dataset_sha256", "reference", "world_size"):
        if value[name] != run_manifest[name]:
            raise GRPOStageError(f"The GRPO checkpoint {name} does not match its run.")
    if expected_global_step is None:
        match = _CHECKPOINT_NAME.fullmatch(checkpoint.name)
        if match is None or int(match.group(1)) != value["global_step"]:
            raise GRPOStageError("The GRPO checkpoint directory and global step disagree.")
    elif value["global_step"] != expected_global_step:
        raise GRPOStageError("The GRPO checkpoint manifest has the wrong global step.")
    hashes = value["artifact_sha256"]
    if not isinstance(hashes, Mapping) or not hashes:
        raise GRPOStageError("The GRPO checkpoint manifest has no artifact hashes.")
    artifacts = _artifact_inventory(
        checkpoint,
        manifest_filename=GRPO_CHECKPOINT_MANIFEST_FILENAME,
    )
    if set(artifacts) != set(hashes):
        raise GRPOStageError("The GRPO checkpoint artifact set does not match its manifest.")
    required = {
        "pytorch_model.bin",
        "reference_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng.pt",
        "state.json",
        "nar_vae_manifest.json",
        GRPO_DATASET_MANIFEST_FILENAME,
        GRPO_REFERENCE_MANIFEST_FILENAME,
    }
    if not required.issubset(artifacts):
        raise GRPOStageError("The GRPO checkpoint is missing resumable state artifacts.")
    for relative, path in artifacts.items():
        checksum = hashes.get(relative)
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise GRPOStageError(f"GRPO artifact {relative!r} has an invalid hash.")
        if _file_sha256(path) != checksum:
            raise GRPOStageError(f"GRPO artifact {relative!r} failed SHA-256 validation.")
    if (
        _read_json(
            checkpoint / GRPO_DATASET_MANIFEST_FILENAME,
            description="GRPO checkpoint dataset manifest",
        )
        != run_manifest["dataset"]
    ):
        raise GRPOStageError("The GRPO checkpoint dataset manifest does not match its run.")
    expected_reference = {
        "reference": run_manifest["reference"],
        "reference_state_sha256": run_manifest["reference_state_sha256"],
    }
    if (
        _read_json(
            checkpoint / GRPO_REFERENCE_MANIFEST_FILENAME,
            description="GRPO checkpoint reference manifest",
        )
        != expected_reference
    ):
        raise GRPOStageError("The GRPO checkpoint reference manifest does not match its run.")
    return value


def _resolve_resume_checkpoint(
    config: GRPOStageConfig,
    *,
    run_manifest: Mapping[str, Any],
) -> Path | None:
    selector = config.resume_from_checkpoint
    if selector is None:
        return None
    if selector is True:
        candidates = []
        for path in config.save_folder.iterdir():
            match = _CHECKPOINT_NAME.fullmatch(path.name)
            if match is not None and path.is_dir() and not path.is_symlink():
                candidates.append((int(match.group(1)), path))
        for _, candidate in sorted(candidates, reverse=True):
            if (candidate / GRPO_CHECKPOINT_MANIFEST_FILENAME).is_file():
                try:
                    _validate_checkpoint_manifest(candidate, run_manifest=run_manifest)
                except (GRPOStageError, OSError):
                    # Automatic resume means the latest *valid* immutable seal. A torn write,
                    # corrupt artifact, or checkpoint from a different run cannot prevent an
                    # older valid checkpoint from being selected. Explicit paths below remain
                    # fail-closed so operator mistakes are never silently redirected.
                    continue
                return candidate.resolve()
        raise GRPOStageError(
            "resume_from_checkpoint is true, but no valid sealed checkpoint exists."
        )
    checkpoint = Path(selector).resolve()
    if (
        checkpoint.parent != config.save_folder
        or _CHECKPOINT_NAME.fullmatch(checkpoint.name) is None
    ):
        raise GRPOStageError(
            "GRPO resume must select checkpoint-N inside the immutable save_folder."
        )
    _validate_checkpoint_manifest(checkpoint, run_manifest=run_manifest)
    return checkpoint


def _scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be smaller than the total GRPO optimizer steps.")

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if kind == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _seed_process(seed: int, *, rank: int, device: torch.device) -> None:
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    torch.manual_seed(process_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(process_seed)


class _WandbLogger:
    def __init__(self, config: GRPOStageConfig, *, enabled: bool) -> None:
        self.run = None
        if not enabled:
            return
        try:
            import wandb
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - lazy dependency path
            raise RuntimeError(
                "W&B is required for GRPO but wandb is unavailable. "
                "Install the complete nar-vae package before starting training."
            ) from exc
        self.run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.manifest_payload(),
        )

    def log(self, metrics: Mapping[str, float], *, step: int) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def _save_checkpoint(
    *,
    config: GRPOStageConfig,
    runtime: GRPOStageRuntime,
    policy: nn.Module,
    reference_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    state: Mapping[str, int],
    run_manifest: Mapping[str, Any],
    context: DistributedContext,
    device: torch.device,
    rollout_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> Path:
    local_rng = None
    rng_error: Exception | None = None
    try:
        local_rng = _capture_rng_state(
            device,
            rollout_generator=rollout_generator,
            loader_generator=loader_generator,
        )
    except Exception as exc:
        rng_error = exc
    _synchronize_rank_error(context, rng_error, "GRPO RNG-state capture")
    assert local_rng is not None
    rng_states = _all_rank_rng_states(local_rng, context=context)
    if context.is_distributed:
        dist.barrier()
    destination = config.save_folder / f"checkpoint-{state['global_step']}"
    save_error: BaseException | None = None
    if context.is_main_process:
        temporary: Path | None = None
        try:
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.",
                    suffix=".tmp",
                    dir=config.save_folder,
                )
            )
            if destination.exists():
                try:
                    _validate_checkpoint_manifest(destination, run_manifest=run_manifest)
                except (GRPOStageError, OSError):
                    # Automatic latest-valid recovery may intentionally replay past a corrupt
                    # newer seal. Preserve those bytes under a non-candidate name before
                    # publishing the replacement; a valid immutable checkpoint is never moved.
                    rejected = config.save_folder / (
                        f".rejected-{destination.name}.{uuid.uuid4().hex}"
                    )
                    os.rename(destination, rejected)
                else:
                    raise GRPOStageError(
                        f"Refusing to overwrite valid GRPO checkpoint: {destination}."
                    )
            current_reference = _state_dict_fingerprint(reference_policy)
            if current_reference != run_manifest["reference_state_sha256"]:
                raise GRPOStageError("The frozen GRPO reference policy changed during training.")
            torch.save(_cpu_state_dict(policy), temporary / "pytorch_model.bin")
            torch.save(_cpu_state_dict(reference_policy), temporary / "reference_model.bin")
            torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
            torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
            torch.save(rng_states, temporary / "rng.pt")
            _atomic_write_json(temporary / "state.json", state)
            _atomic_write_json(
                temporary / GRPO_DATASET_MANIFEST_FILENAME,
                run_manifest["dataset"],
            )
            _atomic_write_json(
                temporary / GRPO_REFERENCE_MANIFEST_FILENAME,
                {
                    "reference": run_manifest["reference"],
                    "reference_state_sha256": run_manifest["reference_state_sha256"],
                },
            )
            write_model_manifest(
                temporary,
                runtime.model_export_config,
                stage="grpo",
                checkpoint_files=("pytorch_model.bin",),
                parent_manifest=runtime.parent_model_manifest,
                parent_checkpoint_path=config.parent_checkpoint,
                parent_base_checkpoint_path=(
                    config.parent_checkpoint.with_name(
                        run_manifest["reference"]["base_checkpoint_filename"]
                    )
                    if run_manifest["reference"]["base_checkpoint_filename"] is not None
                    else None
                ),
            )
            artifacts = _artifact_inventory(
                temporary,
                manifest_filename=GRPO_CHECKPOINT_MANIFEST_FILENAME,
            )
            manifest = {
                "schema_version": GRPO_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
                "library": GRPO_LIBRARY_NAME,
                "stage": GRPO_STAGE_NAME,
                "kind": "training_checkpoint",
                "run_id": run_manifest["run_id"],
                "config_sha256": run_manifest["config_sha256"],
                "dataset_sha256": run_manifest["dataset_sha256"],
                "reference": run_manifest["reference"],
                "world_size": run_manifest["world_size"],
                "global_step": state["global_step"],
                "artifact_sha256": {
                    relative: _file_sha256(path) for relative, path in artifacts.items()
                },
            }
            _atomic_write_json(temporary / GRPO_CHECKPOINT_MANIFEST_FILENAME, manifest)
            _validate_checkpoint_manifest(
                temporary,
                run_manifest=run_manifest,
                expected_global_step=state["global_step"],
            )
            os.rename(temporary, destination)
        except BaseException as exc:
            save_error = exc
        finally:
            if temporary is not None and temporary.exists():
                try:
                    shutil.rmtree(temporary)
                except BaseException as exc:
                    if save_error is None:
                        save_error = exc
    if context.is_distributed:
        errors = [repr(save_error) if save_error is not None else None]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            if save_error is not None:
                raise save_error
            raise GRPOStageError(f"Rank zero could not save the GRPO checkpoint: {errors[0]}")
        dist.barrier()
    elif save_error is not None:
        raise save_error
    return destination


def _load_checkpoint(
    checkpoint: Path,
    *,
    policy: nn.Module,
    reference_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    run_manifest: Mapping[str, Any],
    context: DistributedContext,
    device: torch.device,
    rollout_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> dict[str, int]:
    _validate_checkpoint_manifest(checkpoint, run_manifest=run_manifest)
    policy_state = torch.load(
        checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    reference_state = torch.load(
        checkpoint / "reference_model.bin", map_location="cpu", weights_only=True
    )
    _unwrap_module(policy).load_state_dict(policy_state, strict=True)
    _unwrap_module(reference_policy).load_state_dict(reference_state, strict=True)
    if _state_dict_fingerprint(reference_policy) != run_manifest["reference_state_sha256"]:
        raise GRPOStageError("The resumed reference policy differs from the immutable SFT parent.")
    optimizer.load_state_dict(
        torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    state = _read_json(checkpoint / "state.json", description="GRPO trainer state")
    expected_state = {"global_step", "epoch", "next_batch_index"}
    if set(state) != expected_state or any(
        isinstance(state[name], bool) or not isinstance(state[name], int) or state[name] < 0
        for name in expected_state
    ):
        raise GRPOStageError("The GRPO trainer state has an invalid schema.")
    if state["global_step"] != int(checkpoint.name.rsplit("-", 1)[1]):
        raise GRPOStageError("The GRPO trainer state and checkpoint step disagree.")
    rng_states = torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False)
    if not isinstance(rng_states, list) or len(rng_states) != context.world_size:
        raise GRPOStageError("The GRPO checkpoint does not contain one RNG state per rank.")
    _restore_rng_state(
        rng_states[context.rank],
        device,
        rollout_generator=rollout_generator,
        loader_generator=loader_generator,
    )
    return {name: int(state[name]) for name in expected_state}


def _write_final_export(
    *,
    config: GRPOStageConfig,
    runtime: GRPOStageRuntime,
    policy: nn.Module,
    run_manifest: Mapping[str, Any],
    global_step: int,
    context: DistributedContext,
) -> Path:
    if context.is_distributed:
        dist.barrier()
    destination = config.save_folder / "final"
    export_error: BaseException | None = None
    if context.is_main_process:
        temporary: Path | None = None
        try:
            temporary = Path(
                tempfile.mkdtemp(prefix=".final.", suffix=".tmp", dir=config.save_folder)
            )
            if destination.exists():
                raise GRPOStageError(f"Refusing to overwrite GRPO export: {destination}.")
            torch.save(_cpu_state_dict(policy), temporary / "pytorch_model.bin")
            _atomic_write_json(
                temporary / GRPO_DATASET_MANIFEST_FILENAME,
                run_manifest["dataset"],
            )
            _atomic_write_json(
                temporary / GRPO_REFERENCE_MANIFEST_FILENAME,
                {
                    "reference": run_manifest["reference"],
                    "reference_state_sha256": run_manifest["reference_state_sha256"],
                },
            )
            write_model_manifest(
                temporary,
                runtime.model_export_config,
                stage="grpo",
                checkpoint_files=("pytorch_model.bin",),
                parent_manifest=runtime.parent_model_manifest,
                parent_checkpoint_path=config.parent_checkpoint,
                parent_base_checkpoint_path=(
                    config.parent_checkpoint.with_name(
                        run_manifest["reference"]["base_checkpoint_filename"]
                    )
                    if run_manifest["reference"]["base_checkpoint_filename"] is not None
                    else None
                ),
            )
            artifacts = _artifact_inventory(
                temporary,
                manifest_filename=GRPO_EXPORT_MANIFEST_FILENAME,
            )
            manifest = {
                "schema_version": GRPO_EXPORT_MANIFEST_SCHEMA_VERSION,
                "library": GRPO_LIBRARY_NAME,
                "stage": GRPO_STAGE_NAME,
                "run_id": run_manifest["run_id"],
                "config_sha256": run_manifest["config_sha256"],
                "dataset_sha256": run_manifest["dataset_sha256"],
                "reference": run_manifest["reference"],
                "global_step": global_step,
                "artifact_sha256": {
                    relative: _file_sha256(path) for relative, path in artifacts.items()
                },
            }
            _atomic_write_json(temporary / GRPO_EXPORT_MANIFEST_FILENAME, manifest)
            os.rename(temporary, destination)
        except BaseException as exc:
            export_error = exc
        finally:
            if temporary is not None and temporary.exists():
                try:
                    shutil.rmtree(temporary)
                except BaseException as exc:
                    if export_error is None:
                        export_error = exc
    if context.is_distributed:
        errors = [repr(export_error) if export_error is not None else None]
        dist.broadcast_object_list(errors, src=0)
        if errors[0] is not None:
            if export_error is not None:
                raise export_error
            raise GRPOStageError(f"Rank zero could not export GRPO weights: {errors[0]}")
        dist.barrier()
    elif export_error is not None:
        raise export_error
    return destination


def run_grpo_stage(
    config: GRPOStageConfig,
    *,
    runtime: GRPOStageRuntime,
    dataset: Dataset,
    dataset_identity: Mapping[str, Any],
    reference_identity: Mapping[str, Any],
    context: DistributedContext | None = None,
    device: torch.device | None = None,
) -> Path:
    """Execute one single-process or torchrun GRPO stage and return the final export.

    ``runtime`` is deliberately injected so CPU tests can exercise the complete stage with a toy
    flow policy, while the production adapter can attach the NAR-VAE codec and versioned speech
    evaluators.  One dataset row is one prompt group; candidates are created only after sharding.
    """
    context = context or DistributedContext.from_environment()
    if context.is_distributed and not dist.is_initialized():
        raise GRPOStageError("torchrun GRPO requires an initialized distributed process group.")
    if device is None:
        device = _rank_consistent_call(
            context,
            context.device,
            description="GRPO stage device initialization",
        )
    assert isinstance(device, torch.device)

    def validate_startup() -> None:
        if context.is_distributed and device.type != "cuda":
            raise GRPOStageError("Multi-GPU GRPO requires one CUDA device per torchrun process.")
        if config.mixed_precision == "bf16" and device.type != "cuda":
            raise GRPOStageError(
                "BF16 GRPO is a server-GPU option; use fp32 for CPU contract tests."
            )
        validate_prompt_group_dataset(dataset, prompt_id_column=config.prompt_id_column)
        if len(dataset) < context.world_size * config.prompt_batch_size:
            raise GRPOStageError(
                "The GRPO dataset must provide at least one complete prompt batch per rank."
            )
        if runtime.parent_model_manifest.stage != "sft":
            raise GRPOStageError("The runtime parent manifest must be the validated SFT reference.")
        _validate_reference_identity(reference_identity)

    _rank_consistent_call(context, validate_startup, description="GRPO stage validation")

    def initialize_runtime_state() -> tuple[str, torch.Generator, torch.Generator]:
        runtime.policy.to(device)
        runtime.reference_policy.to(device)
        runtime.reference_policy.eval()
        for parameter in runtime.reference_policy.parameters():
            parameter.requires_grad_(False)
        state_hash = _state_dict_fingerprint(runtime.reference_policy)
        _seed_process(config.seed, rank=context.rank, device=device)
        rollout_rng = torch.Generator(device=device)
        rollout_rng.manual_seed(config.seed + 104729 * (context.rank + 1))
        loader_rng = torch.Generator()
        loader_rng.manual_seed(config.data_seed + 130363 * (context.rank + 1))
        return state_hash, rollout_rng, loader_rng

    runtime_state = _rank_consistent_call(
        context,
        initialize_runtime_state,
        description="GRPO runtime device initialization",
    )
    assert runtime_state is not None
    reference_state_sha256, rollout_generator, loader_generator = runtime_state
    runtime_identity = {
        "parent_manifest_sha256": runtime.parent_model_manifest.sha256,
        "reference_identity_sha256": _canonical_sha256(
            _validate_reference_identity(reference_identity)
        ),
        "reference_state_sha256": reference_state_sha256,
        "model_export_config_sha256": _canonical_sha256(runtime.model_export_config),
    }
    node_runtime_identity = resolve_node_consistent_value(
        context,
        lambda: runtime_identity,
        description="GRPO runtime/reference identity",
    )
    _rank_consistent_call(
        context,
        lambda: (
            None
            if runtime_identity == node_runtime_identity
            else (_ for _ in ()).throw(
                GRPOStageError("GRPO runtime/reference identity differs within this node.")
            )
        ),
        description="GRPO runtime/reference identity agreement",
    )

    run_manifest = _initialize_run_manifest(
        config,
        dataset_identity=dataset_identity,
        reference_identity=reference_identity,
        reference_state_sha256=reference_state_sha256,
        context=context,
    )

    def build_loader() -> tuple[RankPromptSampler, DataLoader, int]:
        prompt_sampler = RankPromptSampler(
            len(dataset),
            rank=context.rank,
            world_size=context.world_size,
            seed=config.data_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.prompt_batch_size,
            sampler=prompt_sampler,
            collate_fn=runtime.collate_fn,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory and device.type == "cuda",
            drop_last=config.dataloader_drop_last,
            generator=loader_generator,
        )
        if len(loader) == 0:
            raise GRPOStageError("GRPO prompt batching produced no complete rank-local batches.")
        available_steps = len(loader) * config.epochs
        planned_steps = min(available_steps, config.max_steps or available_steps)
        if planned_steps <= 0:
            raise GRPOStageError("GRPO configuration produced no optimizer steps.")
        return prompt_sampler, loader, planned_steps

    loader_state = _rank_consistent_call(
        context,
        build_loader,
        description="GRPO data-loader construction",
    )
    assert loader_state is not None
    sampler, dataloader, total_steps = loader_state

    trainable = [parameter for parameter in runtime.policy.parameters() if parameter.requires_grad]
    _rank_consistent_call(
        context,
        lambda: (
            None
            if trainable
            else (_ for _ in ()).throw(
                GRPOStageError("The GRPO policy has no trainable parameters after freezing.")
            )
        ),
        description="GRPO trainable-parameter validation",
    )

    def wrap_policy() -> nn.Module:
        if not context.is_distributed:
            return runtime.policy
        return DistributedDataParallel(
            runtime.policy,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=config.ddp_find_unused_parameters,
        )

    policy = _rank_consistent_call(
        context,
        wrap_policy,
        description="GRPO DDP construction",
    )
    assert isinstance(policy, nn.Module)

    def build_optimizer() -> tuple[
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LambdaLR,
    ]:
        if config.optimizer == "muon":
            policy_optimizer = build_muon_optimizer(
                policy,
                {
                    "optimizer": "muon",
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "adam_beta1": config.adam_beta1,
                    "adam_beta2": config.adam_beta2,
                    "adam_epsilon": config.adam_epsilon,
                    "muon_learning_rate": (
                        config.learning_rate
                        if config.muon_learning_rate is None
                        else config.muon_learning_rate
                    ),
                    "muon_weight_decay": (
                        config.weight_decay
                        if config.muon_weight_decay is None
                        else config.muon_weight_decay
                    ),
                    "muon_momentum": config.muon_momentum,
                    "muon_nesterov": config.muon_nesterov,
                    "muon_ns_steps": config.muon_ns_steps,
                    "muon_epsilon": config.muon_epsilon,
                    "muon_adjust_lr_fn": config.muon_adjust_lr_fn,
                },
            )
        else:
            policy_trainable = [
                parameter for parameter in policy.parameters() if parameter.requires_grad
            ]
            policy_optimizer = torch.optim.AdamW(
                policy_trainable,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
            )
        policy_scheduler = _scheduler(
            policy_optimizer,
            kind=config.lr_scheduler_type,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps * config.policy_update_epochs,
        )
        return policy_optimizer, policy_scheduler

    optimizer_state = _rank_consistent_call(
        context,
        build_optimizer,
        description="GRPO optimizer/scheduler construction",
    )
    assert optimizer_state is not None
    optimizer, scheduler = optimizer_state

    resume_checkpoint = _rank_consistent_call(
        context,
        lambda: _resolve_resume_checkpoint(config, run_manifest=run_manifest),
        description="GRPO resume selection",
    )
    state = {"global_step": 0, "epoch": 0, "next_batch_index": 0}
    reference_lineage_hash = run_manifest["reference"]["model_manifest_sha256"]
    if resume_checkpoint is not None:
        resumed_state = _rank_consistent_call(
            context,
            lambda: _load_checkpoint(
                resume_checkpoint,
                policy=policy,
                reference_policy=runtime.reference_policy,
                optimizer=optimizer,
                scheduler=scheduler,
                run_manifest=run_manifest,
                context=context,
                device=device,
                rollout_generator=rollout_generator,
                loader_generator=loader_generator,
            ),
            description="GRPO checkpoint deserialization",
        )
        assert resumed_state is not None
        state = resumed_state

    trainer = _rank_consistent_call(
        context,
        lambda: FlowGRPOTrainer(
            policy=policy,
            reference_policy=runtime.reference_policy,
            optimizer=optimizer,
            velocity_adapter=runtime.velocity_adapter,
            decode=runtime.decode,
            reward=runtime.reward,
            reward_weights=config.reward_weights,
            config=config.flow_config(),
            supervised_loss=runtime.supervised_loss,
            policy_reference_manifest_sha256=(
                reference_lineage_hash if resume_checkpoint is not None else None
            ),
            reference_manifest_sha256=(
                reference_lineage_hash if resume_checkpoint is not None else None
            ),
            optimizer_step_callback=scheduler.step,
            distributed_error_synchronizer=lambda error, description: _synchronize_rank_error(
                context,
                error,
                description,
            ),
        ),
        description="GRPO trainer construction",
    )
    assert isinstance(trainer, FlowGRPOTrainer)
    logger = None
    logger_error: Exception | None = None
    try:
        logger = _WandbLogger(
            config,
            enabled=context.is_main_process,
        )
    except Exception as exc:
        logger_error = exc
    _synchronize_rank_error(context, logger_error, "W&B initialization")
    assert logger is not None
    last_checkpoint_step = int(resume_checkpoint.name.rsplit("-", 1)[1]) if resume_checkpoint else 0
    active_error: BaseException | None = None
    try:
        stop = state["global_step"] >= total_steps
        for epoch in range(state["epoch"], config.epochs):
            if stop:
                break
            sampler.set_epoch(epoch)
            restore_loader_after_iterator = (
                resume_checkpoint is not None
                and epoch == state["epoch"]
                and state["next_batch_index"] > 0
            )
            data_iterator = None
            iterator_error: Exception | None = None
            try:
                loader_state = (
                    loader_generator.get_state() if restore_loader_after_iterator else None
                )
                data_iterator = iter(dataloader)
                if loader_state is not None:
                    # Iterator construction consumes a worker seed. A mid-epoch resume must not
                    # advance the saved loader stream a second time; skipped rows are deterministic.
                    loader_generator.set_state(loader_state)
            except Exception as exc:
                iterator_error = exc
            _synchronize_rank_error(context, iterator_error, "GRPO data-iterator construction")
            assert data_iterator is not None
            for batch_index in range(len(dataloader)):
                batch = None
                loading_error: Exception | None = None
                try:
                    batch = next(data_iterator)
                    if batch is None:
                        raise GRPOStageError("The GRPO data loader returned an empty batch.")
                except StopIteration as exc:
                    loading_error = GRPOStageError(
                        "The GRPO data loader ended before its declared rank-local length."
                    )
                    loading_error.__cause__ = exc
                except Exception as exc:
                    loading_error = exc
                _synchronize_rank_error(context, loading_error, "GRPO rank-local batch loading")
                assert batch is not None
                if epoch == state["epoch"] and batch_index < state["next_batch_index"]:
                    continue
                prepared = None
                preparation_error: Exception | None = None
                try:
                    prepared = runtime.prepare_batch(
                        batch,
                        device,
                        config.group_size,
                        rollout_generator,
                    )
                    if not isinstance(prepared, GRPOPreparedBatch):
                        raise GRPOStageError("prepare_batch must return GRPOPreparedBatch.")
                    prompt_count = (
                        len(batch["prompt_ids"])
                        if isinstance(batch, Mapping) and "prompt_ids" in batch
                        else prepared.initial_state.shape[0]
                    )
                    if prepared.initial_state.shape[:2] != (
                        prompt_count,
                        config.group_size,
                    ):
                        raise GRPOStageError(
                            "prepare_batch must preserve prompt batches and create exactly "
                            "group_size candidates."
                        )
                except Exception as exc:
                    preparation_error = exc
                _synchronize_rank_error(context, preparation_error, "GRPO prompt preparation")
                assert prepared is not None
                autocast_enabled = config.mixed_precision == "bf16"
                local_metrics = None
                step_error: Exception | None = None
                try:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        local_metrics = trainer.step(
                            initial_state=prepared.initial_state,
                            conditioning=prepared.conditioning,
                            batch=prepared.trainer_batch,
                            generator=rollout_generator,
                            event_mask=prepared.event_mask,
                        )
                except Exception as exc:
                    step_error = exc
                _synchronize_rank_error(context, step_error, "GRPO optimizer update")
                assert local_metrics is not None
                metrics = reduce_grpo_metrics(local_metrics, device=device)
                state["global_step"] += 1
                state["epoch"] = epoch
                state["next_batch_index"] = batch_index + 1
                if state["next_batch_index"] >= len(dataloader):
                    state["epoch"] = epoch + 1
                    state["next_batch_index"] = 0
                if state["global_step"] % config.logging_steps == 0:
                    metrics["learning_rate"] = float(optimizer.param_groups[0]["lr"])
                    if config.optimizer == "muon":
                        metrics["optimizer/muon_enabled"] = 1.0
                        for group in optimizer.param_groups:
                            role = group.get("optimizer_role")
                            if role == "muon":
                                metrics["learning_rate/muon"] = float(group["lr"])
                            elif role in {"adamw_decay", "adamw_no_decay"}:
                                metrics.setdefault(
                                    "learning_rate/aux_adamw",
                                    float(group["lr"]),
                                )
                    logging_error: Exception | None = None
                    try:
                        logger.log(metrics, step=state["global_step"])
                    except Exception as exc:
                        logging_error = exc
                    _synchronize_rank_error(context, logging_error, "W&B metric logging")
                if state["global_step"] % config.save_steps == 0:
                    _save_checkpoint(
                        config=config,
                        runtime=runtime,
                        policy=policy,
                        reference_policy=runtime.reference_policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        run_manifest=run_manifest,
                        context=context,
                        device=device,
                        rollout_generator=rollout_generator,
                        loader_generator=loader_generator,
                    )
                    last_checkpoint_step = state["global_step"]
                if state["global_step"] >= total_steps:
                    stop = True
                    break
        if state["global_step"] == 0:
            raise GRPOStageError("The GRPO run completed without an optimizer update.")
        if last_checkpoint_step != state["global_step"]:
            _save_checkpoint(
                config=config,
                runtime=runtime,
                policy=policy,
                reference_policy=runtime.reference_policy,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
                run_manifest=run_manifest,
                context=context,
                device=device,
                rollout_generator=rollout_generator,
                loader_generator=loader_generator,
            )
        return _write_final_export(
            config=config,
            runtime=runtime,
            policy=policy,
            run_manifest=run_manifest,
            global_step=state["global_step"],
            context=context,
        )
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        finish_error: Exception | None = None
        try:
            logger.finish()
        except Exception as exc:
            finish_error = exc
        try:
            _synchronize_rank_error(context, finish_error, "W&B finalization")
        except BaseException:
            if active_error is None:
                raise


__all__ = [
    "GRPO_CHECKPOINT_MANIFEST_FILENAME",
    "GRPO_DATASET_MANIFEST_FILENAME",
    "GRPO_EXPORT_MANIFEST_FILENAME",
    "GRPO_REFERENCE_MANIFEST_FILENAME",
    "GRPO_RUN_MANIFEST_FILENAME",
    "GRPOPreparedBatch",
    "GRPOStageConfig",
    "GRPOStageError",
    "GRPOStageRuntime",
    "RankPromptSampler",
    "grpo_reference_identity",
    "load_grpo_stage_config",
    "reduce_grpo_metrics",
    "run_grpo_stage",
    "validate_prompt_group_dataset",
]
