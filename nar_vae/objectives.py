"""Versioned generative objectives for continuous DACVAE latents.

The original NAR-VAE checkpoints use a straight rectified-flow path.  New
checkpoints can instead use a variance-preserving (VP) diffusion process with
the numerically stable ``v`` parameterization.  Keeping the two names explicit
prevents a rectified-flow checkpoint from being presented as strict diffusion.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

RECTIFIED_FLOW_OBJECTIVE = "rectified_flow"
VP_DIFFUSION_OBJECTIVE = "vp_diffusion_v"
GENERATIVE_OBJECTIVES = (RECTIFIED_FLOW_OBJECTIVE, VP_DIFFUSION_OBJECTIVE)

# Only non-legacy objectives need checkpoint metadata.  Absence of the key is
# the backwards-compatible, unambiguous marker for a rectified-flow checkpoint.
GENERATIVE_OBJECTIVE_METADATA_KEY = "generative_objective_metadata"
DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY = "diffusion_schedule_shift_metadata"
VP_DIFFUSION_OBJECTIVE_VERSION = 1
DEFAULT_DIFFUSION_SCHEDULE_SHIFT = 1.0
_OBJECTIVE_TO_CODE = {VP_DIFFUSION_OBJECTIVE: VP_DIFFUSION_OBJECTIVE_VERSION}
_CODE_TO_OBJECTIVE = {value: key for key, value in _OBJECTIVE_TO_CODE.items()}


def normalize_generative_objective(value: object) -> str:
    """Return one canonical objective name or raise an actionable error."""
    if not isinstance(value, str):
        raise ValueError(
            f"generative_objective must be one of {GENERATIVE_OBJECTIVES}; "
            f"received {type(value).__name__}."
        )
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "flow": RECTIFIED_FLOW_OBJECTIVE,
        "flow_matching": RECTIFIED_FLOW_OBJECTIVE,
        "rf": RECTIFIED_FLOW_OBJECTIVE,
        "diffusion": VP_DIFFUSION_OBJECTIVE,
        "vp": VP_DIFFUSION_OBJECTIVE,
        "vp_diffusion": VP_DIFFUSION_OBJECTIVE,
        "v_prediction": VP_DIFFUSION_OBJECTIVE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in GENERATIVE_OBJECTIVES:
        raise ValueError(
            f"Unknown generative_objective {value!r}; expected one of {GENERATIVE_OBJECTIVES}."
        )
    return normalized


def objective_metadata_code(objective: object) -> int | None:
    """Return the checkpoint code, or ``None`` for legacy rectified flow."""
    return _OBJECTIVE_TO_CODE.get(normalize_generative_objective(objective))


def objective_from_metadata(value: torch.Tensor | None) -> str:
    """Resolve checkpoint metadata, treating an absent key as legacy flow."""
    if value is None:
        return RECTIFIED_FLOW_OBJECTIVE
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise ValueError("Generative-objective checkpoint metadata must be a scalar tensor.")
    raw_code = value.detach().cpu().item()
    if isinstance(raw_code, bool) or not isinstance(raw_code, Real):
        raise ValueError("Generative-objective checkpoint metadata must be an integer code.")
    numeric_code = float(raw_code)
    if not math.isfinite(numeric_code) or not numeric_code.is_integer():
        raise ValueError("Generative-objective checkpoint metadata must be an integer code.")
    code = int(numeric_code)
    try:
        return _CODE_TO_OBJECTIVE[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported generative-objective checkpoint code: {code}.") from exc


def validate_diffusion_schedule_shift(value: Real) -> float:
    """Return a finite positive log-SNR schedule scale."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("diffusion_schedule_shift must be a finite positive number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("diffusion_schedule_shift must be a finite positive number.")
    return result


def unwrap_generation_contract_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the canonical model beneath DDP/DataParallel/compile wrappers.

    Objective and schedule checks must authenticate the model that performs the
    forward pass, not optional attributes copied onto an outer wrapper.  Wrapper
    cycles are rejected instead of being treated as metadata-free legacy models.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The generative objective contract requires a torch.nn.Module.")

    current = model
    visited = {id(current)}
    while True:
        nested = None
        wrapper_attribute = None
        for attribute in ("module", "_orig_mod"):
            candidate = getattr(current, attribute, None)
            if isinstance(candidate, torch.nn.Module):
                nested = candidate
                wrapper_attribute = attribute
                break
        if nested is None:
            return current
        if id(nested) in visited:
            raise ValueError(
                "Model wrapper cycle detected while following "
                f"{wrapper_attribute!r} for the generative objective contract."
            )
        visited.add(id(nested))
        current = nested


def shifted_cosine_vp_coefficients(
    timesteps: torch.Tensor,
    schedule_shift: Real = DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean/noise coefficients for a shifted cosine VP path.

    Time follows the library's generation direction: ``t=0`` is Gaussian noise
    and ``t=1`` is clean data.  Before shifting, ``alpha=sin(pi*t/2)`` and
    ``sigma=cos(pi*t/2)``. ``schedule_shift`` adds ``2*log(schedule_shift)``
    to log-SNR and is checkpointed because training and sampling must agree.
    """
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
        raise ValueError("timesteps must be a rank-one tensor.")
    if not timesteps.is_floating_point():
        timesteps = timesteps.float()
    shift = validate_diffusion_schedule_shift(schedule_shift)
    angles = timesteps * (math.pi / 2.0)
    clean = torch.sin(angles) * shift
    noise = torch.cos(angles).clamp_min(0.0)
    normalizer = torch.sqrt(clean.square() + noise.square()).clamp_min(
        torch.finfo(clean.dtype).tiny
    )
    return clean / normalizer, noise / normalizer


def cosine_vp_coefficients(timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unshifted cosine VP coefficients (compatibility helper)."""
    return shifted_cosine_vp_coefficients(timesteps, DEFAULT_DIFFUSION_SCHEDULE_SHIFT)


def diffusion_probability_flow_scale(
    timesteps: torch.Tensor | None = None,
    schedule_shift: Real = DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
) -> float | torch.Tensor:
    """Return ``d angle / dt`` for the shifted-cosine v-prediction path.

    The scalar no-argument form preserves the unshifted ``pi/2`` result.  A
    shifted schedule has a time-varying chain-rule factor, so solver call sites
    pass their complete rank-one timestep tensor.
    """
    shift = validate_diffusion_schedule_shift(schedule_shift)
    if timesteps is None:
        if shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT:
            raise ValueError("Shifted diffusion schedules require explicit timesteps.")
        return math.pi / 2.0
    if not isinstance(timesteps, torch.Tensor) or timesteps.ndim != 1:
        raise ValueError("timesteps must be a rank-one tensor.")
    if not timesteps.is_floating_point():
        timesteps = timesteps.float()
    angles = timesteps * (math.pi / 2.0)
    denominator = torch.cos(angles).square() + (shift * torch.sin(angles)).square()
    return (math.pi / 2.0) * shift / denominator.clamp_min(torch.finfo(denominator.dtype).tiny)


__all__ = [
    "GENERATIVE_OBJECTIVES",
    "GENERATIVE_OBJECTIVE_METADATA_KEY",
    "DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY",
    "DEFAULT_DIFFUSION_SCHEDULE_SHIFT",
    "RECTIFIED_FLOW_OBJECTIVE",
    "VP_DIFFUSION_OBJECTIVE",
    "VP_DIFFUSION_OBJECTIVE_VERSION",
    "cosine_vp_coefficients",
    "diffusion_probability_flow_scale",
    "normalize_generative_objective",
    "objective_from_metadata",
    "objective_metadata_code",
    "shifted_cosine_vp_coefficients",
    "unwrap_generation_contract_model",
    "validate_diffusion_schedule_shift",
]
