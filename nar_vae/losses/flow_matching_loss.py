import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nar_vae.models.alignment import durations_from_alignment
from nar_vae.models.duration import DurationAlignmentOutput
from nar_vae.objectives import (
    DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    RECTIFIED_FLOW_OBJECTIVE,
    VP_DIFFUSION_OBJECTIVE,
    normalize_generative_objective,
    shifted_cosine_vp_coefficients,
    unwrap_generation_contract_model,
    validate_diffusion_schedule_shift,
)

_LEGACY_SIGMA_MIN = 1e-4


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _exposed_model_generation_contract(model: nn.Module) -> tuple[str, float] | None:
    """Resolve the optional contract from the canonical training model."""
    current = unwrap_generation_contract_model(model)
    missing = object()
    objective = getattr(current, "generative_objective", missing)
    schedule_shift = getattr(current, "diffusion_schedule_shift", missing)
    if objective is missing and schedule_shift is missing:
        return None
    if objective is missing or schedule_shift is missing:
        raise ValueError(
            "The training model exposes an incomplete generative objective/schedule contract."
        )
    normalized_objective = normalize_generative_objective(objective)
    normalized_shift = validate_diffusion_schedule_shift(schedule_shift)
    if (
        normalized_objective == RECTIFIED_FLOW_OBJECTIVE
        and normalized_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT
    ):
        raise ValueError(
            "A rectified-flow training model cannot declare a shifted diffusion schedule."
        )
    return normalized_objective, normalized_shift


@dataclass(frozen=True)
class AccumulationLossNormalization:
    """Valid-item totals for one complete gradient-accumulation window.

    The training DataLoader integration can reduce all four counters across ranks
    in one packed collective before the window's forward passes.  In that case
    ``globally_reduced`` is true and ``world_size`` records the DDP gradient-average
    compensation.  Direct callers may retain rank-local counts for compatibility.
    """

    velocity_elements: torch.Tensor
    examples: torch.Tensor
    text_tokens: torch.Tensor
    alignment_frames: torch.Tensor
    globally_reduced: bool = False
    world_size: int = 1


def _global_valid_mean(
    numerator: torch.Tensor,
    count: torch.Tensor,
    accumulation_count: torch.Tensor | None = None,
    *,
    accumulation_is_global: bool = False,
    normalization_world_size: int | None = None,
) -> torch.Tensor:
    """Normalize ragged losses globally while preserving correct DDP gradient scaling."""
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    if accumulation_count is not None and accumulation_is_global:
        if numerator.numel() != 1 or count.numel() != 1 or accumulation_count.numel() != 1:
            raise ValueError("Loss numerator and valid-item counts must be scalar.")
        world_size = dist.get_world_size() if distributed else 1
        if normalization_world_size is None:
            normalization_world_size = world_size
        if normalization_world_size != world_size:
            raise ValueError(
                "Accumulation normalization world size does not match the active process group."
            )
        local_count = count.to(device=numerator.device, dtype=torch.float32)
        denominator = accumulation_count.to(device=numerator.device, dtype=torch.float32)
        # These assertions enqueue device-side checks without forcing a host sync
        # on every objective and microbatch. Dataset/collator validation owns shape
        # errors; this guard retains prompt failure for nonfinite training state.
        torch._assert_async(
            torch.isfinite(numerator.detach()).all(),
            "Loss numerator must be finite.",
        )
        torch._assert_async(
            torch.isfinite(local_count).all()
            & (local_count > 0).all()
            & (local_count <= denominator).all(),
            "Local valid-item count must be finite, positive, and no greater than the "
            "window count.",
        )
        torch._assert_async(
            torch.isfinite(denominator).all() & (denominator > 0).all(),
            "Global accumulation count must be finite and positive.",
        )
        # DDP averages gradients. Multiplying each rank's local numerator by the
        # world size yields the exact derivative of the global valid-item mean,
        # without another forward-path collective. Detached global metrics are
        # intentionally reduced only at logging boundaries by the Trainer.
        return numerator * world_size / denominator

    count_error = False
    try:
        count = count.to(device=numerator.device, dtype=torch.float32)
        count_error = (
            numerator.numel() != 1
            or count.numel() != 1
            or not bool(torch.isfinite(numerator).all())
            or not bool(torch.isfinite(count).all())
            or bool((count <= 0).any())
        )
    except Exception:
        count_error = True
        count = torch.zeros((), device=numerator.device, dtype=torch.float32)
    denominator = count
    accumulation_present = accumulation_count is not None
    accumulation_error = False
    if accumulation_count is not None:
        try:
            denominator = accumulation_count.to(device=numerator.device, dtype=torch.float32)
            accumulation_error = (
                denominator.numel() != 1
                or not bool(torch.isfinite(denominator).all())
                or bool((denominator <= 0).any())
                or bool((denominator < count).any())
            )
        except Exception:
            accumulation_error = True
            denominator = torch.zeros((), device=numerator.device, dtype=torch.float32)
    if not distributed:
        if count_error:
            raise ValueError(
                "Loss numerator must be finite scalar and valid-element count finite and positive."
            )
        if accumulation_error:
            raise ValueError(
                "Accumulation-window count must be finite, positive, and no smaller than the "
                "current microbatch count."
            )
        return numerator / denominator

    # One fixed-schema verdict precedes statistics on every rank. The presence total detects a
    # divergent accumulation contract without letting one rank skip a validation collective.
    validation = torch.tensor(
        (int(count_error), int(accumulation_error), int(accumulation_present)),
        device=numerator.device,
        dtype=torch.int32,
    )
    dist.all_reduce(validation, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    if int(validation[0].item()):
        raise ValueError(
            "Loss numerator must be finite scalar and valid-element count finite and positive."
        )
    if int(validation[2].item()) not in {0, world_size}:
        raise ValueError("Accumulation normalization presence must agree across every rank.")
    if int(validation[1].item()):
        raise ValueError(
            "Accumulation-window count must be finite, positive, and no smaller than the "
            "current microbatch count."
        )

    statistics = torch.stack((numerator.detach().float(), denominator))
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    global_numerator, global_count = statistics
    if (
        not torch.isfinite(global_numerator)
        or not torch.isfinite(global_count)
        or global_count <= 0
    ):
        raise FloatingPointError("Distributed loss normalization produced invalid statistics.")
    # DDP averages parameter gradients across ranks, so compensate by world size here.
    differentiable = numerator * world_size / global_count
    global_value = global_numerator / global_count
    return differentiable + (global_value - differentiable.detach())


def sample_logit_normal(
    batch_size: int,
    device: torch.device,
    loc: float = 0.0,
    scale: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Sample from logit-normal distribution.

    The logit-normal distribution is obtained by applying the logistic sigmoid
    to a normally distributed variable. Its concentration depends on ``loc`` and
    ``scale``; the default standard normal puts more mass near the middle than a
    uniform timestep distribution.

    Args:
        batch_size: Number of samples
        device: Device to create tensor on
        loc: Mean of the underlying normal distribution
        scale: Std of the underlying normal distribution
        eps: Small value for numerical stability

    Returns:
        Samples in (0, 1)
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not _finite_real(loc) or not _finite_real(scale):
        raise ValueError("loc and scale must be finite numbers")
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not _finite_real(eps) or not 0 < eps < 0.5:
        raise ValueError("eps must be finite and satisfy 0 < eps < 0.5")
    eps = float(eps)
    # Sample from normal distribution
    u = torch.randn(batch_size, device=device) * scale + loc
    # Apply sigmoid to get logit-normal
    t = torch.sigmoid(u)
    # Clamp for numerical stability
    t = t.clamp(eps, 1.0 - eps)
    return t


def sample_stratified_logit_normal(
    batch_size: int,
    device: torch.device,
    loc: float = 0.0,
    scale: float = 1.0,
    num_strata: int = 10,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Sample from stratified logit-normal distribution.

    Stratification ensures better coverage of the timestep range by dividing
    [0, 1] into strata and sampling one point per stratum.

    Args:
        batch_size: Number of samples
        device: Device to create tensor on
        loc: Mean of the underlying normal distribution
        scale: Std of the underlying normal distribution
        num_strata: Number of strata for stratification
        eps: Small value for numerical stability

    Returns:
        Samples in (0, 1)
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(num_strata, bool) or not isinstance(num_strata, int) or num_strata <= 0:
        raise ValueError("num_strata must be a positive integer")
    if not _finite_real(loc) or not _finite_real(scale):
        raise ValueError("loc and scale must be finite numbers")
    loc = float(loc)
    scale = float(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not _finite_real(eps) or not 0 < eps < 0.5:
        raise ValueError("eps must be finite and satisfy 0 < eps < 0.5")
    eps = float(eps)

    # Stratify the *CDF probabilities*, then map them through the inverse normal
    # CDF. Applying logit to the probabilities would collapse to a uniform
    # distribution when loc=0 and scale=1 instead of producing logit-normal
    # samples.
    effective_strata = min(batch_size, num_strata)
    stratum_ids = torch.arange(batch_size, device=device) % effective_strata
    probabilities = (stratum_ids + torch.rand(batch_size, device=device)) / effective_strata
    probabilities = probabilities.clamp(eps, 1.0 - eps)
    normal_samples = (2.0**0.5) * torch.erfinv(2.0 * probabilities - 1.0)
    t = torch.sigmoid(normal_samples * scale + loc).clamp(eps, 1.0 - eps)

    # Shuffle so batch position is independent of the selected stratum.
    perm = torch.randperm(t.shape[0], device=device)
    return t[perm]


class FlowMatchingLoss(nn.Module):
    """
    Continuous-latent generative loss with versioned flow and diffusion paths.

    ``rectified_flow`` implements the legacy conditional flow-matching objective:
    - Sample timestep t ~ LogitNormal (stratified) or Uniform
    - Sample noise x_0 ~ N(0, I)
    - Interpolate: x_t = (1-t) * x_0 + t * x_1
    - Target velocity: v = x_1 - x_0
    - Loss: MSE(model(x_t, t, cond), v)

    ``vp_diffusion_v`` is strict cosine variance-preserving diffusion with
    v-prediction.  In the library's noise-to-data time direction:
    - alpha(t) = sin(pi*t/2), sigma(t) = cos(pi*t/2)
    - x_t = alpha(t) * x_1 + sigma(t) * epsilon
    - v = sigma(t) * x_1 - alpha(t) * epsilon

    Args:
        sigma_min: Fixed legacy compatibility sentinel. The straight interpolation path does
            not apply terminal noise, so values other than ``1e-4`` are rejected instead of
            silently pretending to change the objective.
        velocity_weighted: Whether to weight loss by velocity magnitude
        timestep_distribution: "uniform", "logit_normal", or "stratified_logit_normal"
        logit_normal_loc: Mean for logit-normal distribution
        logit_normal_scale: Scale for logit-normal distribution
        num_strata: Number of strata for stratified sampling
        duration_loss_weight: Weight for EchoDiT v2 log-frame Huber loss; zero disables it
        duration_huber_delta: Huber transition point for log-frame regression
        mas_duration_loss_weight: Per-token MAS duration Huber weight
        mas_alignment_loss_weight: Hard-path Gaussian likelihood weight
    """

    def __init__(
        self,
        sigma_min: float = 1e-4,
        generative_objective: str = RECTIFIED_FLOW_OBJECTIVE,
        diffusion_schedule_shift: float = DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
        velocity_weighted: bool = False,
        timestep_distribution: str = "stratified_logit_normal",
        logit_normal_loc: float = 0.0,
        logit_normal_scale: float = 1.0,
        num_strata: int = 10,
        duration_loss_weight: float = 0.0,
        duration_huber_delta: float = 1.0,
        mas_duration_loss_weight: float = 0.0,
        mas_alignment_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.generative_objective = normalize_generative_objective(generative_objective)
        self.diffusion_schedule_shift = validate_diffusion_schedule_shift(diffusion_schedule_shift)
        if (
            self.generative_objective == RECTIFIED_FLOW_OBJECTIVE
            and self.diffusion_schedule_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT
        ):
            raise ValueError("diffusion_schedule_shift is only meaningful for vp_diffusion_v.")
        if not isinstance(velocity_weighted, bool):
            raise ValueError("velocity_weighted must be a boolean")
        if timestep_distribution not in {
            "uniform",
            "logit_normal",
            "stratified_logit_normal",
        }:
            raise ValueError(f"Unknown timestep distribution: {timestep_distribution}")
        if not _finite_real(logit_normal_loc) or not _finite_real(logit_normal_scale):
            raise ValueError("Logit-normal location and scale must be finite numbers")
        logit_normal_loc = float(logit_normal_loc)
        logit_normal_scale = float(logit_normal_scale)
        if logit_normal_scale <= 0:
            raise ValueError("logit_normal_scale must be positive")
        if isinstance(num_strata, bool) or not isinstance(num_strata, int) or num_strata <= 0:
            raise ValueError("num_strata must be a positive integer")
        objective_values = {
            "duration_loss_weight": duration_loss_weight,
            "duration_huber_delta": duration_huber_delta,
            "mas_duration_loss_weight": mas_duration_loss_weight,
            "mas_alignment_loss_weight": mas_alignment_loss_weight,
        }
        if any(not _finite_real(value) for value in objective_values.values()):
            raise ValueError("Duration objective weights and delta must be finite.")
        objective_values = {name: float(value) for name, value in objective_values.items()}
        duration_loss_weight = objective_values["duration_loss_weight"]
        duration_huber_delta = objective_values["duration_huber_delta"]
        mas_duration_loss_weight = objective_values["mas_duration_loss_weight"]
        mas_alignment_loss_weight = objective_values["mas_alignment_loss_weight"]
        if (
            duration_loss_weight < 0
            or mas_duration_loss_weight < 0
            or mas_alignment_loss_weight < 0
        ):
            raise ValueError("Duration objective weights must be non-negative")
        if duration_huber_delta <= 0:
            raise ValueError("duration_huber_delta must be positive")
        if (mas_duration_loss_weight > 0) != (mas_alignment_loss_weight > 0):
            raise ValueError(
                "MAS training requires both positive per-token duration and alignment weights."
            )
        if not _finite_real(sigma_min):
            raise ValueError("sigma_min must be finite")
        sigma_min = float(sigma_min)
        if sigma_min != _LEGACY_SIGMA_MIN:
            raise ValueError(
                "sigma_min is a legacy no-op for the straight flow path and must remain 1e-4; "
                "non-default terminal-noise semantics require a versioned flow objective."
            )
        self.sigma_min = sigma_min
        self.velocity_weighted = velocity_weighted
        self.timestep_distribution = timestep_distribution
        self.logit_normal_loc = logit_normal_loc
        self.logit_normal_scale = logit_normal_scale
        self.num_strata = num_strata
        self.duration_loss_weight = duration_loss_weight
        self.duration_huber_delta = duration_huber_delta
        self.mas_duration_loss_weight = mas_duration_loss_weight
        self.mas_alignment_loss_weight = mas_alignment_loss_weight

    def forward(
        self,
        model: nn.Module,
        latents: torch.Tensor,  # [B, D, T] target latents
        conditioning_ids: torch.Tensor,  # [B, L] text token IDs
        conditioning_mask: torch.Tensor | None = None,  # [B, L]
        latent_mask: torch.Tensor | None = None,  # [B, T]
        speaker_latent: torch.Tensor | None = None,  # [B, D, T_speaker]
        speaker_mask: torch.Tensor | None = None,  # [B, T_speaker] or [B, L_speaker]
        language_ids: torch.Tensor | None = None,  # [B] target-language IDs
        token_durations: torch.Tensor | None = None,  # [B, L] fixed MAS allocation
        accumulation_normalization: AccumulationLossNormalization | None = None,
        token_language_ids: torch.Tensor | None = None,  # [B, L]
        alignment_mask: torch.Tensor | None = None,  # [B, L] speakable tokens
        conditioning_features: torch.Tensor | None = None,  # [B, L, F]
    ) -> torch.Tensor:
        """
        Compute the checkpoint's versioned generative velocity loss.

        Args:
            model: Flow matching model
            latents: Target latents [B, D, T]
            conditioning_ids: Text conditioning [B, L]
            conditioning_mask: Mask for text [B, L]
            token_language_ids: Optional per-token language IDs for code-switched text
            alignment_mask: Possibly non-contiguous subset of text tokens allowed to own frames
            conditioning_features: Optional cached frozen-backbone states aligned to the
                conditioning token axis
            latent_mask: Mask for latents [B, T]
            speaker_latent: Speaker reference latents [B, D, T_speaker] (optional)
            speaker_mask: Valid speaker frames or patches (optional)
            language_ids: Stable target-language IDs (optional for legacy English)
            token_durations: Fixed inference-graph token allocation for velocity-only replay
            accumulation_normalization: Per-objective valid-item totals across all
                microbatches in the current optimizer step, optionally reduced globally.

        Returns:
            Loss scalar
        """
        model_contract = _exposed_model_generation_contract(model)
        loss_contract = (self.generative_objective, self.diffusion_schedule_shift)
        if model_contract is not None and model_contract != loss_contract:
            raise ValueError(
                "FlowMatchingLoss generative objective/schedule does not match the training "
                f"model: loss={loss_contract!r}, model={model_contract!r}."
            )
        B, D, T = latents.shape
        device = latents.device
        if conditioning_ids.ndim != 2 or conditioning_ids.shape[0] != B:
            raise ValueError("conditioning_ids must have shape [batch, token].")
        expected_conditioning_shape = tuple(conditioning_ids.shape)
        if conditioning_features is not None:
            if (
                conditioning_features.ndim != 3
                or tuple(conditioning_features.shape[:2]) != expected_conditioning_shape
            ):
                raise ValueError(
                    "conditioning_features must have shape [batch, conditioning_token, feature]."
                )
            if conditioning_features.shape[-1] <= 0:
                raise ValueError("conditioning_features must have a positive feature width.")
            if not torch.is_floating_point(conditioning_features):
                raise TypeError("conditioning_features must use a floating-point dtype.")
        if conditioning_mask is not None:
            if tuple(conditioning_mask.shape) != expected_conditioning_shape:
                raise ValueError("conditioning_mask must have the conditioning_ids shape.")
            conditioning_mask = conditioning_mask.to(device=device, dtype=torch.bool)
        if token_language_ids is not None:
            if tuple(token_language_ids.shape) != expected_conditioning_shape:
                raise ValueError("token_language_ids must have the conditioning_ids shape.")
            token_language_ids = token_language_ids.to(device=device, dtype=torch.long)
        if alignment_mask is not None:
            if tuple(alignment_mask.shape) != expected_conditioning_shape:
                raise ValueError("alignment_mask must have the conditioning_ids shape.")
            alignment_mask = alignment_mask.to(device=device, dtype=torch.bool)
            if conditioning_mask is not None and bool((alignment_mask & ~conditioning_mask).any()):
                raise ValueError("alignment_mask cannot select padded conditioning tokens.")
            if not bool(alignment_mask.any(dim=1).all()):
                raise ValueError("Every alignment_mask row must select at least one token.")
        effective_alignment_mask = (
            alignment_mask
            if alignment_mask is not None
            else (
                conditioning_mask
                if conditioning_mask is not None
                else torch.ones_like(conditioning_ids, dtype=torch.bool, device=device)
            )
        )
        if latent_mask is not None:
            expected_shape = (B, T)
            if tuple(latent_mask.shape) != expected_shape:
                raise ValueError(
                    f"latent_mask must have shape {expected_shape}; got {tuple(latent_mask.shape)}."
                )
            latent_mask = latent_mask.to(device=device, dtype=torch.bool)

        # Sample timesteps based on distribution
        if self.timestep_distribution == "uniform":
            t = torch.rand(B, device=device)
        elif self.timestep_distribution == "logit_normal":
            t = sample_logit_normal(
                B,
                device,
                loc=self.logit_normal_loc,
                scale=self.logit_normal_scale,
            )
        elif self.timestep_distribution == "stratified_logit_normal":
            t = sample_stratified_logit_normal(
                B,
                device,
                loc=self.logit_normal_loc,
                scale=self.logit_normal_scale,
                num_strata=self.num_strata,
            )
        else:
            raise ValueError(f"Unknown timestep distribution: {self.timestep_distribution}")

        # Sample source noise. The clean/noise notation remains explicit so the
        # strict diffusion path cannot accidentally fall back to straight flow.
        x_0 = torch.randn_like(latents)
        x_1 = latents

        if self.generative_objective == RECTIFIED_FLOW_OBJECTIVE:
            x_t = (1 - t[:, None, None]) * x_0 + t[:, None, None] * x_1
            v_target = x_1 - x_0
        elif self.generative_objective == VP_DIFFUSION_OBJECTIVE:
            alpha, sigma = shifted_cosine_vp_coefficients(
                t,
                self.diffusion_schedule_shift,
            )
            alpha = alpha[:, None, None].to(dtype=latents.dtype)
            sigma = sigma[:, None, None].to(dtype=latents.dtype)
            x_t = alpha * x_1 + sigma * x_0
            # v-parameterization permits stable reconstruction of either clean
            # data or noise without dividing by a vanishing schedule term.
            v_target = sigma * x_1 - alpha * x_0
        else:  # pragma: no cover - constructor normalization is exhaustive
            raise AssertionError(f"Unhandled generative objective: {self.generative_objective}.")

        # Model prediction (use keyword arguments)
        model_kwargs = {
            "latents": x_t,
            "conditioning_ids": conditioning_ids,
            "timesteps": t,
            "attention_mask": conditioning_mask,
            "speaker_latent": speaker_latent,
            "speaker_mask": speaker_mask,
            "language_ids": language_ids,
            "use_cfg_dropout": True,
        }
        if token_language_ids is not None:
            model_kwargs["token_language_ids"] = token_language_ids
        if alignment_mask is not None:
            model_kwargs["alignment_mask"] = alignment_mask
        if conditioning_features is not None:
            model_kwargs["conditioning_features"] = conditioning_features
        if latent_mask is not None:
            model_kwargs["latent_mask"] = latent_mask
        use_mas_duration = self.mas_duration_loss_weight > 0
        if token_durations is not None:
            if use_mas_duration:
                raise ValueError(
                    "Fixed token_durations and a trainable MAS alignment objective are mutually "
                    "exclusive."
                )
            expected_duration_shape = tuple(conditioning_ids.shape)
            if tuple(token_durations.shape) != expected_duration_shape:
                raise ValueError(
                    "token_durations must have the conditioning shape "
                    f"{expected_duration_shape}; got {tuple(token_durations.shape)}."
                )
            if token_durations.dtype == torch.bool or token_durations.is_floating_point():
                raise ValueError("token_durations must use an integer dtype.")
            if torch.any(token_durations < 0):
                raise ValueError("token_durations must be nonnegative.")
            model_kwargs["token_durations"] = token_durations
        if self.duration_loss_weight > 0 or use_mas_duration:
            model_kwargs["return_duration_prediction"] = True
        if use_mas_duration:
            model_kwargs["return_duration_alignment"] = True
            model_kwargs["duration_target_latents"] = x_1
        model_output = model(**model_kwargs)
        duration_prediction = None
        if self.duration_loss_weight > 0 or use_mas_duration:
            if not isinstance(model_output, tuple) or len(model_output) != 2:
                raise RuntimeError(
                    "Duration training requires a versioned EchoDiT v2 duration output."
                )
            v_pred, duration_prediction = model_output
            total_prediction = (
                duration_prediction.total_log_frames
                if isinstance(duration_prediction, DurationAlignmentOutput)
                else duration_prediction
            )
            if tuple(total_prediction.shape) != (B,):
                raise ValueError(
                    "Duration prediction must have shape "
                    f"{(B,)}; got {tuple(total_prediction.shape)}."
                )
        else:
            if not isinstance(model_output, torch.Tensor):
                raise RuntimeError("Legacy flow loss expects a tensor velocity prediction.")
            v_pred = model_output
        if not isinstance(v_pred, torch.Tensor):
            raise RuntimeError("Flow model must return a tensor velocity prediction.")
        if tuple(v_pred.shape) != tuple(v_target.shape):
            raise ValueError(
                "Velocity prediction must have the target latent shape "
                f"{tuple(v_target.shape)}; got {tuple(v_pred.shape)}."
            )

        if accumulation_normalization is not None and not isinstance(
            accumulation_normalization,
            AccumulationLossNormalization,
        ):
            raise TypeError("accumulation_normalization must be an AccumulationLossNormalization.")

        def valid_mean(
            numerator: torch.Tensor,
            count: torch.Tensor,
            normalization_count: torch.Tensor | None,
        ) -> torch.Tensor:
            return _global_valid_mean(
                numerator,
                count,
                normalization_count,
                accumulation_is_global=(
                    accumulation_normalization.globally_reduced
                    if accumulation_normalization is not None
                    else False
                ),
                normalization_world_size=(
                    accumulation_normalization.world_size
                    if accumulation_normalization is not None
                    else None
                ),
            )

        # Compute MSE loss
        loss = F.mse_loss(v_pred, v_target, reduction="none")

        # Weight each latent position before reducing to a scalar.
        if self.velocity_weighted:
            velocity_norm = torch.linalg.vector_norm(
                v_target,
                dim=1,
                keepdim=True,
            )
            loss = loss * velocity_norm

        # Apply latent mask if provided
        if latent_mask is not None:
            # Expand mask to match latent dimensions [B, T] -> [B, D, T]
            mask = latent_mask[:, None, :].expand_as(loss)
            loss = loss * mask
            loss = valid_mean(
                loss.sum(),
                mask.sum(),
                None
                if accumulation_normalization is None
                else accumulation_normalization.velocity_elements,
            )
        else:
            loss = valid_mean(
                loss.sum(),
                torch.tensor(loss.numel(), device=device),
                None
                if accumulation_normalization is None
                else accumulation_normalization.velocity_elements,
            )

        if self.duration_loss_weight > 0 or use_mas_duration:
            if latent_mask is None:
                target_frames = torch.full((B,), T, device=device, dtype=torch.float32)
            else:
                target_frames = latent_mask.sum(dim=1, dtype=torch.float32)
                if not bool((target_frames > 0).all()):
                    raise ValueError("Every duration target must contain at least one valid frame.")
            duration_target = torch.log1p(target_frames)
            assert duration_prediction is not None
            total_prediction = (
                duration_prediction.total_log_frames
                if isinstance(duration_prediction, DurationAlignmentOutput)
                else duration_prediction
            )
            if self.duration_loss_weight > 0:
                total_errors = F.huber_loss(
                    total_prediction.float(),
                    duration_target,
                    delta=self.duration_huber_delta,
                    reduction="none",
                )
                duration_loss = valid_mean(
                    total_errors.sum(),
                    torch.tensor(B, device=device),
                    None
                    if accumulation_normalization is None
                    else accumulation_normalization.examples,
                )
                loss = loss + self.duration_loss_weight * duration_loss

        if use_mas_duration:
            if not isinstance(duration_prediction, DurationAlignmentOutput):
                raise RuntimeError("MAS objective requires a versioned DurationAlignmentOutput.")
            token_mask = effective_alignment_mask
            frame_mask = (
                latent_mask
                if latent_mask is not None
                else torch.ones((B, T), device=device, dtype=torch.bool)
            )
            expected_likelihood_shape = (B, conditioning_ids.shape[1], T)
            if tuple(duration_prediction.log_likelihoods.shape) != expected_likelihood_shape:
                raise ValueError(
                    "Alignment likelihoods must have shape "
                    f"{expected_likelihood_shape}; got "
                    f"{tuple(duration_prediction.log_likelihoods.shape)}."
                )
            if tuple(duration_prediction.token_durations.shape) != token_mask.shape:
                raise ValueError("Predicted token durations must have the conditioning shape.")
            with torch.no_grad():
                alignment = duration_prediction.hard_alignment
                if alignment.dtype != torch.bool or tuple(alignment.shape) != (
                    B,
                    conditioning_ids.shape[1],
                    T,
                ):
                    raise ValueError(
                        "MAS hard alignment must be boolean with shape "
                        f"{(B, conditioning_ids.shape[1], T)}."
                    )
                target_token_durations = durations_from_alignment(
                    alignment,
                    token_mask,
                    frame_mask,
                )

            token_errors = F.huber_loss(
                torch.log1p(duration_prediction.token_durations.float()),
                torch.log1p(target_token_durations.float()),
                delta=self.duration_huber_delta,
                reduction="none",
            )
            token_duration_loss = valid_mean(
                token_errors.masked_select(token_mask).sum(),
                token_mask.sum(),
                None
                if accumulation_normalization is None
                else accumulation_normalization.text_tokens,
            )
            alignment_loss = valid_mean(
                -duration_prediction.log_likelihoods.masked_select(alignment).sum(),
                alignment.sum(),
                None
                if accumulation_normalization is None
                else accumulation_normalization.alignment_frames,
            )
            loss = (
                loss
                + self.mas_duration_loss_weight * token_duration_loss
                + self.mas_alignment_loss_weight * alignment_loss
            )

        return loss
