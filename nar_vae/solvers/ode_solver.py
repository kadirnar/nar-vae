import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from nar_vae.caching.cache_dit import assert_cache_dit_healthy
from nar_vae.configuration import cfg_guidance_active
from nar_vae.objectives import (
    DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    RECTIFIED_FLOW_OBJECTIVE,
    VP_DIFFUSION_OBJECTIVE,
    diffusion_probability_flow_scale,
    normalize_generative_objective,
    shifted_cosine_vp_coefficients,
    unwrap_generation_contract_model,
    validate_diffusion_schedule_shift,
)

if TYPE_CHECKING:
    from nar_vae.caching import SCMContext


def temporal_score_rescaling(v: torch.Tensor, k: float = 1.0, sigma: float = 3.0) -> torch.Tensor:
    if (
        isinstance(k, bool)
        or isinstance(sigma, bool)
        or not isinstance(k, (int, float))
        or not isinstance(sigma, (int, float))
        or not math.isfinite(k)
        or not math.isfinite(sigma)
        or k <= 0
        or sigma <= 0
    ):
        raise ValueError("Temporal rescale k and sigma must be finite positive numbers.")
    if k == 1.0:
        return v
    std = v.std(dim=-1, keepdim=True)
    rescale_factor = k * (std + sigma / k) / (std + sigma + 1e-8)
    return v * rescale_factor


class ODESolver:
    @classmethod
    @torch.no_grad()
    def sample(
        cls,
        model: nn.Module,
        conditioning_ids: torch.Tensor,
        num_steps: int = 30,
        latent_shape: tuple = (1, 128, 640),
        solver: str = "euler",
        cfg_scale: float = 1.0,
        cfg_mode: str = "joint",
        cfg_scale_text: float | None = None,
        cfg_scale_speaker: float | None = None,
        cfg_min_t: float = 0.0,
        cfg_max_t: float = 1.0,
        initial_noise_scale: float = 1.0,
        temporal_rescale_k: float = 1.0,
        temporal_rescale_sigma: float = 3.0,
        conditioning_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        show_progress: bool = False,
        device: torch.device | None = None,
        target_latent_std: float | None = None,
        scm_ctx: Optional["SCMContext"] = None,
        step_callback: Callable[[int, torch.Tensor], None] | None = None,
        integration_start_callback: Callable[[], None] | None = None,
        fuse_cfg_branches: bool = False,
        language_ids: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        token_durations: torch.Tensor | None = None,
        prepared_conditioning: object | None = None,
        prepared_cfg_conditioning: object | None = None,
        generative_objective: str | None = None,
        diffusion_schedule_shift: float | None = None,
    ) -> torch.Tensor:
        # A failed Cache-DiT teardown can leave third-party forward hooks or
        # monkey patches installed. Never run that model again, even through an
        # otherwise uncached solver path.
        assert_cache_dit_healthy(model)
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive non-boolean integer.")
        if solver not in {"ddim", "euler", "midpoint", "heun", "rk4"}:
            raise ValueError("solver must be 'ddim', 'euler', 'midpoint', 'heun', or 'rk4'.")
        if cfg_mode not in {"joint", "independent", "alternating"}:
            raise ValueError("cfg_mode must be 'joint', 'independent', or 'alternating'.")
        numerical_fields = {
            "cfg_scale": cfg_scale,
            "cfg_min_t": cfg_min_t,
            "cfg_max_t": cfg_max_t,
            "initial_noise_scale": initial_noise_scale,
            "temporal_rescale_k": temporal_rescale_k,
            "temporal_rescale_sigma": temporal_rescale_sigma,
        }
        if cfg_scale_text is not None:
            numerical_fields["cfg_scale_text"] = cfg_scale_text
        if cfg_scale_speaker is not None:
            numerical_fields["cfg_scale_speaker"] = cfg_scale_speaker
        if target_latent_std is not None:
            numerical_fields["target_latent_std"] = target_latent_std
        invalid_numerics = []
        for name, value in numerical_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                invalid_numerics.append(name)
        if invalid_numerics:
            raise ValueError(f"ODE numerical options must be finite numbers: {invalid_numerics}.")
        guidance_scales = [cfg_scale]
        if cfg_scale_text is not None:
            guidance_scales.append(cfg_scale_text)
        if cfg_scale_speaker is not None:
            guidance_scales.append(cfg_scale_speaker)
        if min(guidance_scales) < 0:
            raise ValueError("CFG scales must be nonnegative.")
        if not 0 <= cfg_min_t <= cfg_max_t <= 1:
            raise ValueError("CFG bounds must satisfy 0 <= cfg_min_t <= cfg_max_t <= 1.")
        if initial_noise_scale <= 0:
            raise ValueError("initial_noise_scale must be positive.")
        if temporal_rescale_k <= 0 or temporal_rescale_sigma <= 0:
            raise ValueError("Temporal rescale k and sigma must be positive.")
        if target_latent_std is not None and target_latent_std <= 0:
            raise ValueError("target_latent_std must be positive when provided.")
        contract_model = unwrap_generation_contract_model(model)
        missing_contract_value = object()
        model_objective_value = getattr(
            contract_model,
            "generative_objective",
            missing_contract_value,
        )
        model_schedule_shift_value = getattr(
            contract_model,
            "diffusion_schedule_shift",
            missing_contract_value,
        )
        if (model_objective_value is missing_contract_value) != (
            model_schedule_shift_value is missing_contract_value
        ):
            raise ValueError(
                "The diffusion model exposes an incomplete generative objective/schedule contract."
            )
        model_contract_present = model_objective_value is not missing_contract_value
        model_objective = (
            normalize_generative_objective(model_objective_value)
            if model_contract_present
            else None
        )
        model_schedule_shift = (
            validate_diffusion_schedule_shift(model_schedule_shift_value)
            if model_contract_present
            else None
        )
        if (
            model_objective == RECTIFIED_FLOW_OBJECTIVE
            and model_schedule_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT
        ):
            raise ValueError(
                "A rectified-flow diffusion model cannot declare a shifted diffusion schedule."
            )
        objective = normalize_generative_objective(
            generative_objective
            if generative_objective is not None
            else (model_objective or RECTIFIED_FLOW_OBJECTIVE)
        )
        if model_objective is not None and objective != model_objective:
            raise ValueError(
                "The requested generative objective does not match the loaded model: "
                f"requested={objective!r}, model={model_objective!r}."
            )
        schedule_shift = validate_diffusion_schedule_shift(
            diffusion_schedule_shift
            if diffusion_schedule_shift is not None
            else (
                model_schedule_shift
                if model_schedule_shift is not None
                else DEFAULT_DIFFUSION_SCHEDULE_SHIFT
            )
        )
        if (
            model_schedule_shift is not None
            and diffusion_schedule_shift is not None
            and schedule_shift != model_schedule_shift
        ):
            raise ValueError(
                "The requested diffusion schedule shift does not match the loaded model."
            )
        if (
            objective == RECTIFIED_FLOW_OBJECTIVE
            and schedule_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT
        ):
            raise ValueError("Rectified-flow sampling cannot use a shifted diffusion schedule.")
        if solver == "ddim" and objective != VP_DIFFUSION_OBJECTIVE:
            raise ValueError("solver='ddim' requires a vp_diffusion_v checkpoint.")
        if solver == "ddim" and temporal_rescale_k != 1.0:
            raise ValueError("Temporal score rescaling is not defined for the DDIM update.")
        if solver == "ddim" and initial_noise_scale != 1.0:
            raise ValueError(
                "solver='ddim' requires initial_noise_scale=1.0 to preserve the trained VP prior."
            )
        if solver == "ddim" and target_latent_std is not None:
            raise ValueError(
                "target_latent_std is an untrained post-hoc transform and is not supported "
                "by the DDIM path."
            )
        if solver == "ddim" and scm_ctx is not None:
            raise ValueError(
                "SCM/step caching has not been validated for analytic DDIM updates; "
                "disable the cache or use an ODE solver."
            )
        guidance_active = cfg_guidance_active(
            cfg_scale=cfg_scale,
            cfg_mode=cfg_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
        )
        if (
            not isinstance(latent_shape, tuple)
            or len(latent_shape) != 3
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in latent_shape
            )
        ):
            raise ValueError(
                "latent_shape must be a positive integer [batch, channels, frames] tuple."
            )
        if not isinstance(conditioning_ids, torch.Tensor) or conditioning_ids.ndim != 2:
            raise ValueError("conditioning_ids must be a rank-two tensor.")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if (
            conditioning_ids.dtype not in integer_dtypes
            or conditioning_ids.shape[0] != latent_shape[0]
        ):
            raise ValueError(
                "conditioning_ids must contain integer IDs and match latent_shape batch size."
            )
        if conditioning_ids.shape[1] <= 0:
            raise ValueError("conditioning_ids must contain at least one text token.")
        parameters = tuple(model.parameters())
        if not parameters:
            raise ValueError("The diffusion model must expose at least one floating parameter.")
        state_parameter = parameters[0]
        supported_state_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        if (
            not torch.is_floating_point(state_parameter)
            or state_parameter.dtype not in supported_state_dtypes
        ):
            raise ValueError(
                "The diffusion model's state parameter must use float16, bfloat16, "
                "float32, or float64."
            )
        state_dtype = state_parameter.dtype
        state_device = state_parameter.device
        if any(
            not torch.is_floating_point(parameter)
            or parameter.dtype != state_dtype
            or parameter.device != state_device
            for parameter in parameters[1:]
        ):
            raise ValueError(
                "All diffusion-model parameters must share one floating dtype and device."
            )
        if device is None:
            device = state_device
        else:
            requested_device = torch.device(device)
            if requested_device.type != state_device.type or (
                requested_device.index is not None and requested_device.index != state_device.index
            ):
                raise ValueError(
                    "The requested solver device must match the diffusion model device: "
                    f"{requested_device} != {state_device}."
                )
            # Resolve an index-less CUDA request (``cuda``) to the exact model
            # device (for example ``cuda:0``) before allocating solver state.
            device = state_device

        batch_size = latent_shape[0]

        # Start from noise
        x = torch.randn(latent_shape, device=device, dtype=state_dtype) * initial_noise_scale

        # Move to device
        conditioning_ids = conditioning_ids.to(device)
        if conditioning_features is not None:
            if conditioning_features.ndim != 3 or tuple(conditioning_features.shape[:2]) != tuple(
                conditioning_ids.shape
            ):
                raise ValueError(
                    "conditioning_features must have shape [batch, token, feature] and share "
                    "the conditioning_ids token axes."
                )
            if not torch.is_floating_point(conditioning_features):
                raise TypeError("conditioning_features must use a floating-point dtype.")
            conditioning_features = conditioning_features.to(
                device=device,
                dtype=state_dtype,
            )
        if language_ids is not None:
            if language_ids.ndim != 1 or language_ids.shape[0] != batch_size:
                raise ValueError("language_ids must have shape [batch].")
            language_ids = language_ids.to(device=device, dtype=torch.long)
        if conditioning_mask is not None:
            if tuple(conditioning_mask.shape) != tuple(conditioning_ids.shape):
                raise ValueError("conditioning_mask must have the conditioning_ids shape.")
            conditioning_mask = conditioning_mask.to(device=device, dtype=torch.bool)
        if token_language_ids is not None:
            if tuple(token_language_ids.shape) != tuple(conditioning_ids.shape):
                raise ValueError("token_language_ids must have the conditioning_ids shape.")
            token_language_ids = token_language_ids.to(device=device, dtype=torch.long)
        if alignment_mask is not None:
            if tuple(alignment_mask.shape) != tuple(conditioning_ids.shape):
                raise ValueError("alignment_mask must have the conditioning_ids shape.")
            alignment_mask = alignment_mask.to(device=device, dtype=torch.bool)
            if conditioning_mask is not None and bool((alignment_mask & ~conditioning_mask).any()):
                raise ValueError("alignment_mask cannot enable padded conditioning tokens.")
            if not bool(alignment_mask.any(dim=1).all()):
                raise ValueError("Every row must contain at least one alignable text token.")
        if token_durations is not None:
            if tuple(token_durations.shape) != tuple(conditioning_ids.shape):
                raise ValueError("token_durations must have the conditioning_ids shape.")
            if token_durations.dtype == torch.bool or token_durations.is_complex():
                raise TypeError("token_durations must contain nonnegative integers.")
            token_durations = token_durations.to(device=device)
            if torch.is_floating_point(token_durations):
                if not bool(torch.isfinite(token_durations).all()) or not torch.equal(
                    token_durations, token_durations.round()
                ):
                    raise ValueError("token_durations must contain finite integers.")
            token_durations = token_durations.to(dtype=torch.long)
            if bool((token_durations < 0).any()):
                raise ValueError("token_durations must be nonnegative.")
            expected_frames = int(latent_shape[-1])
            if not bool((token_durations.sum(dim=1) == expected_frames).all()):
                raise ValueError(
                    "Every token_durations row must sum to the generated latent frame count."
                )
            duration_mask = alignment_mask if alignment_mask is not None else conditioning_mask
            if duration_mask is not None:
                if bool((token_durations.masked_select(~duration_mask) != 0).any()):
                    raise ValueError("Non-alignable conditioning tokens must have zero duration.")
                if bool((token_durations.masked_select(duration_mask) <= 0).any()):
                    raise ValueError(
                        "Every alignable conditioning token must have positive duration."
                    )
        if speaker_latent is not None:
            if speaker_latent.ndim != 3:
                raise ValueError("speaker_latent must have shape [batch, latent_channels, frames].")
            if speaker_latent.shape[0] != batch_size:
                raise ValueError(
                    "speaker_latent batch size must match the generated latent batch size."
                )
            if speaker_latent.shape[-1] == 0:
                raise ValueError("speaker_latent must contain at least one frame.")
            if not torch.isfinite(speaker_latent).all():
                raise ValueError("speaker_latent contains non-finite values.")
            speaker_latent = speaker_latent.to(device=device, dtype=state_dtype)
        if speaker_mask is not None:
            if speaker_latent is None:
                raise ValueError("speaker_mask requires speaker_latent.")
            if speaker_mask.ndim != 2 or speaker_mask.shape[0] != batch_size:
                raise ValueError("speaker_mask must have shape [batch, frames or patches].")
            speaker_mask = speaker_mask.to(device=device, dtype=torch.bool)

        # Text and speaker encoders are invariant across an ODE trajectory. Prefer
        # the prepared-conditioning protocol whenever a model implements it, even
        # when CFG branches intentionally remain sequential for lower peak memory.
        # Simple/legacy callable models continue through the original forward path.
        prepare_conditioning = getattr(model, "prepare_inference_conditioning", None)
        prepare_cfg = getattr(model, "prepare_fused_cfg_conditioning", None)
        forward_prepared = getattr(model, "forward_prepared", None)
        forward_with_prepared_cfg = getattr(model, "forward_with_prepared_cfg", None)

        # Do not silently remove training-mode CFG dropout for callers that pass
        # a model still in train mode. Runtime inference models are put in eval
        # mode; the explicit fused flag retains its historical opt-in behavior.
        allow_prepared_conditioning = fuse_cfg_branches or not getattr(model, "training", False)
        if not allow_prepared_conditioning and (
            prepared_conditioning is not None or prepared_cfg_conditioning is not None
        ):
            raise ValueError("Externally prepared conditioning requires an eval-mode model.")

        expected_layouts = {
            "joint": (2, 1),
            "independent": (3, 1),
            "alternating": (2, 2),
        }

        def has_cfg_protocol(candidate: object) -> bool:
            candidate_variants = getattr(candidate, "variants", ())
            expected_layout = expected_layouts[cfg_mode]
            return bool(
                getattr(candidate, "mode", None) == cfg_mode
                and isinstance(candidate_variants, tuple)
                and int(getattr(candidate, "branch_count", 0)) == expected_layout[0]
                and len(candidate_variants) == expected_layout[1]
                and hasattr(candidate, "conditional")
                and all(
                    callable(getattr(variant, "slice_batch", None))
                    for variant in candidate_variants
                )
            )

        if prepared_conditioning is not None and not callable(forward_prepared):
            raise ValueError("prepared_conditioning requires model.forward_prepared().")
        if prepared_cfg_conditioning is not None:
            if not has_cfg_protocol(prepared_cfg_conditioning):
                raise ValueError(
                    "prepared_cfg_conditioning does not match the requested CFG mode/layout."
                )
            if not callable(forward_prepared) or (
                fuse_cfg_branches and not callable(forward_with_prepared_cfg)
            ):
                raise ValueError(
                    "prepared_cfg_conditioning requires the model prepared-CFG protocol."
                )
            prepared_conditioning = prepared_cfg_conditioning.conditional

        can_prepare_cfg = (
            allow_prepared_conditioning and callable(prepare_cfg) and callable(forward_prepared)
        )
        if fuse_cfg_branches:
            can_prepare_cfg = can_prepare_cfg and callable(forward_with_prepared_cfg)
        if guidance_active and prepared_cfg_conditioning is None and can_prepare_cfg:
            prepare_cfg_kwargs = {
                "cfg_mode": cfg_mode,
                "speaker_mask": speaker_mask,
            }
            if language_ids is not None:
                prepare_cfg_kwargs["language_ids"] = language_ids
            if token_language_ids is not None:
                prepare_cfg_kwargs["token_language_ids"] = token_language_ids
            if alignment_mask is not None:
                prepare_cfg_kwargs["alignment_mask"] = alignment_mask
            if token_durations is not None:
                prepare_cfg_kwargs["token_durations"] = token_durations
            if conditioning_features is not None:
                prepare_cfg_kwargs["conditioning_features"] = conditioning_features
            candidate = prepare_cfg(
                conditioning_ids,
                conditioning_mask,
                speaker_latent,
                **prepare_cfg_kwargs,
            )
            if has_cfg_protocol(candidate):
                prepared_cfg_conditioning = candidate
                prepared_conditioning = candidate.conditional

        if (
            prepared_conditioning is None
            and allow_prepared_conditioning
            and callable(prepare_conditioning)
            and callable(forward_prepared)
        ):
            prepare_kwargs = {}
            if language_ids is not None:
                prepare_kwargs["language_ids"] = language_ids
            if token_language_ids is not None:
                prepare_kwargs["token_language_ids"] = token_language_ids
            if alignment_mask is not None:
                prepare_kwargs["alignment_mask"] = alignment_mask
            if token_durations is not None:
                prepare_kwargs["token_durations"] = token_durations
            if conditioning_features is not None:
                prepare_kwargs["conditioning_features"] = conditioning_features
            prepared_conditioning = prepare_conditioning(
                conditioning_ids,
                conditioning_mask,
                speaker_latent,
                speaker_mask,
                **prepare_kwargs,
            )

        def sequential_prepared_cfg(
            x_in: torch.Tensor,
            t_in: torch.Tensor,
            *,
            step_idx: int,
        ) -> torch.Tensor:
            """Evaluate cached CFG branches separately to bound peak activation memory."""
            assert prepared_cfg_conditioning is not None
            assert callable(forward_prepared)
            branch_count = int(prepared_cfg_conditioning.branch_count)
            variant_index = step_idx % 2 if prepared_cfg_conditioning.mode == "alternating" else 0
            variant = prepared_cfg_conditioning.variants[variant_index]
            current_batch = x_in.shape[0]
            branches = tuple(
                forward_prepared(
                    x_in,
                    t_in,
                    variant.slice_batch(
                        branch_index * current_batch,
                        (branch_index + 1) * current_batch,
                    ),
                )
                for branch_index in range(branch_count)
            )
            text_scale = cfg_scale if cfg_scale_text is None else cfg_scale_text
            speaker_scale = cfg_scale if cfg_scale_speaker is None else cfg_scale_speaker
            if prepared_cfg_conditioning.mode == "joint":
                conditional, unconditional = branches
                return unconditional + cfg_scale * (conditional - unconditional)
            if prepared_cfg_conditioning.mode == "independent":
                conditional, unconditional_text, unconditional_speaker = branches
                return (
                    conditional
                    + text_scale * (conditional - unconditional_text)
                    + speaker_scale * (conditional - unconditional_speaker)
                )
            if prepared_cfg_conditioning.mode == "alternating":
                conditional, unconditional = branches
                scale = text_scale if step_idx % 2 == 0 else speaker_scale
                return unconditional + scale * (conditional - unconditional)
            raise ValueError(f"Unknown CFG mode: {prepared_cfg_conditioning.mode}")

        def raw_model_velocity(
            x_in: torch.Tensor,
            t_in: torch.Tensor,
            *,
            step_idx: int,
            apply_cfg: bool,
        ) -> torch.Tensor:
            if apply_cfg and prepared_cfg_conditioning is not None:
                if fuse_cfg_branches:
                    assert callable(forward_with_prepared_cfg)
                    return forward_with_prepared_cfg(
                        x_in,
                        t_in,
                        prepared_cfg_conditioning,
                        cfg_scale=cfg_scale,
                        cfg_scale_text=cfg_scale_text,
                        cfg_scale_speaker=cfg_scale_speaker,
                        step_idx=step_idx,
                    )
                return sequential_prepared_cfg(x_in, t_in, step_idx=step_idx)
            if not apply_cfg and prepared_conditioning is not None:
                assert callable(forward_prepared)
                return forward_prepared(x_in, t_in, prepared_conditioning)
            if apply_cfg:
                cfg_kwargs = {}
                if fuse_cfg_branches:
                    cfg_kwargs["fuse_cfg_branches"] = True
                if language_ids is not None:
                    cfg_kwargs["language_ids"] = language_ids
                if token_language_ids is not None:
                    cfg_kwargs["token_language_ids"] = token_language_ids
                if alignment_mask is not None:
                    cfg_kwargs["alignment_mask"] = alignment_mask
                if token_durations is not None:
                    cfg_kwargs["token_durations"] = token_durations
                if conditioning_features is not None:
                    cfg_kwargs["conditioning_features"] = conditioning_features
                return model.forward_with_cfg(
                    x_in,
                    conditioning_ids,
                    t_in,
                    cfg_scale=cfg_scale,
                    attention_mask=conditioning_mask,
                    speaker_latent=speaker_latent,
                    cfg_mode=cfg_mode,
                    cfg_scale_text=cfg_scale_text,
                    cfg_scale_speaker=cfg_scale_speaker,
                    step_idx=step_idx,
                    speaker_mask=speaker_mask,
                    **cfg_kwargs,
                )
            if speaker_latent is None:
                # Preserve compatibility with simple callable models that
                # implement only the original four-argument protocol.
                if (
                    language_ids is not None
                    or token_language_ids is not None
                    or alignment_mask is not None
                    or conditioning_features is not None
                ):
                    model_kwargs = {}
                    if language_ids is not None:
                        model_kwargs["language_ids"] = language_ids
                    if token_language_ids is not None:
                        model_kwargs["token_language_ids"] = token_language_ids
                    if alignment_mask is not None:
                        model_kwargs["alignment_mask"] = alignment_mask
                    if token_durations is not None:
                        model_kwargs["token_durations"] = token_durations
                    if conditioning_features is not None:
                        model_kwargs["conditioning_features"] = conditioning_features
                    return model(
                        latents=x_in,
                        conditioning_ids=conditioning_ids,
                        timesteps=t_in,
                        attention_mask=conditioning_mask,
                        **model_kwargs,
                    )
                if token_durations is not None:
                    return model(
                        latents=x_in,
                        conditioning_ids=conditioning_ids,
                        timesteps=t_in,
                        attention_mask=conditioning_mask,
                        token_durations=token_durations,
                    )
                return model(x_in, conditioning_ids, t_in, conditioning_mask)
            if speaker_mask is None:
                if (
                    language_ids is not None
                    or token_language_ids is not None
                    or alignment_mask is not None
                    or conditioning_features is not None
                ):
                    model_kwargs = {}
                    if language_ids is not None:
                        model_kwargs["language_ids"] = language_ids
                    if token_language_ids is not None:
                        model_kwargs["token_language_ids"] = token_language_ids
                    if alignment_mask is not None:
                        model_kwargs["alignment_mask"] = alignment_mask
                    if token_durations is not None:
                        model_kwargs["token_durations"] = token_durations
                    if conditioning_features is not None:
                        model_kwargs["conditioning_features"] = conditioning_features
                    return model(
                        latents=x_in,
                        conditioning_ids=conditioning_ids,
                        timesteps=t_in,
                        attention_mask=conditioning_mask,
                        speaker_latent=speaker_latent,
                        use_cfg_dropout=False,
                        **model_kwargs,
                    )
                if token_durations is not None:
                    return model(
                        latents=x_in,
                        conditioning_ids=conditioning_ids,
                        timesteps=t_in,
                        attention_mask=conditioning_mask,
                        speaker_latent=speaker_latent,
                        use_cfg_dropout=False,
                        token_durations=token_durations,
                    )
                return model(
                    x_in,
                    conditioning_ids,
                    t_in,
                    conditioning_mask,
                    speaker_latent,
                    False,
                )
            model_kwargs = {
                "latents": x_in,
                "conditioning_ids": conditioning_ids,
                "timesteps": t_in,
                "attention_mask": conditioning_mask,
                "speaker_latent": speaker_latent,
                "use_cfg_dropout": False,
                "speaker_mask": speaker_mask,
            }
            if language_ids is not None:
                model_kwargs["language_ids"] = language_ids
            if token_language_ids is not None:
                model_kwargs["token_language_ids"] = token_language_ids
            if alignment_mask is not None:
                model_kwargs["alignment_mask"] = alignment_mask
            if token_durations is not None:
                model_kwargs["token_durations"] = token_durations
            if conditioning_features is not None:
                model_kwargs["conditioning_features"] = conditioning_features
            return model(**model_kwargs)

        def model_velocity(
            x_in: torch.Tensor,
            t_in: torch.Tensor,
            *,
            step_idx: int,
            apply_cfg: bool,
        ) -> torch.Tensor:
            """Evaluate one model state while preserving the solver-state dtype."""
            prediction = raw_model_velocity(
                x_in,
                t_in.to(device=x_in.device, dtype=state_dtype),
                step_idx=step_idx,
                apply_cfg=apply_cfg,
            )
            if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != tuple(
                x_in.shape
            ):
                actual_shape = (
                    None if not isinstance(prediction, torch.Tensor) else tuple(prediction.shape)
                )
                raise ValueError(
                    "The diffusion model prediction must preserve the latent state shape: "
                    f"{actual_shape} != {tuple(x_in.shape)}."
                )
            if not torch.is_floating_point(prediction):
                raise TypeError("The diffusion model prediction must use a floating dtype.")
            return prediction.to(device=x_in.device, dtype=state_dtype)

        def probability_flow_derivative(
            prediction: torch.Tensor,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            """Convert the checkpoint prediction into the solver's dx/dt field."""
            if objective != VP_DIFFUSION_OBJECTIVE:
                return prediction
            scale = diffusion_probability_flow_scale(timestep, schedule_shift)
            assert isinstance(scale, torch.Tensor)
            return prediction * scale.to(device=prediction.device, dtype=prediction.dtype).view(
                -1, 1, 1
            )

        # Invariant text/speaker encoders and their projected KV caches are part
        # of request conditioning, not iterative ODE work.  Realtime callers use
        # this boundary to report mutually exclusive, end-to-end stage timings.
        # Keep the callback optional so ordinary solver users pay no timing or
        # synchronization cost.
        if integration_start_callback is not None:
            integration_start_callback()

        dt = 1.0 / num_steps

        # Pre-compute timesteps on GPU
        schedule_dtype = torch.float64 if state_dtype == torch.float64 else torch.float32
        timesteps = torch.linspace(
            0,
            1 - dt,
            num_steps,
            device=device,
            dtype=schedule_dtype,
        )
        half_timesteps = timesteps + dt / 2 if solver in ("midpoint", "rk4") else None
        next_timesteps = timesteps + dt if solver in ("ddim", "heun", "rk4") else None

        iterator = range(num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating")

        # Reset SCM context if provided
        if scm_ctx is not None:
            scm_ctx.reset()

        if solver == "ddim":
            assert next_timesteps is not None
            # Deterministic DDIM update for the model's v-prediction.  Unlike a
            # generic ODE step, this moves analytically between two points on the
            # shifted VP path and therefore does not use the d-angle/dt factor.
            for i in iterator:
                t = i * dt
                t_tensor = timesteps[i].unsqueeze(0).expand(batch_size)
                apply_cfg = cfg_min_t <= t <= cfg_max_t and guidance_active

                prediction = model_velocity(
                    x,
                    t_tensor,
                    step_idx=i,
                    apply_cfg=apply_cfg,
                )

                t_next_tensor = next_timesteps[i].unsqueeze(0).expand(batch_size)
                alpha, sigma = shifted_cosine_vp_coefficients(t_tensor, schedule_shift)
                alpha_next, sigma_next = shifted_cosine_vp_coefficients(
                    t_next_tensor,
                    schedule_shift,
                )
                alpha = alpha.to(device=x.device, dtype=x.dtype).view(-1, 1, 1)
                sigma = sigma.to(device=x.device, dtype=x.dtype).view(-1, 1, 1)
                alpha_next = alpha_next.to(device=x.device, dtype=x.dtype).view(-1, 1, 1)
                sigma_next = sigma_next.to(device=x.device, dtype=x.dtype).view(-1, 1, 1)
                clean_prediction = alpha * x + sigma * prediction
                noise_prediction = sigma * x - alpha * prediction
                x = alpha_next * clean_prediction + sigma_next * noise_prediction

                if step_callback is not None:
                    step_callback(i, x)

        elif solver == "euler":
            # Optimized Euler: pre-computed timesteps with SCM-based step skipping
            for i in iterator:
                # Keep the CFG decision on the host. Calling ``.item()`` on the
                # CUDA timestep here serialized every solver step with the CPU.
                t = i * dt
                t_tensor = timesteps[i].unsqueeze(0).expand(batch_size)

                apply_cfg = cfg_min_t <= t <= cfg_max_t and guidance_active

                if scm_ctx is not None and not scm_ctx.is_compute_step(i):
                    # Cache step: predict velocity, skip model forward entirely
                    v = scm_ctx.predictor.predict()
                    scm_ctx.cached_steps += 1
                else:
                    # Compute step: run model normally
                    v = probability_flow_derivative(
                        model_velocity(
                            x,
                            t_tensor,
                            step_idx=i,
                            apply_cfg=apply_cfg,
                        ),
                        t_tensor,
                    )
                    # Update predictor with computed velocity
                    if scm_ctx is not None:
                        scm_ctx.predictor.update(v)

                if temporal_rescale_k != 1.0:
                    v = temporal_score_rescaling(
                        v, k=temporal_rescale_k, sigma=temporal_rescale_sigma
                    )

                if scm_ctx is not None:
                    scm_ctx.total_steps += 1

                x = x + dt * v
                if step_callback is not None:
                    step_callback(i, x)

        elif solver == "midpoint":
            assert half_timesteps is not None
            for i in iterator:
                t = i * dt
                t_tensor = timesteps[i].expand(batch_size)
                t_half_tensor = half_timesteps[i].expand(batch_size)

                apply_cfg = cfg_min_t <= t <= cfg_max_t and guidance_active
                v1 = probability_flow_derivative(
                    model_velocity(
                        x,
                        t_tensor,
                        step_idx=i,
                        apply_cfg=apply_cfg,
                    ),
                    t_tensor,
                )
                if temporal_rescale_k != 1.0:
                    v1 = temporal_score_rescaling(
                        v1, k=temporal_rescale_k, sigma=temporal_rescale_sigma
                    )

                t_half = t + dt / 2
                apply_cfg_half = cfg_min_t <= t_half <= cfg_max_t and guidance_active
                v2 = probability_flow_derivative(
                    model_velocity(
                        x + dt / 2 * v1,
                        t_half_tensor,
                        step_idx=i,
                        apply_cfg=apply_cfg_half,
                    ),
                    t_half_tensor,
                )
                if temporal_rescale_k != 1.0:
                    v2 = temporal_score_rescaling(
                        v2, k=temporal_rescale_k, sigma=temporal_rescale_sigma
                    )

                x = x + dt * v2
                if step_callback is not None:
                    step_callback(i, x)

        elif solver == "heun":
            assert next_timesteps is not None
            # Note: Caching not implemented for Heun solver (predictor-corrector)
            for i in iterator:
                t = i * dt
                t_tensor = timesteps[i].expand(batch_size)
                t_next_tensor = next_timesteps[i].expand(batch_size)

                apply_cfg = cfg_min_t <= t <= cfg_max_t and guidance_active

                # Predictor
                v1 = probability_flow_derivative(
                    model_velocity(
                        x,
                        t_tensor,
                        step_idx=i,
                        apply_cfg=apply_cfg,
                    ),
                    t_tensor,
                )

                if temporal_rescale_k != 1.0:
                    v1 = temporal_score_rescaling(
                        v1, k=temporal_rescale_k, sigma=temporal_rescale_sigma
                    )

                x_pred = x + dt * v1

                # Corrector
                t_next = t + dt
                apply_cfg_next = cfg_min_t <= t_next <= cfg_max_t and guidance_active

                v2 = probability_flow_derivative(
                    model_velocity(
                        x_pred,
                        t_next_tensor,
                        step_idx=i,
                        apply_cfg=apply_cfg_next,
                    ),
                    t_next_tensor,
                )

                if temporal_rescale_k != 1.0:
                    v2 = temporal_score_rescaling(
                        v2, k=temporal_rescale_k, sigma=temporal_rescale_sigma
                    )

                x = x + dt * (v1 + v2) / 2
                if step_callback is not None:
                    step_callback(i, x)

        elif solver == "rk4":
            assert half_timesteps is not None and next_timesteps is not None
            for i in iterator:
                t = i * dt
                t_tensor = timesteps[i].expand(batch_size)
                t_half_tensor = half_timesteps[i].expand(batch_size)
                t_next_tensor = next_timesteps[i].expand(batch_size)

                def get_velocity(x_in, t_in, time_value):
                    apply_cfg = cfg_min_t <= time_value <= cfg_max_t and guidance_active
                    v = probability_flow_derivative(
                        model_velocity(
                            x_in,
                            t_in,
                            step_idx=i,
                            apply_cfg=apply_cfg,
                        ),
                        t_in,
                    )
                    if temporal_rescale_k != 1.0:
                        v = temporal_score_rescaling(
                            v, k=temporal_rescale_k, sigma=temporal_rescale_sigma
                        )
                    return v

                k1 = get_velocity(x, t_tensor, t)
                k2 = get_velocity(x + dt / 2 * k1, t_half_tensor, t + dt / 2)
                k3 = get_velocity(x + dt / 2 * k2, t_half_tensor, t + dt / 2)
                k4 = get_velocity(x + dt * k3, t_next_tensor, t + dt)

                x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if step_callback is not None:
                    step_callback(i, x)

        else:  # pragma: no cover - validated before allocation
            raise AssertionError(f"Unhandled validated solver: {solver}")

        # Rescale latents
        if target_latent_std is not None:
            # Normalize each utterance independently. A shared batch statistic
            # couples otherwise independent requests and makes output depend on
            # which compatible texts happened to be scheduled together.
            reduction_dims = tuple(range(1, x.ndim))
            current_std = x.std(dim=reduction_dims, keepdim=True)
            # Preserve the zero-variance behavior without converting a CUDA
            # boolean to a Python bool (which would force a device sync).
            scale = torch.where(
                current_std > 1e-6,
                target_latent_std / current_std.clamp_min(1e-6),
                torch.ones_like(current_std),
            )
            x = x * scale

        return x
