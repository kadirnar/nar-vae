"""Versioned duration prediction for EchoDiT v2 checkpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F

from .alignment import allocate_integer_durations

ECHODIT_ARCHITECTURE_VERSION = 4
DURATION_PREDICTOR_VERSION = 1
MONOTONIC_ALIGNMENT_VERSION = 1


@dataclass(frozen=True)
class DurationAlignmentOutput:
    """Differentiable predictions used by the versioned MAS training objective."""

    total_log_frames: torch.Tensor
    token_durations: torch.Tensor
    log_likelihoods: torch.Tensor
    hard_alignment: torch.Tensor


def _fixed_dct_projection(hidden_size: int, latent_size: int) -> torch.Tensor:
    """Build a deterministic orthonormal projection over DACVAE channels."""
    positions = torch.arange(latent_size, dtype=torch.float32) + 0.5
    frequencies = torch.arange(hidden_size, dtype=torch.float32)[:, None]
    projection = torch.cos(math.pi * frequencies * positions / latent_size)
    projection[0] *= math.sqrt(1.0 / latent_size)
    if hidden_size > 1:
        projection[1:] *= math.sqrt(2.0 / latent_size)
    return projection


class EchoDurationAlignment(nn.Module):
    """Compact Gaussian text prior over a fixed projection of clean acoustic frames.

    The fixed acoustic projection keeps hard-EM alignment training from finding the trivial
    solution where both a learned acoustic projection and the text prior collapse to a constant.
    """

    def __init__(self, *, text_size: int, latent_size: int, hidden_size: int):
        super().__init__()
        if hidden_size <= 0 or hidden_size > latent_size:
            raise ValueError(
                "duration alignment hidden_size must be positive and no larger than latent_size"
            )
        self.hidden_size = hidden_size
        self.text_statistics = nn.Linear(text_size, hidden_size * 2)
        self.register_buffer(
            "latent_projection",
            _fixed_dct_projection(hidden_size, latent_size),
        )

    def forward(
        self,
        text_state: torch.Tensor,
        clean_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Return average diagonal-Gaussian log likelihoods ``[batch, token, frame]``."""
        if text_state.ndim != 3:
            raise ValueError("text_state must have shape [batch, token, channel].")
        if clean_latents.ndim != 3:
            raise ValueError("clean_latents must have shape [batch, channel, frame].")
        if text_state.shape[0] != clean_latents.shape[0]:
            raise ValueError("Text states and clean latents must use the same batch size.")
        if clean_latents.shape[1] != self.latent_projection.shape[1]:
            raise ValueError("Clean latent channels do not match the alignment projection.")
        if not torch.isfinite(clean_latents).all():
            raise ValueError("clean_latents must contain only finite values.")

        statistics = self.text_statistics(text_state).float()
        token_means, raw_log_scales = statistics.chunk(2, dim=-1)
        # Smooth finite bounds avoid zero variances while retaining gradients near the limits.
        token_log_scales = 4.0 * torch.tanh(raw_log_scales / 4.0)
        inverse_variances = torch.exp(-2.0 * token_log_scales)
        weighted_means = token_means * inverse_variances

        frame_states = F.linear(
            clean_latents.detach().transpose(1, 2).float(),
            self.latent_projection.float(),
        )
        # Expand the Gaussian quadratic algebraically, avoiding a [B, L, T, H] tensor.
        quadratic = torch.bmm(
            frame_states.square(),
            inverse_variances.transpose(1, 2),
        )
        quadratic = quadratic - 2.0 * torch.bmm(
            frame_states,
            weighted_means.transpose(1, 2),
        )
        token_constant = (token_means.square() * inverse_variances).sum(dim=-1)
        quadratic = (quadratic + token_constant[:, None, :]).clamp_min(0.0)
        normalizer = 2.0 * token_log_scales.sum(dim=-1) + self.hidden_size * math.log(2.0 * math.pi)
        log_likelihoods = -0.5 * (quadratic + normalizer[:, None, :]) / self.hidden_size
        log_likelihoods = log_likelihoods.transpose(1, 2)
        if not torch.isfinite(log_likelihoods).all():
            raise FloatingPointError("Duration alignment produced nonfinite log likelihoods.")
        return log_likelihoods


def _masked_mean(state: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Pool a sequence without allowing padded positions to influence its mean."""
    if mask is None:
        return state.mean(dim=1)
    expected_shape = state.shape[:2]
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"Conditioning mask must have shape {expected_shape}; got {tuple(mask.shape)}."
        )
    mask = mask.to(device=state.device, dtype=torch.bool)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every duration-prediction row must contain a valid conditioning token.")
    weights = mask.unsqueeze(-1).to(dtype=state.dtype)
    return (state * weights).sum(dim=1) / weights.sum(dim=1)


class DurationResidualBlock(nn.Module):
    """Small pre-normalized residual MLP used by the duration head."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.in_projection = nn.Linear(hidden_size, hidden_size * 2)
        self.out_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        residual = self.norm(state)
        residual = self.out_projection(F.silu(self.in_projection(residual)))
        return state + residual


class EchoDurationPredictor(nn.Module):
    """Predict ``log1p`` DACVAE frames as a sum of non-negative token contributions."""

    def __init__(
        self,
        *,
        text_size: int,
        hidden_size: int,
        num_layers: int,
        speaker_size: int | None = None,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("duration predictor hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("duration predictor num_layers must be positive")

        self.uses_speaker = speaker_size is not None
        self.text_projection = nn.Linear(text_size, hidden_size)
        self.speaker_projection = (
            nn.Linear(speaker_size, hidden_size, bias=False) if speaker_size is not None else None
        )
        self.blocks = nn.ModuleList(DurationResidualBlock(hidden_size) for _ in range(num_layers))
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, 1)
        nn.init.constant_(self.output_projection.bias, 9.0)

    def forward(
        self,
        text_state: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        speaker_state: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        *,
        return_token_durations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return total log frames and, when requested, positive token contributions."""
        if text_mask is None:
            text_weights = torch.ones(
                text_state.shape[:2],
                dtype=text_state.dtype,
                device=text_state.device,
            )
        else:
            expected_shape = text_state.shape[:2]
            if tuple(text_mask.shape) != expected_shape:
                raise ValueError(
                    f"Conditioning mask must have shape {expected_shape}; "
                    f"got {tuple(text_mask.shape)}."
                )
            text_mask = text_mask.to(device=text_state.device, dtype=torch.bool)
            if not bool(text_mask.any(dim=1).all()):
                raise ValueError(
                    "Every duration-prediction row must contain a valid conditioning token."
                )
            text_weights = text_mask.to(dtype=text_state.dtype)

        state = self.text_projection(text_state)
        if self.speaker_projection is not None:
            if speaker_state is None:
                raise ValueError("This duration predictor requires speaker conditioning state.")
            speaker_condition = self.speaker_projection(
                _masked_mean(speaker_state, speaker_mask)
            ).unsqueeze(1)
            state = state + speaker_condition
        elif speaker_state is not None:
            raise ValueError("This duration predictor was not trained with speaker conditioning.")

        for block in self.blocks:
            state = block(state)
        token_frames = F.softplus(self.output_projection(self.output_norm(state))).squeeze(-1)
        token_frames = token_frames * text_weights
        total_log_frames = torch.log1p(token_frames.sum(dim=1))
        if return_token_durations:
            return total_log_frames, token_frames
        return total_log_frames


def _normalize_requested_totals(
    total_frames: int | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(total_frames, bool):
        raise TypeError("total_frames must be an integer, not bool.")
    if isinstance(total_frames, Integral):
        totals = torch.full((batch_size,), int(total_frames), device=device, dtype=torch.long)
    elif isinstance(total_frames, torch.Tensor):
        totals = total_frames.to(device=device)
        if totals.ndim == 0:
            totals = totals.expand(batch_size)
        elif tuple(totals.shape) != (batch_size,):
            raise ValueError(f"total_frames must be scalar or have shape {(batch_size,)}.")
        if totals.is_complex() or totals.dtype == torch.bool:
            raise TypeError("total_frames must contain integers.")
        if torch.is_floating_point(totals):
            if not torch.isfinite(totals).all() or not torch.equal(totals, totals.round()):
                raise ValueError("total_frames must contain finite integral values.")
        totals = totals.to(dtype=torch.long)
    else:
        raise TypeError("total_frames must be an integer or tensor.")
    return totals


@torch.no_grad()
def allocate_positive_token_durations(
    expected_durations: torch.Tensor,
    total_frames: int | torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate at least one frame per valid token while exactly matching requested totals."""
    if expected_durations.ndim == 1:
        batched = expected_durations.unsqueeze(0)
        squeeze_batch = True
    elif expected_durations.ndim == 2:
        batched = expected_durations
        squeeze_batch = False
    else:
        raise ValueError("expected_durations must have shape [token] or [batch, token].")
    if any(size == 0 for size in batched.shape):
        raise ValueError("expected_durations dimensions must be non-empty.")

    if token_mask is None:
        valid_tokens = torch.ones_like(batched, dtype=torch.bool)
    else:
        valid_tokens = token_mask
        if expected_durations.ndim == 1 and valid_tokens.ndim == 1:
            valid_tokens = valid_tokens.unsqueeze(0)
        if tuple(valid_tokens.shape) != tuple(batched.shape):
            raise ValueError("token_mask must have the expected-duration shape.")
        valid_tokens = valid_tokens.to(device=batched.device, dtype=torch.bool)
    token_counts = valid_tokens.sum(dim=1, dtype=torch.long)
    if not bool((token_counts > 0).all()):
        raise ValueError("Every row must contain at least one valid token.")
    expected_prefix = (
        torch.arange(batched.shape[1], device=batched.device)[None, :] < token_counts[:, None]
    )
    if not torch.equal(valid_tokens, expected_prefix):
        raise ValueError("token_mask must be a left-aligned contiguous prefix mask.")

    totals = _normalize_requested_totals(
        total_frames,
        batch_size=batched.shape[0],
        device=batched.device,
    )
    if bool((totals < token_counts).any()):
        raise ValueError("total_frames must provide at least one frame per valid token.")
    # The mandatory one-frame floor is already represented in a predictor's
    # full per-token frame estimate. Allocate only its mass above that floor;
    # otherwise adding the floor after apportioning the full estimates inflates
    # short tokens and compresses long ones (for example [1, 9] -> [2, 8]).
    residual_weights = torch.clamp(
        batched - valid_tokens.to(dtype=batched.dtype),
        min=0,
    )
    residual = allocate_integer_durations(
        residual_weights,
        totals - token_counts,
        valid_tokens,
    )
    allocation = residual + valid_tokens.to(dtype=torch.long)
    return allocation[0] if squeeze_batch else allocation


def expand_text_by_durations(
    text_state: torch.Tensor,
    token_durations: torch.Tensor,
    *,
    target_frames: int | None = None,
) -> torch.Tensor:
    """Expand token states to an exact frame sequence using integral durations."""
    if text_state.ndim != 3:
        raise ValueError("text_state must have shape [batch, token, channel].")
    if token_durations.ndim != 2 or token_durations.shape != text_state.shape[:2]:
        raise ValueError("token_durations must have shape [batch, token].")
    if token_durations.dtype == torch.bool or token_durations.is_complex():
        raise TypeError("token_durations must contain nonnegative integers.")
    durations = token_durations.to(device=text_state.device)
    if torch.is_floating_point(durations):
        if not bool(torch.isfinite(durations).all()) or not torch.equal(
            durations, durations.round()
        ):
            raise ValueError("token_durations must contain finite integers.")
    durations = durations.to(dtype=torch.long)
    if bool((durations < 0).any()):
        raise ValueError("token_durations must be nonnegative.")
    totals = durations.sum(dim=1)
    if not bool((totals > 0).all()):
        raise ValueError("Every duration row must allocate at least one frame.")
    if target_frames is None:
        if not bool((totals == totals[0]).all()):
            raise ValueError("Duration rows require one shared frame count for tensor expansion.")
        target_frames = int(totals[0].item())
    elif (
        isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames <= 0
    ):
        raise ValueError("target_frames must be a positive integer.")
    if not bool((totals == target_frames).all()):
        raise ValueError("Every duration row must sum exactly to target_frames.")

    boundaries = durations.cumsum(dim=1).contiguous()
    frame_positions = (
        torch.arange(
            1,
            target_frames + 1,
            device=text_state.device,
            dtype=torch.long,
        )
        .expand(text_state.shape[0], -1)
        .contiguous()
    )
    token_indices = torch.searchsorted(boundaries, frame_positions, right=False)
    return torch.gather(
        text_state,
        1,
        token_indices.unsqueeze(-1).expand(-1, -1, text_state.shape[-1]),
    )


__all__ = [
    "DURATION_PREDICTOR_VERSION",
    "DurationAlignmentOutput",
    "ECHODIT_ARCHITECTURE_VERSION",
    "EchoDurationAlignment",
    "EchoDurationPredictor",
    "MONOTONIC_ALIGNMENT_VERSION",
    "allocate_positive_token_durations",
    "expand_text_by_durations",
]
