"""Parameter-free monotonic alignment and exact duration-allocation utilities."""

from __future__ import annotations

from numbers import Integral

import torch


def _as_batched_matrix(tensor: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 2:
        batched, squeeze_batch = tensor.unsqueeze(0), True
    elif tensor.ndim == 3:
        batched, squeeze_batch = tensor, False
    else:
        raise ValueError(f"{name} must have shape [tokens, frames] or [batch, tokens, frames].")
    if any(size == 0 for size in batched.shape):
        raise ValueError(f"{name} dimensions must be non-empty.")
    return batched, squeeze_batch


def _as_batched_rows(tensor: torch.Tensor, *, name: str) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        batched, squeeze_batch = tensor.unsqueeze(0), True
    elif tensor.ndim == 2:
        batched, squeeze_batch = tensor, False
    else:
        raise ValueError(f"{name} must have shape [tokens] or [batch, tokens].")
    if any(size == 0 for size in batched.shape):
        raise ValueError(f"{name} dimensions must be non-empty.")
    return batched, squeeze_batch


def _prefix_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    length: int,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a left-aligned padding mask and return it with row lengths."""
    if mask is None:
        normalized = torch.ones((batch_size, length), dtype=torch.bool, device=device)
    else:
        if mask.ndim == 1:
            if batch_size != 1:
                raise ValueError(f"{name} must have shape [{batch_size}, {length}].")
            mask = mask.unsqueeze(0)
        if tuple(mask.shape) != (batch_size, length):
            raise ValueError(
                f"{name} must have shape {(batch_size, length)}; got {tuple(mask.shape)}."
            )
        normalized = mask.to(device=device, dtype=torch.bool)

    lengths = normalized.sum(dim=1, dtype=torch.long)
    expected = torch.arange(length, device=device)[None, :] < lengths[:, None]
    if not torch.equal(normalized, expected):
        raise ValueError(f"{name} must be a left-aligned contiguous prefix mask.")
    if not bool((lengths > 0).all()):
        raise ValueError(f"Every row in {name} must contain at least one valid position.")
    return normalized, lengths


@torch.no_grad()
def monotonic_alignment_search(
    log_likelihoods: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find the maximum-likelihood hard monotonic token-to-frame alignment.

    The path starts at the first valid token and frame, ends at the final valid
    token and frame, and may either remain on the current token or advance by one
    token at each frame. Consequently every valid frame is assigned exactly once
    and every valid token receives at least one frame.

    Args:
        log_likelihoods: Token-by-frame scores with shape ``[N, T]`` or ``[B, N, T]``.
        token_mask: Optional left-aligned valid-token mask with shape ``[N]`` or ``[B, N]``.
        frame_mask: Optional left-aligned valid-frame mask with shape ``[T]`` or ``[B, T]``.

    Returns:
        A boolean hard alignment with the same shape as ``log_likelihoods``.
    """
    if not torch.is_floating_point(log_likelihoods):
        raise TypeError("log_likelihoods must be a floating-point tensor.")
    scores, squeeze_batch = _as_batched_matrix(log_likelihoods, name="log_likelihoods")
    batch_size, max_tokens, max_frames = scores.shape
    valid_tokens, token_lengths = _prefix_mask(
        token_mask,
        batch_size=batch_size,
        length=max_tokens,
        device=scores.device,
        name="token_mask",
    )
    valid_frames, frame_lengths = _prefix_mask(
        frame_mask,
        batch_size=batch_size,
        length=max_frames,
        device=scores.device,
        name="frame_mask",
    )
    del valid_tokens, valid_frames

    alignment = torch.zeros_like(scores, dtype=torch.bool)
    for batch_index in range(batch_size):
        token_count = int(token_lengths[batch_index].item())
        frame_count = int(frame_lengths[batch_index].item())
        if frame_count < token_count:
            raise ValueError(
                f"Alignment row {batch_index} has {frame_count} valid frames for "
                f"{token_count} valid tokens; at least one frame per token is required."
            )

        row_scores = scores[batch_index, :token_count, :frame_count]
        if bool(torch.isnan(row_scores).any()) or bool(torch.isposinf(row_scores).any()):
            raise ValueError(
                f"Alignment row {batch_index} contains NaN or positive-infinite scores."
            )

        previous = row_scores.new_full((token_count,), float("-inf"))
        previous[0] = row_scores[0, 0]
        advanced = torch.zeros(
            (token_count, frame_count),
            dtype=torch.bool,
            device=row_scores.device,
        )
        negative_infinity = row_scores.new_full((1,), float("-inf"))

        for frame_index in range(1, frame_count):
            from_previous_token = torch.cat((negative_infinity, previous[:-1]))
            # Prefer advancing on exact ties so the result is deterministic.
            choose_advance = from_previous_token >= previous
            previous = row_scores[:, frame_index] + torch.where(
                choose_advance,
                from_previous_token,
                previous,
            )
            advanced[:, frame_index] = choose_advance

        if not bool(torch.isfinite(previous[token_count - 1])):
            raise ValueError(f"Alignment row {batch_index} has no finite monotonic path.")

        # Copy the compact direction table once instead of synchronizing the device
        # for every scalar decision during backtracking.
        directions = advanced.detach().cpu()
        token_path = [0] * frame_count
        token_index = token_count - 1
        for frame_index in range(frame_count - 1, -1, -1):
            token_path[frame_index] = token_index
            if frame_index > 0 and bool(directions[token_index, frame_index]):
                token_index -= 1
        if token_index != 0:
            raise RuntimeError("Monotonic alignment backtracking did not reach the first token.")

        frame_indices = torch.arange(frame_count, device=alignment.device)
        path_indices = torch.tensor(token_path, dtype=torch.long, device=alignment.device)
        alignment[batch_index, path_indices, frame_indices] = True

    return alignment[0] if squeeze_batch else alignment


@torch.no_grad()
def durations_from_alignment(
    alignment: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract integer frame counts from a hard monotonic alignment."""
    hard_alignment, squeeze_batch = _as_batched_matrix(alignment, name="alignment")
    if hard_alignment.dtype != torch.bool:
        is_binary = (hard_alignment == 0) | (hard_alignment == 1)
        if not bool(is_binary.all()):
            raise ValueError("alignment must contain only binary values.")
        hard_alignment = hard_alignment.to(dtype=torch.bool)

    batch_size, max_tokens, max_frames = hard_alignment.shape
    valid_tokens, token_lengths = _prefix_mask(
        token_mask,
        batch_size=batch_size,
        length=max_tokens,
        device=hard_alignment.device,
        name="token_mask",
    )
    valid_frames, frame_lengths = _prefix_mask(
        frame_mask,
        batch_size=batch_size,
        length=max_frames,
        device=hard_alignment.device,
        name="frame_mask",
    )
    valid_region = valid_tokens[:, :, None] & valid_frames[:, None, :]
    if bool((hard_alignment & ~valid_region).any()):
        raise ValueError("alignment assigns a padded token or frame.")

    assigned_per_frame = hard_alignment.sum(dim=1)
    if not bool((assigned_per_frame[valid_frames] == 1).all()):
        raise ValueError("Every valid frame must be assigned to exactly one valid token.")

    for batch_index in range(batch_size):
        token_count = int(token_lengths[batch_index].item())
        frame_count = int(frame_lengths[batch_index].item())
        token_indices = (
            hard_alignment[
                batch_index,
                :token_count,
                :frame_count,
            ]
            .to(dtype=torch.long)
            .argmax(dim=0)
        )
        if frame_count > 1:
            transitions = token_indices[1:] - token_indices[:-1]
            if bool(((transitions < 0) | (transitions > 1)).any()):
                raise ValueError(
                    "alignment must be monotonic and may only stay on a token or advance by one token."
                )
        if token_indices[0] != 0 or token_indices[-1] != token_count - 1:
            raise ValueError("alignment must start at the first token and end at the final token.")

    durations = hard_alignment.sum(dim=-1, dtype=torch.long)
    if not bool((durations[valid_tokens] > 0).all()):
        raise ValueError("Every valid token must receive at least one frame.")
    return durations[0] if squeeze_batch else durations


def _normalize_totals(
    total_frames: int | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(total_frames, bool):
        raise TypeError("total_frames must be an integer, not bool.")
    if isinstance(total_frames, Integral):
        totals = torch.full(
            (batch_size,),
            int(total_frames),
            dtype=torch.long,
            device=device,
        )
    elif isinstance(total_frames, torch.Tensor):
        totals = total_frames.to(device=device)
        if totals.ndim == 0:
            totals = totals.expand(batch_size)
        elif tuple(totals.shape) != (batch_size,):
            raise ValueError(
                f"total_frames must be scalar or have shape {(batch_size,)}; "
                f"got {tuple(totals.shape)}."
            )
        if totals.is_complex():
            raise TypeError("total_frames must contain integers.")
        if torch.is_floating_point(totals):
            if not bool(torch.isfinite(totals).all()):
                raise ValueError("total_frames must be finite.")
            if not bool((totals == totals.round()).all()):
                raise ValueError("total_frames must contain integral values.")
        totals = totals.to(dtype=torch.long)
    else:
        raise TypeError("total_frames must be an integer or tensor.")

    if bool((totals < 0).any()):
        raise ValueError("total_frames must be nonnegative.")
    return totals


@torch.no_grad()
def allocate_integer_durations(
    expected_durations: torch.Tensor,
    total_frames: int | torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate nonnegative integer token durations that exactly match a total.

    Valid token weights are normalized to the requested total, floored, and the
    remaining frames are assigned by descending fractional remainder. Ties retain
    token order. If every valid weight is zero, frames are distributed as evenly as
    possible, again favoring earlier tokens for an exact tie.
    """
    if expected_durations.is_complex():
        raise TypeError("expected_durations must be real-valued.")
    weights, squeeze_batch = _as_batched_rows(
        expected_durations,
        name="expected_durations",
    )
    batch_size, max_tokens = weights.shape
    valid_tokens, _ = _prefix_mask(
        token_mask,
        batch_size=batch_size,
        length=max_tokens,
        device=weights.device,
        name="token_mask",
    )
    totals = _normalize_totals(
        total_frames,
        batch_size=batch_size,
        device=weights.device,
    )
    weights = weights.detach().to(dtype=torch.float64)
    valid_weights = weights[valid_tokens]
    if not bool(torch.isfinite(valid_weights).all()):
        raise ValueError("Valid expected_durations must be finite.")
    if bool((valid_weights < 0).any()):
        raise ValueError("Valid expected_durations must be nonnegative.")

    allocation = torch.zeros_like(weights, dtype=torch.long)
    for batch_index in range(batch_size):
        valid_indices = valid_tokens[batch_index].nonzero(as_tuple=False).squeeze(1)
        row_weights = weights[batch_index].index_select(0, valid_indices)
        total = int(totals[batch_index].item())
        if total == 0:
            continue
        if not bool((row_weights > 0).any()):
            row_weights = torch.ones_like(row_weights)

        quotas = row_weights / row_weights.sum() * total
        row_allocation = torch.floor(quotas).to(dtype=torch.long)
        remainder_count = total - int(row_allocation.sum().item())
        if remainder_count < 0 or remainder_count > row_allocation.numel():
            raise RuntimeError("Duration apportionment exceeded its numerical bounds.")
        if remainder_count:
            fractional = quotas - row_allocation.to(dtype=quotas.dtype)
            order = torch.argsort(fractional, descending=True, stable=True)
            row_allocation[order[:remainder_count]] += 1

        allocation[batch_index, valid_indices] = row_allocation

    if not torch.equal(allocation.sum(dim=1), totals):
        raise RuntimeError("Duration allocation failed to match the requested total.")
    return allocation[0] if squeeze_batch else allocation


__all__ = [
    "allocate_integer_durations",
    "durations_from_alignment",
    "monotonic_alignment_search",
]
