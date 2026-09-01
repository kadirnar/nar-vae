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


def _token_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    length: int,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize an arbitrary ordered token mask and return its row counts.

    Unlike acoustic frame masks, alignment masks may exclude control tokens at
    either edge of the text sequence or between speakable spans.  Their original
    order still defines the monotonic path; compacting the selected positions is
    an implementation detail of :func:`monotonic_alignment_search`.
    """
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
        token_mask: Optional ordered, possibly non-contiguous valid-token mask with shape
            ``[N]`` or ``[B, N]``.
        frame_mask: Optional left-aligned valid-frame mask with shape ``[T]`` or ``[B, T]``.

    Returns:
        A boolean hard alignment with the same shape as ``log_likelihoods``.
    """
    if not torch.is_floating_point(log_likelihoods):
        raise TypeError("log_likelihoods must be a floating-point tensor.")
    scores, squeeze_batch = _as_batched_matrix(log_likelihoods, name="log_likelihoods")
    batch_size, max_tokens, max_frames = scores.shape
    valid_tokens, token_lengths = _token_mask(
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
    if bool((frame_lengths < token_lengths).any()):
        raise ValueError(
            "Every alignment row requires at least one frame per token selected by token_mask."
        )

    valid_region = valid_tokens[:, :, None] & valid_frames[:, None, :]
    invalid_scores = torch.isnan(scores) | torch.isposinf(scores)
    if bool((invalid_scores & valid_region).any()):
        raise ValueError("Valid alignment scores cannot contain NaN or positive-infinite values.")

    # Compact arbitrary speakable-token masks without transferring token indices
    # or direction tables to the CPU.  Invalid positions sort behind every valid
    # position while stable token order is retained within each row.
    token_positions = torch.arange(max_tokens, device=scores.device).expand(batch_size, -1)
    sort_keys = torch.where(valid_tokens, token_positions, token_positions + max_tokens)
    compact_to_original = sort_keys.argsort(dim=1)
    compact_scores = torch.gather(
        scores,
        1,
        compact_to_original[:, :, None].expand(-1, -1, max_frames),
    )
    compact_valid_tokens = (
        torch.arange(max_tokens, device=scores.device)[None, :] < token_lengths[:, None]
    )

    negative_infinity = scores.new_full((batch_size, 1), float("-inf"))
    previous = scores.new_full((batch_size, max_tokens), float("-inf"))
    previous[:, 0] = compact_scores[:, 0, 0]
    previous = previous.masked_fill(~compact_valid_tokens, float("-inf"))
    advanced = torch.zeros_like(scores, dtype=torch.bool)

    for frame_index in range(1, max_frames):
        from_previous_token = torch.cat((negative_infinity, previous[:, :-1]), dim=1)
        # Prefer advancing on exact ties so the batched implementation preserves
        # the historical deterministic path rule.
        choose_advance = from_previous_token >= previous
        candidate = compact_scores[:, :, frame_index] + torch.where(
            choose_advance,
            from_previous_token,
            previous,
        )
        candidate = candidate.masked_fill(~compact_valid_tokens, float("-inf"))
        active_rows = frame_index < frame_lengths
        previous = torch.where(active_rows[:, None], candidate, previous)
        advanced[:, :, frame_index] = choose_advance & active_rows[:, None]

    batch_indices = torch.arange(batch_size, device=scores.device)
    final_scores = previous[batch_indices, token_lengths - 1]
    if not bool(torch.isfinite(final_scores).all()):
        raise ValueError("At least one alignment row has no finite monotonic path.")

    # Backtrack every row together on the original device. Rows whose valid
    # acoustic sequence has already ended remain untouched until their last frame
    # is reached. This avoids the former per-row device synchronization and CPU
    # direction-table copy while producing the same maximum-likelihood paths.
    alignment = torch.zeros_like(scores, dtype=torch.bool)
    compact_token_index = token_lengths - 1
    for frame_index in range(max_frames - 1, -1, -1):
        active_rows = frame_index < frame_lengths
        active_batch = batch_indices[active_rows]
        active_compact_tokens = compact_token_index[active_rows]
        original_tokens = compact_to_original[active_batch, active_compact_tokens]
        alignment[active_batch, original_tokens, frame_index] = True
        if frame_index:
            took_advance = advanced[
                batch_indices,
                compact_token_index,
                frame_index,
            ]
            compact_token_index = compact_token_index - (active_rows & took_advance).to(torch.long)
    if bool((compact_token_index != 0).any()):
        raise RuntimeError("Monotonic alignment backtracking did not reach the first valid token.")

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
    valid_tokens, token_lengths = _token_mask(
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

    original_token_indices = hard_alignment.to(dtype=torch.long).argmax(dim=1)
    compact_ranks = valid_tokens.cumsum(dim=1, dtype=torch.long) - 1
    token_indices = torch.gather(compact_ranks, 1, original_token_indices)
    if max_frames > 1:
        transitions = token_indices[:, 1:] - token_indices[:, :-1]
        invalid_transition = ((transitions < 0) | (transitions > 1)) & valid_frames[:, 1:]
        if bool(invalid_transition.any()):
            raise ValueError(
                "alignment must be monotonic and may only stay on a token or advance by one token."
            )
    batch_indices = torch.arange(batch_size, device=hard_alignment.device)
    final_token_indices = token_indices[batch_indices, frame_lengths - 1]
    if bool((token_indices[:, 0] != 0).any()) or bool(
        (final_token_indices != token_lengths - 1).any()
    ):
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
    valid_tokens, _ = _token_mask(
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
