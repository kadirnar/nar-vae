"""Low-parameter fixed-token adapter for variable-length reference states.

This optional adapter sits outside DACVAE: it consumes already encoded speaker
states and emits a bounded number of conditioning tokens.  It neither depends
on an external speaker model nor changes the codec representation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _headwise_rms_norm(value: torch.Tensor, eps: float) -> torch.Tensor:
    """Parameter-free QK normalization with float32 reduction."""
    dtype = value.dtype
    normalized = value.float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(dtype=dtype)


class ReferenceResampler(nn.Module):
    """Compress ``[B,S,H]`` speaker states into fixed ``[B,N,H]`` tokens.

    Query and key heads receive parameter-free RMS normalization before scaled
    dot-product attention.  Query zero is the designated global timbre token and
    receives a direct masked-mean residual from the source states; the remaining
    queries retain local reference detail.  A bottleneck residual MLP keeps the
    adapter small (about ``5 * H**2`` parameters with the default ratio, plus
    learned queries and norms).

    Args:
        hidden_size: Input and output feature width ``H``.
        num_queries: Fixed output token count ``N``. Query zero is always global.
        num_heads: Cross-attention head count; must divide ``hidden_size``.
        mlp_ratio: Residual-MLP bottleneck width relative to ``hidden_size``.
        norm_eps: Epsilon for input, QK, and output normalization.
    """

    GLOBAL_TOKEN_INDEX = 0

    def __init__(
        self,
        hidden_size: int,
        num_queries: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 0.5,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_integer(hidden_size, name="hidden_size")
        self.num_queries = _positive_integer(num_queries, name="num_queries")
        self.num_heads = _positive_integer(num_heads, name="num_heads")
        self.mlp_ratio = _positive_finite(mlp_ratio, name="mlp_ratio")
        self.norm_eps = _positive_finite(norm_eps, name="norm_eps")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.head_dim = self.hidden_size // self.num_heads
        mlp_hidden_size = max(1, round(self.hidden_size * self.mlp_ratio))

        self.query_tokens = nn.Parameter(torch.empty(self.num_queries, self.hidden_size))
        self.input_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.kv_proj = nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.mlp_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, mlp_hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_size, self.hidden_size),
        )
        self.output_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)

        nn.init.normal_(self.query_tokens, mean=0.0, std=self.hidden_size**-0.5)

    def _validated_mask(
        self,
        speaker_states: torch.Tensor,
        speaker_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, source_length = speaker_states.shape[:2]
        if speaker_mask is None:
            return torch.ones(
                (batch_size, source_length),
                dtype=torch.bool,
                device=speaker_states.device,
            )
        if not isinstance(speaker_mask, torch.Tensor):
            raise TypeError("speaker_mask must be a torch.Tensor or None.")
        if tuple(speaker_mask.shape) != (batch_size, source_length):
            raise ValueError(
                "speaker_mask must have shape "
                f"{(batch_size, source_length)}; got {tuple(speaker_mask.shape)}."
            )
        mask = speaker_mask.to(device=speaker_states.device, dtype=torch.bool)
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every reference row must contain at least one valid speaker state.")
        return mask

    def forward(
        self,
        speaker_states: torch.Tensor,
        speaker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-count reference tokens, masking padded source positions."""
        if not isinstance(speaker_states, torch.Tensor):
            raise TypeError("speaker_states must be a torch.Tensor.")
        if speaker_states.ndim != 3:
            raise ValueError("speaker_states must have shape [B, S, H].")
        batch_size, source_length, hidden_size = speaker_states.shape
        if batch_size <= 0 or source_length <= 0:
            raise ValueError("speaker_states batch and source dimensions must be non-empty.")
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"speaker_states width is {hidden_size}, expected hidden_size={self.hidden_size}."
            )
        if not speaker_states.is_floating_point():
            raise ValueError("speaker_states must use a floating-point dtype.")
        mask = self._validated_mask(speaker_states, speaker_mask)

        # Clear padding before any normalization or matrix multiplication.  In
        # addition to saving its gradients, this prevents even NaN/Inf padding
        # sentinels from leaking through an implementation-specific SDPA path.
        masked_states = speaker_states.masked_fill(~mask.unsqueeze(-1), 0.0)
        source = self.input_norm(masked_states)
        queries = self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        query = self.q_proj(queries).view(
            batch_size,
            self.num_queries,
            self.num_heads,
            self.head_dim,
        )
        key, value = self.kv_proj(source).chunk(2, dim=-1)
        key = key.view(batch_size, source_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, source_length, self.num_heads, self.head_dim)

        query = _headwise_rms_norm(query.transpose(1, 2), self.norm_eps)
        key = _headwise_rms_norm(key.transpose(1, 2), self.norm_eps)
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            self.num_queries,
            self.hidden_size,
        )
        tokens = queries + self.out_proj(attended)

        # Preserve an explicit global timbre path without another H-by-H matrix.
        valid_counts = mask.sum(dim=1, keepdim=True).to(dtype=speaker_states.dtype)
        global_summary = masked_states.sum(dim=1) / valid_counts
        tokens = torch.cat((tokens[:, :1] + global_summary.unsqueeze(1), tokens[:, 1:]), dim=1)

        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return self.output_norm(tokens)


__all__ = ["ReferenceResampler"]
