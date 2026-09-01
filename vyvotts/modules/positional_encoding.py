import math

import torch
import torch.nn as nn


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for Transformers.

    Applies rotary position embeddings to query and key tensors in attention.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Args:
        dim: Dimension of the embeddings (should be even)
        max_seq_len: Maximum sequence length
        base: Base for the geometric progression (default: 10000)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute positional encodings for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos and sin for all positions."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Apply rotary position embeddings.

        Args:
            x: Input tensor of shape [B, num_heads, seq_len, head_dim]
               or [B, seq_len, num_heads, head_dim]
            seq_len: Optional sequence length (inferred from x if not provided)

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        if seq_len is None:
            seq_len = x.shape[1] if x.ndim == 4 and x.shape[1] > x.shape[2] else x.shape[2]

        # Rebuild cache if needed
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        return apply_rotary_pos_emb(x, self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len])


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.

    Args:
        x: Input tensor [..., dim]

    Returns:
        Rotated tensor of same shape
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to input tensor.

    Args:
        x: Input tensor of shape [B, seq_len, num_heads, head_dim] or
           [B, num_heads, seq_len, head_dim]
        cos: Cosine tensor from RoPE [1, seq_len, 1, dim]
        sin: Sine tensor from RoPE [1, seq_len, 1, dim]

    Returns:
        Tensor with rotary embeddings applied
    """
    # Handle both [B, S, H, D] and [B, H, S, D] formats
    if x.ndim == 4:
        if x.shape[1] > x.shape[2]:
            # Format: [B, S, H, D] - transpose to [B, H, S, D]
            x = x.transpose(1, 2)
            cos = cos.transpose(1, 2) if cos.shape[1] != 1 else cos
            sin = sin.transpose(1, 2) if sin.shape[1] != 1 else sin
            result = (x * cos) + (rotate_half(x) * sin)
            return result.transpose(1, 2)

    # Default case: [B, H, S, D]
    return (x * cos) + (rotate_half(x) * sin)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding as an alternative to RoPE.

    Simpler than RoPE but requires learning position embeddings.

    Args:
        max_seq_len: Maximum sequence length
        dim: Embedding dimension
    """

    def __init__(self, max_seq_len: int, dim: int):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.

        Args:
            x: Input tensor of shape [B, seq_len, dim]

        Returns:
            x + positional_embeddings
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        return x + self.positional_embedding[:, :seq_len, :]


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need".

    Fixed (non-learned) sinusoidal position embeddings.

    Args:
        dim: Embedding dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, dim: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_len, dim]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add sinusoidal positional embeddings.

        Args:
            x: Input tensor of shape [B, seq_len, dim]

        Returns:
            x + positional_embeddings with dropout applied
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
