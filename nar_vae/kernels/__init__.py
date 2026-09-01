"""
Flash Attention 3 support for NAR-VAE via the `kernels` library.

Uses PyTorch SDPA by default. Flash Attention 3 is an explicit server-side
optimization enabled with ``NAR_VAE_USE_FA3=1``.
"""

import os

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Flash Attention 3 via `kernels` library
# ---------------------------------------------------------------------------
_HAS_FA3 = False
_FA3_LOAD_ATTEMPTED = False
_fa3_kernel = None
_FA3_DTYPES = (torch.float16, torch.bfloat16)


def _load_fa3_kernel():
    """Load Flash Attention 3 on first use, never during package import."""
    global _FA3_LOAD_ATTEMPTED, _HAS_FA3, _fa3_kernel, USING_FA3

    use_fa3 = os.environ.get("NAR_VAE_USE_FA3", "0")
    if _FA3_LOAD_ATTEMPTED or use_fa3 != "1":
        return _fa3_kernel

    _FA3_LOAD_ATTEMPTED = True
    try:
        from kernels import get_kernel

        _fa3_kernel = get_kernel(
            "kernels-community/flash-attn3",
            revision="557701fc200e8964180fafa996316fbd72b854d6",
        )
        _HAS_FA3 = True
    except Exception:
        _HAS_FA3 = False
        _fa3_kernel = None

    USING_FA3 = _HAS_FA3
    return _fa3_kernel


def flash_attention(q, k, v, attn_mask=None, is_causal=False):
    """Flash Attention dispatch: FA3 when available, SDPA fallback otherwise.

    FA3 is used when all conditions are met:
      - NAR_VAE_USE_FA3=1 explicitly opts into the kernel
      - kernels-community/flash-attn3 loaded successfully
      - no arbitrary attention mask (attn_mask is None)
      - dtype is fp16 or bf16 (FA3 requirement)

    Otherwise falls back to PyTorch SDPA (which uses FA2 internally when possible).

    Args:
        q, k, v: [B, S, H, D] (BSHD format)
        attn_mask: optional attention mask
        is_causal: causal masking

    Returns:
        [B, S, H, D]
    """
    fa3_kernel = None
    if attn_mask is None and q.dtype in _FA3_DTYPES:
        fa3_kernel = _load_fa3_kernel()

    if fa3_kernel is not None:
        return fa3_kernel.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=None,
            causal=is_causal,
        )
    # SDPA fallback - transpose to BHSD format
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    return out.transpose(1, 2)


# Expose for introspection
USING_FA3 = _HAS_FA3

__all__ = ["flash_attention", "USING_FA3"]
