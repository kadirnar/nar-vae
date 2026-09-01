import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from nar_vae.kernels import flash_attention as _flash_attention
from nar_vae.tokenization import TOTAL_VOCAB_SIZE

from .reference_resampler import ReferenceResampler
from .text_conditioning import (
    FROZEN_FEATURE_TEXT_CONDITIONING,
    SCRATCH_TOKEN_TEXT_CONDITIONING,
    FrozenTextFeatureAdapter,
    resolve_text_conditioning_metadata,
)

_ROTARY_CACHE_LENGTH = 4096


def _validate_norm_eps(eps: float) -> float:
    if isinstance(eps, bool):
        raise ValueError("norm_eps must be a finite positive number.")
    try:
        value = float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("norm_eps must be a finite positive number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("norm_eps must be a finite positive number.")
    return value


def _non_reentrant_checkpoint_kwargs(
    options: dict | None = None,
) -> dict:
    """Normalize checkpoint options while requiring the modern non-reentrant path."""
    options = dict(options or {})
    if options.pop("use_reentrant", False):
        raise ValueError("EchoDiT activation checkpointing requires use_reentrant=False.")
    options["use_reentrant"] = False
    return options


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Precompute complex exponentials for RoPE."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32)[: (dim // 2)] / dim)
    )
    t = torch.arange(end, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.complex(torch.cos(freqs), torch.sin(freqs))
    return freqs_cis


def _rotary_frequencies(module: nn.Module, end: int, device: torch.device) -> torch.Tensor:
    """Return a device-resident RoPE slice, growing the non-persistent cache if needed."""
    cache = module._rope_cache  # type: ignore[attr-defined]
    if cache.device != device:
        cache = cache.to(device)
    if end > cache.shape[0]:
        cache = precompute_freqs_cis(
            module.head_dim,  # type: ignore[attr-defined]
            max(end, cache.shape[0] * 2),
            device=device,
        )
    if cache is not module._rope_cache:  # type: ignore[attr-defined]
        module._rope_cache = cache  # type: ignore[attr-defined]
    return cache[:end]


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:3], -1, 2))
    x_ = x_ * freqs_cis[..., None, :]
    x_ = torch.view_as_real(x_).reshape(x.shape)
    return x_.type_as(x)


def get_timestep_embedding(
    timestep: torch.Tensor,
    embed_size: int,
    frequencies: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings for diffusion."""
    assert embed_size % 2 == 0

    half = embed_size // 2

    if frequencies is None:
        frequencies = 1000 * torch.exp(
            -torch.log(torch.tensor(10000.0, device=timestep.device))
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timestep.device)
            / half
        )

    args = timestep[..., None] * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding.to(timestep.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, model_size: int | tuple[int, ...], eps: float = 1e-6):
        super().__init__()
        self.eps = _validate_norm_eps(eps)

        if isinstance(model_size, int):
            model_size = (model_size,)
        self.weight = nn.Parameter(torch.ones(model_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(torch.pow(x.float(), 2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight
        return x.to(x_dtype)


class LowRankAdaLN(nn.Module):
    """
    Low-Rank Adaptive Layer Normalization.

    More efficient than standard AdaLN by using low-rank factorization
    of the modulation parameters.
    """

    def __init__(self, model_size: int, rank: int, eps: float = 1e-6):
        super().__init__()
        self.eps = _validate_norm_eps(eps)

        # Low-rank down projections
        self.shift_down = nn.Linear(model_size, rank, bias=False)
        self.scale_down = nn.Linear(model_size, rank, bias=False)
        self.gate_down = nn.Linear(model_size, rank, bias=False)

        # Low-rank up projections
        self.shift_up = nn.Linear(rank, model_size, bias=True)
        self.scale_up = nn.Linear(rank, model_size, bias=True)
        self.gate_up = nn.Linear(rank, model_size, bias=True)
        # DiT-style adaLN-Zero: modulation residuals start inactive. These tensors
        # retain their historical names/shapes, so strict checkpoint loads replace
        # the scratch initialization without migration.
        for projection in (self.shift_up, self.scale_up, self.gate_up):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, T, D]
            cond_embed: Conditioning (timestep) embedding [B, 1, D*3]

        Returns:
            Tuple of (normalized_x, gate)
        """
        shift, scale, gate = cond_embed.chunk(3, dim=-1)

        # Low-rank factorization: down -> up with residual
        shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(F.silu(gate))) + gate

        # RMS normalization
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(torch.pow(x.float(), 2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (scale + 1) + shift

        gate = torch.tanh(gate)

        return x.to(x_dtype), gate


class JointAttention(nn.Module):
    """
    Joint attention to text, speaker, and latent contexts.

    This is the key innovation: attending to multiple context types
    simultaneously with proper masking and KV caching.
    """

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        text_model_size: int,
        speaker_model_size: int,
        speaker_patch_size: int,
        use_speaker_conditioning: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        del speaker_patch_size
        self.use_speaker_conditioning = use_speaker_conditioning
        self.num_heads = num_heads

        # Query projections (for main sequence)
        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)

        # Key/Value projections for text context
        self.wk_text = nn.Linear(text_model_size, model_size, bias=False)
        self.wv_text = nn.Linear(text_model_size, model_size, bias=False)

        # Speaker projections are part of the topology only for a checkpoint that
        # actually learns reference-audio conditioning.
        self.wk_speaker = (
            nn.Linear(speaker_model_size, model_size, bias=False)
            if use_speaker_conditioning
            else None
        )
        self.wv_speaker = (
            nn.Linear(speaker_model_size, model_size, bias=False)
            if use_speaker_conditioning
            else None
        )

        assert model_size % num_heads == 0
        self.head_dim = model_size // num_heads

        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)

        self.gate = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)

    def _apply_rotary_half(self, y: torch.Tensor, fc: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to only half of the heads (for speaker/latent contexts)."""
        y1, y2 = y.chunk(2, dim=-2)
        y1 = apply_rotary_emb(y1, fc)
        return torch.cat([y1, y2], dim=-2)

    def forward(
        self,
        x: torch.Tensor,
        text_mask: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        kv_cache_text: tuple[torch.Tensor, torch.Tensor],
        kv_cache_speaker: tuple[torch.Tensor, torch.Tensor],
        start_pos: int | None = None,
        kv_cache_latent: tuple[torch.Tensor, torch.Tensor] | None = None,
        self_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with joint attention.

        Args:
            x: Input sequence [B, T, D]
            text_mask: Text attention mask [B, L_text]
            speaker_mask: Speaker attention mask [B, L_speaker]
            freqs_cis: RoPE frequencies
            kv_cache_text: Cached text K, V
            kv_cache_speaker: Cached speaker K, V
            start_pos: Removed autoregressive offset; only ``None`` or zero is accepted
            kv_cache_latent: Removed autoregressive context; must be ``None``
            self_mask: Valid positions in the current latent sequence [B, T]

        Returns:
            Attention output [B, T, D]
        """
        batch_size, seq_len = x.shape[:2]

        # Self-attention Q, K, V
        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xk_self = self.wk(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xv_self = self.wv(x).reshape(batch_size, seq_len, self.num_heads, -1)

        xq = self.q_norm(xq)
        xk_self = self.k_norm(xk_self)

        gate = self.gate(x)

        if start_pos not in (None, 0) or kv_cache_latent is not None:
            raise ValueError(
                "NAR-VAE has no autoregressive latent-prefix path; "
                "start_pos must be zero and kv_cache_latent must be None."
            )
        freqs_q = freqs_cis[:seq_len]

        # Apply RoPE to query and self-keys (only half)
        xq = self._apply_rotary_half(xq, freqs_q)
        xk_self = self._apply_rotary_half(xk_self, freqs_q)

        # Get cached text and speaker K, V
        xk_text, xv_text = kv_cache_text
        xk_speaker, xv_speaker = kv_cache_speaker

        if not self.use_speaker_conditioning and xk_speaker.shape[1] != 0:
            raise ValueError("A speaker-disabled NAR-VAE block requires an empty speaker cache.")
        if not self.use_speaker_conditioning and speaker_mask is not None:
            raise ValueError("speaker_mask requires a speaker-conditioned NAR-VAE checkpoint.")

        key_parts = [xk_self, xk_text, xk_speaker]
        value_parts = [xv_self, xv_text, xv_speaker]
        xk = torch.cat(key_parts, dim=1)
        xv = torch.cat(value_parts, dim=1)

        # ``None`` is the all-valid sentinel. It avoids a GPU-to-CPU sync from
        # ``mask.all()`` in every DiT block and lets SDPA choose its fused path.
        if self_mask is None and text_mask is None and speaker_mask is None:
            output = _flash_attention(xq, xk, xv, attn_mask=None, is_causal=False)
        else:
            if self_mask is None:
                self_mask = torch.ones(
                    (batch_size, seq_len),
                    dtype=torch.bool,
                    device=x.device,
                )
            elif tuple(self_mask.shape) != (batch_size, seq_len):
                raise ValueError(
                    "Latent mask must have shape "
                    f"{(batch_size, seq_len)}; got {tuple(self_mask.shape)}."
                )
            else:
                self_mask = self_mask.to(device=x.device, dtype=torch.bool)

            mask_parts = [self_mask]
            mask_parts.append(
                text_mask
                if text_mask is not None
                else torch.ones((batch_size, xk_text.shape[1]), dtype=torch.bool, device=x.device)
            )
            mask_parts.append(
                speaker_mask
                if speaker_mask is not None
                else torch.ones(
                    (batch_size, xk_speaker.shape[1]), dtype=torch.bool, device=x.device
                )
            )
            mask = torch.cat(mask_parts, dim=1)
            mask = mask[:, None, None]  # [B, 1, 1, total_seq_len]
            output = F.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=mask,
                is_causal=False,
            ).transpose(1, 2)

        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate)  # Gated output

        output = self.wo(output)

        return output

    def get_kv_cache_text(self, text_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute K, V for text context."""
        batch_size = text_state.shape[0]
        xk = self.wk_text(text_state).reshape(batch_size, text_state.shape[1], self.num_heads, -1)
        xv = self.wv_text(text_state).reshape(batch_size, text_state.shape[1], self.num_heads, -1)
        xk = self.k_norm(xk)
        return xk, xv

    def get_kv_cache_speaker(
        self, speaker_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute K, V for speaker context."""
        if self.wk_speaker is None or self.wv_speaker is None:
            raise RuntimeError("This NAR-VAE block has no speaker-conditioning projections.")
        batch_size = speaker_state.shape[0]
        xk = self.wk_speaker(speaker_state).reshape(
            batch_size, speaker_state.shape[1], self.num_heads, -1
        )
        xv = self.wv_speaker(speaker_state).reshape(
            batch_size, speaker_state.shape[1], self.num_heads, -1
        )
        xk = self.k_norm(xk)
        return xk, xv

    def empty_context_cache(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-token KV pair without allocating learned null context."""
        shape = (batch_size, 0, self.num_heads, self.head_dim)
        empty = torch.empty(shape, device=device, dtype=dtype)
        return empty, empty


class MLP(nn.Module):
    """SwiGLU MLP as used in LLaMA."""

    def __init__(self, model_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SelfAttention(nn.Module):
    """Standard self-attention for encoder blocks."""

    def __init__(self, model_size: int, num_heads: int, is_causal: bool, norm_eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)
        self.gate = nn.Linear(model_size, model_size, bias=False)

        assert model_size % num_heads == 0
        self.q_norm = RMSNorm((num_heads, model_size // num_heads), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, model_size // num_heads), eps=norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[:2]

        xq = self.wq(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xk = self.wk(x).reshape(batch_size, seq_len, self.num_heads, -1)
        xv = self.wv(x).reshape(batch_size, seq_len, self.num_heads, -1)

        gate = self.gate(x)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = apply_rotary_emb(xq, freqs_cis[:seq_len])
        xk = apply_rotary_emb(xk, freqs_cis[:seq_len])

        if mask is not None:
            assert mask.ndim == 2  # (B, S)
            mask = mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
            # PyTorch 2.2 rejects an explicit attention mask together with
            # ``is_causal=True``. Compose the causal and key-padding masks here
            # so the advertised minimum version follows the same semantics.
            if self.is_causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len),
                    dtype=torch.bool,
                    device=x.device,
                ).tril()
                mask = mask & causal_mask[None, None, :, :]
            # Use SDPA when a padding mask is present (FA3 does not support it).
            output = F.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                attn_mask=mask,
                is_causal=False,
            ).transpose(1, 2)
        else:
            # Use FA3 when no mask (faster path)
            output = _flash_attention(xq, xk, xv, attn_mask=None, is_causal=self.is_causal)

        output = output.reshape(batch_size, seq_len, -1)
        output = output * torch.sigmoid(gate)

        output = self.wo(output)

        return output


class EncoderTransformerBlock(nn.Module):
    """Transformer block for encoders (text, speaker)."""

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        is_causal: bool,
        norm_eps: float,
    ):
        super().__init__()
        self.attention = SelfAttention(
            model_size=model_size, num_heads=num_heads, is_causal=is_causal, norm_eps=norm_eps
        )
        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)

        self.attention_norm = RMSNorm(model_size, norm_eps)
        self.mlp_norm = RMSNorm(model_size, norm_eps)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, freqs_cis: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), mask, freqs_cis)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DiTBlock(nn.Module):
    """
    DiT block with joint attention and low-rank AdaLN.

    This is the main building block of the EchoDiT architecture.
    """

    def __init__(
        self,
        model_size: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        text_model_size: int,
        speaker_model_size: int,
        speaker_patch_size: int,
        adaln_rank: int,
        use_speaker_conditioning: bool = True,
        layer_index: int = 0,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.attention = JointAttention(
            model_size=model_size,
            num_heads=num_heads,
            text_model_size=text_model_size,
            speaker_model_size=speaker_model_size,
            speaker_patch_size=speaker_patch_size,
            use_speaker_conditioning=use_speaker_conditioning,
            norm_eps=norm_eps,
        )

        self.mlp = MLP(model_size=model_size, intermediate_size=intermediate_size)

        self.attention_adaln = LowRankAdaLN(model_size=model_size, rank=adaln_rank, eps=norm_eps)
        self.mlp_adaln = LowRankAdaLN(model_size=model_size, rank=adaln_rank, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,
        text_mask: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        freqs_cis: torch.Tensor,
        kv_cache_text: list[tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        kv_cache_speaker: list[tuple[torch.Tensor, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor],
        start_pos: int | None,
        kv_cache_latent: list[tuple[torch.Tensor, torch.Tensor]]
        | tuple[torch.Tensor, torch.Tensor]
        | None,
        *,
        hidden_states: torch.Tensor | None = None,
        latent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one layer using its own entries from the shared KV caches.

        Cache-DiT temporarily replaces ``EchoDiT.blocks`` with one grouped
        block. Passing the full cache lists through that group keeps every
        original layer paired with its own text, speaker, and latent KV state.

        ``hidden_states`` is a Pattern 3 signature marker. The grouped wrapper
        forwards that tensor positionally, preserving the existing ``x`` API.
        """
        if hidden_states is not None:
            raise TypeError("Pass the hidden state as the first argument or with x=.")
        layer_kv_text = (
            kv_cache_text[self.layer_index] if isinstance(kv_cache_text, list) else kv_cache_text
        )
        layer_kv_speaker = (
            kv_cache_speaker[self.layer_index]
            if isinstance(kv_cache_speaker, list)
            else kv_cache_speaker
        )
        if kv_cache_latent is not None:
            raise ValueError("NAR-VAE blocks do not accept autoregressive latent-prefix caches.")

        # Attention with AdaLN
        x_norm, attention_gate = self.attention_adaln(x, cond_embed)
        attention_kwargs = {}
        if latent_mask is not None:
            attention_kwargs["self_mask"] = latent_mask
        x = x + attention_gate * self.attention(
            x_norm,
            text_mask,
            speaker_mask,
            freqs_cis,
            layer_kv_text,
            layer_kv_speaker,
            start_pos,
            None,
            **attention_kwargs,
        )

        # MLP with AdaLN
        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x = x + mlp_gate * self.mlp(x_norm)

        return x


class TextEncoder(nn.Module):
    """Text encoder (similar to BERT/RoBERTa style)."""

    def __init__(
        self,
        vocab_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
        num_languages: int = 0,
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, model_size)
        self.language_embedding = (
            nn.Embedding(num_languages + 1, model_size, padding_idx=0)
            if num_languages > 0
            else None
        )

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                is_causal=False,
                norm_eps=norm_eps,
            )
            self.blocks.append(block)

        self.head_dim = model_size // num_heads
        self.register_buffer(
            "_rope_cache",
            precompute_freqs_cis(self.head_dim, _ROTARY_CACHE_LENGTH),
            persistent=False,
        )
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = _non_reentrant_checkpoint_kwargs()

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.text_embedding(input_ids)
        if self.language_embedding is not None:
            if language_ids is None:
                raise ValueError("language_ids are required by this multilingual text encoder.")
            if language_ids.ndim == 1:
                if language_ids.shape[0] != input_ids.shape[0]:
                    raise ValueError("language_ids must have shape [batch] or [batch, token].")
                language_state = self.language_embedding(language_ids.to(input_ids.device))[
                    :, None, :
                ]
            elif language_ids.ndim == 2:
                if tuple(language_ids.shape) != tuple(input_ids.shape):
                    raise ValueError("language_ids must have shape [batch] or [batch, token].")
                language_state = self.language_embedding(language_ids.to(input_ids.device))
            else:
                raise ValueError("language_ids must have shape [batch] or [batch, token].")
            x = x + language_state
        elif language_ids is not None:
            raise ValueError("language_ids require a language-conditioned text encoder.")

        freqs_cis = _rotary_frequencies(self, input_ids.shape[1], x.device)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                x = activation_checkpoint(
                    block,
                    x,
                    mask,
                    freqs_cis,
                    **self._gradient_checkpointing_kwargs,
                )
            else:
                x = block(x, mask, freqs_cis)

        return x


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder with patching.

    Processes speaker embeddings or reference audio in patches.
    """

    def __init__(
        self,
        latent_size: int,
        patch_size: int,
        model_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.in_proj = nn.Linear(latent_size * patch_size, model_size, bias=True)
        # A learned summary query preserves a stable, explicitly global timbre
        # channel while all temporal reference patches remain available to DiT.
        self.global_timbre_token = nn.Parameter(torch.zeros(1, 1, model_size))

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = EncoderTransformerBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                # Reference audio is fully available before diffusion starts. A
                # bidirectional encoder lets the summary and every patch use the
                # complete recording instead of inheriting Echo's continuation mask.
                is_causal=False,
                norm_eps=norm_eps,
            )
            self.blocks.append(block)

        self.head_dim = model_size // num_heads
        self.register_buffer(
            "_rope_cache",
            precompute_freqs_cis(self.head_dim, _ROTARY_CACHE_LENGTH),
            persistent=False,
        )
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = _non_reentrant_checkpoint_kwargs()

    def forward(
        self,
        latent: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Reshape to patches: [B, D, T] -> [B, T//patch_size, D*patch_size]
        # Input: [B, D, T], patch along time dimension
        if latent.ndim != 3:
            raise ValueError("Speaker latent must have shape [batch, channels, frames].")
        if latent.shape[-1] % self.patch_size:
            raise ValueError(
                f"Speaker latent has {latent.shape[-1]} frames; expected a multiple "
                f"of patch size {self.patch_size}."
            )
        x = (
            latent.unfold(-1, self.patch_size, self.patch_size)
            .permute(0, 2, 1, 3)
            .reshape(
                latent.shape[0],
                latent.shape[-1] // self.patch_size,
                latent.shape[-2] * self.patch_size,
            )
        )

        x = self.in_proj(x)
        x = x / 6.0  # Activation scaling (from paper)
        patch_count = x.shape[1]
        global_token = self.global_timbre_token.to(dtype=x.dtype).expand(x.shape[0], -1, -1)
        x = torch.cat((global_token, x), dim=1)

        if mask is not None:
            patch_shape = (latent.shape[0], patch_count)
            state_shape = (latent.shape[0], patch_count + 1)
            if tuple(mask.shape) == patch_shape:
                mask = torch.cat(
                    (
                        torch.ones((latent.shape[0], 1), dtype=torch.bool, device=x.device),
                        mask.to(device=x.device, dtype=torch.bool),
                    ),
                    dim=1,
                )
            elif tuple(mask.shape) != state_shape:
                raise ValueError(
                    "Speaker mask must have patch shape "
                    f"{patch_shape} or encoded-state shape {state_shape}; got {tuple(mask.shape)}."
                )
            else:
                mask = mask.to(device=x.device, dtype=torch.bool)

        freqs_cis = _rotary_frequencies(self, x.shape[1], x.device)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                x = activation_checkpoint(
                    block,
                    x,
                    mask,
                    freqs_cis,
                    **self._gradient_checkpointing_kwargs,
                )
            else:
                x = block(x, mask, freqs_cis)

        return x


class EchoDiT(nn.Module):
    """
    EchoDiT: Efficient DiT with Joint Attention for Flow Matching TTS.

    This is adapted from the EchoDiT paper for DACVAE-based flow matching.
    """

    def __init__(
        self,
        latent_size: int = 1024,  # DACVAE latent dimension
        #
        model_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        norm_eps: float = 1e-6,
        #
        text_vocab_size: int = TOTAL_VOCAB_SIZE,
        text_model_size: int = 768,
        text_num_layers: int = 6,
        text_num_heads: int = 12,
        text_intermediate_size: int = 3072,
        text_conditioning_mode: str = SCRATCH_TOKEN_TEXT_CONDITIONING,
        conditioning_feature_size: int | None = None,
        #
        speaker_patch_size: int = 4,
        speaker_model_size: int = 512,
        speaker_num_layers: int = 4,
        speaker_num_heads: int = 8,
        speaker_intermediate_size: int = 2048,
        #
        timestep_embed_size: int = 256,
        adaln_rank: int = 128,
        num_languages: int = 0,
        use_speaker_conditioning: bool = False,
        use_duration_alignment: bool = False,
        target_patch_size: int = 1,
        speaker_num_summary_tokens: int = 0,
    ):
        super().__init__()
        if isinstance(target_patch_size, bool) or not isinstance(target_patch_size, Integral):
            raise ValueError("target_patch_size must be a positive integer.")
        if target_patch_size <= 0:
            raise ValueError("target_patch_size must be a positive integer.")
        if (
            isinstance(speaker_num_summary_tokens, bool)
            or not isinstance(speaker_num_summary_tokens, Integral)
            or speaker_num_summary_tokens < 0
        ):
            raise ValueError("speaker_num_summary_tokens must be zero or a positive integer.")
        if speaker_num_summary_tokens > 0 and not use_speaker_conditioning:
            raise ValueError("speaker_num_summary_tokens requires use_speaker_conditioning=True.")
        self.speaker_patch_size = speaker_patch_size
        self.speaker_num_summary_tokens = int(speaker_num_summary_tokens)
        self.target_patch_size = int(target_patch_size)
        self.speaker_model_size = speaker_model_size
        self.use_speaker_conditioning = use_speaker_conditioning
        self.use_duration_alignment = use_duration_alignment
        self.timestep_embed_size = timestep_embed_size
        self.latent_size = latent_size

        self.text_conditioning = resolve_text_conditioning_metadata(
            text_conditioning_mode,
            conditioning_feature_size,
        )
        if self.text_conditioning.mode == FROZEN_FEATURE_TEXT_CONDITIONING:
            if text_num_layers != 0:
                raise ValueError(
                    "frozen_features text conditioning requires text_num_layers=0; "
                    "the acoustic checkpoint must not train a replacement text Transformer."
                )
            self.text_encoder = FrozenTextFeatureAdapter(
                feature_size=self.text_conditioning.feature_size,
                model_size=text_model_size,
                num_languages=num_languages,
            )
        else:
            # Legacy/default topology: acoustic pretraining learns token embeddings
            # and a bidirectional text Transformer from scratch.
            self.text_encoder = TextEncoder(
                vocab_size=text_vocab_size,
                model_size=text_model_size,
                num_layers=text_num_layers,
                num_heads=text_num_heads,
                intermediate_size=text_intermediate_size,
                norm_eps=norm_eps,
                num_languages=num_languages,
            )
        self.global_language_embedding = (
            nn.Embedding(num_languages + 1, timestep_embed_size, padding_idx=0)
            if num_languages > 0
            else None
        )

        # Speaker conditioning is a versioned topology choice. Text-only models
        # carry neither a learned constant speaker branch nor unused speaker state.
        self.speaker_encoder = (
            SpeakerEncoder(
                latent_size=latent_size,
                patch_size=speaker_patch_size,
                model_size=speaker_model_size,
                num_layers=speaker_num_layers,
                num_heads=speaker_num_heads,
                intermediate_size=speaker_intermediate_size,
                norm_eps=norm_eps,
            )
            if use_speaker_conditioning
            else None
        )
        # This adapter is absent from the legacy topology, leaving its state-dict
        # keys and numerical path unchanged when speaker_num_summary_tokens == 0.
        self.speaker_resampler = (
            ReferenceResampler(
                hidden_size=speaker_model_size,
                num_queries=self.speaker_num_summary_tokens,
                num_heads=speaker_num_heads,
                norm_eps=norm_eps,
            )
            if self.speaker_num_summary_tokens > 0
            else None
        )

        self.text_norm = RMSNorm(text_model_size, norm_eps)
        self.speaker_norm = (
            RMSNorm(speaker_model_size, norm_eps) if use_speaker_conditioning else None
        )

        # Conditioning module (timestep -> 3*model_size for shift, scale, gate)
        self.cond_module = nn.Sequential(
            nn.Linear(timestep_embed_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size * 3, bias=False),
        )
        # Together with the zeroed LowRankAdaLN up projections, this makes every
        # residual block an identity map at scratch initialization.
        nn.init.zeros_(self.cond_module[-1].weight)

        # Input projection
        self.in_proj = nn.Linear(latent_size * self.target_patch_size, model_size, bias=True)
        self.frame_text_proj = (
            nn.Linear(
                text_model_size * self.target_patch_size,
                model_size,
                bias=False,
            )
            if use_duration_alignment
            else None
        )

        # DiT blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            block = DiTBlock(
                model_size=model_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                norm_eps=norm_eps,
                text_model_size=text_model_size,
                speaker_model_size=speaker_model_size,
                speaker_patch_size=speaker_patch_size,
                adaln_rank=adaln_rank,
                use_speaker_conditioning=use_speaker_conditioning,
                layer_index=i,
            )
            self.blocks.append(block)

        # Output projection
        self.out_norm = RMSNorm(model_size, norm_eps)
        self.out_proj = nn.Linear(
            model_size,
            latent_size * self.target_patch_size,
            bias=True,
        )

        # Zero-initialize output for stability
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.head_dim = model_size // num_heads
        self.register_buffer(
            "_rope_cache",
            precompute_freqs_cis(self.head_dim, _ROTARY_CACHE_LENGTH),
            persistent=False,
        )
        timestep_half = timestep_embed_size // 2
        self.register_buffer(
            "_timestep_frequencies",
            1000
            * torch.exp(
                -torch.log(torch.tensor(10000.0))
                * torch.arange(timestep_half, dtype=torch.float32)
                / timestep_half
            ),
            persistent=False,
        )
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs = _non_reentrant_checkpoint_kwargs()

    def forward(
        self,
        x: torch.Tensor,  # [B, D, T] latents
        t: torch.Tensor,  # [B] timesteps
        text_mask: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        kv_cache_text: list[tuple[torch.Tensor, torch.Tensor]],
        kv_cache_speaker: list[tuple[torch.Tensor, torch.Tensor]],
        start_pos: int | None = None,
        kv_cache_latent: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        latent_mask: torch.Tensor | None = None,
        frame_text_state: torch.Tensor | None = None,
        global_language_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy latents [B, latent_size, T]
            t: Timesteps [B]
            text_mask: Text mask [B, L_text]
            speaker_mask: Speaker mask [B, L_speaker]
            kv_cache_text: List of (K, V) for each layer
            kv_cache_speaker: List of (K, V) for each layer
            start_pos: Removed autoregressive offset; only ``None`` or zero is accepted
            kv_cache_latent: Removed autoregressive context; must be ``None``
            latent_mask: Valid positions in the current latent sequence [B, T]
            frame_text_state: MAS-regulated text state [B, T, text_model_size]

        Returns:
            Predicted velocity [B, latent_size, T]
        """
        if start_pos not in (None, 0) or kv_cache_latent is not None:
            raise ValueError(
                "NAR-VAE is non-autoregressive; start_pos must be zero and "
                "kv_cache_latent must be None."
            )
        if x.ndim != 3 or x.shape[1] != self.latent_size:
            raise ValueError(f"Noisy latents must have shape [batch, {self.latent_size}, frames].")
        original_frames = x.shape[-1]
        padded_frames = (
            (original_frames + self.target_patch_size - 1)
            // self.target_patch_size
            * self.target_patch_size
        )
        padding_frames = padded_frames - original_frames

        packed_latent_mask = None
        if latent_mask is not None:
            expected_shape = (x.shape[0], original_frames)
            if tuple(latent_mask.shape) != expected_shape:
                raise ValueError(
                    f"Latent mask must have shape {expected_shape}; got {tuple(latent_mask.shape)}."
                )
            latent_mask = latent_mask.to(device=x.device, dtype=torch.bool)
            # A packed token contains multiple exact frame coordinates. Invalid
            # coordinates must not leak into valid coordinates through in_proj.
            if self.target_patch_size > 1:
                x = x.masked_fill(~latent_mask[:, None, :], 0.0)
            packed_latent_mask = F.pad(latent_mask, (0, padding_frames), value=False)
            packed_latent_mask = packed_latent_mask.reshape(
                x.shape[0],
                padded_frames // self.target_patch_size,
                self.target_patch_size,
            ).any(dim=-1)

        if padding_frames:
            x = F.pad(x, (0, padding_frames))
        # Exact target patching: [B, D, T] -> [B, T/P, P*D]. Every latent
        # coordinate is retained and projected jointly; no averaging/resampling.
        x = x.transpose(1, 2).reshape(
            x.shape[0],
            padded_frames // self.target_patch_size,
            self.latent_size * self.target_patch_size,
        )
        freqs_cis = _rotary_frequencies(self, x.shape[1], x.device)

        # Note: speaker_mask is already at patch level (created in flow_matching.py)

        # Timestep conditioning
        timestep_state = get_timestep_embedding(
            t,
            self.timestep_embed_size,
            self._timestep_frequencies,
        )
        if self.global_language_embedding is None:
            if global_language_state is not None:
                raise ValueError(
                    "global_language_state requires a language-conditioned NAR-VAE checkpoint."
                )
        else:
            if global_language_state is None:
                raise ValueError(
                    "global_language_state is required by a language-conditioned NAR-VAE "
                    "checkpoint."
                )
            expected_shape = (x.shape[0], self.timestep_embed_size)
            if tuple(global_language_state.shape) != expected_shape:
                raise ValueError(
                    f"global_language_state must have shape {expected_shape}; "
                    f"got {tuple(global_language_state.shape)}."
                )
            timestep_state = timestep_state + global_language_state.to(dtype=timestep_state.dtype)
        cond_embed = self.cond_module(timestep_state)
        cond_embed = cond_embed[:, None]  # [B, 1, 3*model_size]

        # Project input
        x = self.in_proj(x)
        if self.frame_text_proj is None:
            if frame_text_state is not None:
                raise ValueError("frame_text_state requires a duration-aligned NAR-VAE checkpoint.")
        else:
            frame_text_size = self.frame_text_proj.in_features // self.target_patch_size
            expected_shape = (x.shape[0], original_frames, frame_text_size)
            if frame_text_state is None or tuple(frame_text_state.shape) != expected_shape:
                actual_shape = None if frame_text_state is None else tuple(frame_text_state.shape)
                raise ValueError(
                    f"Duration-aligned text state must have shape {expected_shape}; "
                    f"got {actual_shape}."
                )
            frame_text_state = frame_text_state.to(device=x.device, dtype=x.dtype)
            if latent_mask is not None and self.target_patch_size > 1:
                frame_text_state = frame_text_state.masked_fill(
                    ~latent_mask[:, :, None],
                    0.0,
                )
            if padding_frames:
                frame_text_state = F.pad(frame_text_state, (0, 0, 0, padding_frames))
            frame_text_state = frame_text_state.reshape(
                x.shape[0],
                padded_frames // self.target_patch_size,
                frame_text_size * self.target_patch_size,
            )
            x = x + self.frame_text_proj(frame_text_state)

        # Apply DiT blocks
        for block in self.blocks:
            block_kwargs = dict(
                cond_embed=cond_embed,
                text_mask=text_mask,
                speaker_mask=speaker_mask,
                freqs_cis=freqs_cis,
                kv_cache_text=kv_cache_text,
                kv_cache_speaker=kv_cache_speaker,
                start_pos=None,
                kv_cache_latent=None,
            )
            if packed_latent_mask is not None:
                block_kwargs["latent_mask"] = packed_latent_mask
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                x = activation_checkpoint(
                    block,
                    x,
                    **block_kwargs,
                    **self._gradient_checkpointing_kwargs,
                )
            else:
                x = block(x, **block_kwargs)

        # Output projection
        x = self.out_norm(x)
        x = self.out_proj(x)

        # Exact inverse of the target packing; crop only synthetic padding.
        x = x.reshape(x.shape[0], padded_frames, self.latent_size)
        x = x[:, :original_frames].transpose(1, 2)

        return x.float()

    def get_kv_cache_text(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        *,
        token_language_ids: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Precompute text KV cache."""
        text_state = self.encode_text(
            text_input_ids,
            text_mask,
            language_ids,
            token_language_ids=token_language_ids,
            conditioning_features=conditioning_features,
        )
        return self.project_text_kv_cache(text_state)

    def project_text_kv_cache(
        self,
        text_state: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Project an already encoded text state into per-layer KV caches."""
        return [block.attention.get_kv_cache_text(text_state) for block in self.blocks]

    def encode_text(
        self,
        text_input_ids: torch.Tensor,
        text_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        *,
        token_language_ids: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode normalized text state for KV projection or auxiliary heads."""
        if text_input_ids.ndim != 2:
            raise ValueError("text_input_ids must have shape [batch, token].")
        if token_language_ids is not None and language_ids is not None:
            if language_ids.ndim != 1:
                raise ValueError("Global language_ids must have shape [batch].")
            text_language_ids = token_language_ids
        else:
            text_language_ids = (
                token_language_ids if token_language_ids is not None else language_ids
            )
        if self.text_conditioning.mode == SCRATCH_TOKEN_TEXT_CONDITIONING:
            if conditioning_features is not None:
                raise ValueError("conditioning_features require frozen_features text conditioning.")
            text_state = self.text_encoder(text_input_ids, text_mask, text_language_ids)
        else:
            if conditioning_features is None:
                raise ValueError(
                    "conditioning_features are required by frozen_features text conditioning."
                )
            if tuple(conditioning_features.shape[:2]) != tuple(text_input_ids.shape):
                raise ValueError(
                    "conditioning_features must share the [batch, token] shape of text_input_ids."
                )
            text_state = self.text_encoder(conditioning_features, text_language_ids)
        return self.text_norm(text_state)

    def encode_global_language(
        self,
        language_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Encode the utterance-level target language used by every DiT block."""
        if self.global_language_embedding is None:
            if language_ids is not None:
                raise ValueError("language_ids require a language-conditioned NAR-VAE checkpoint.")
            return None
        if language_ids is None:
            raise ValueError(
                "language_ids are required by a language-conditioned NAR-VAE checkpoint."
            )
        if language_ids.ndim != 1 or language_ids.shape[0] != batch_size:
            raise ValueError("Global language_ids must have shape [batch].")
        return self.global_language_embedding(language_ids.to(device=device, dtype=torch.long))

    def get_kv_cache_speaker(
        self,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Precompute speaker KV cache."""
        speaker_state = self.encode_speaker(speaker_latent, speaker_mask)
        return self.project_speaker_kv_cache(speaker_state)

    def project_speaker_kv_cache(
        self,
        speaker_state: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Project an already encoded speaker state into per-layer KV caches."""
        if not self.use_speaker_conditioning:
            return [
                block.attention.empty_context_cache(
                    batch_size=speaker_state.shape[0],
                    device=speaker_state.device,
                    dtype=speaker_state.dtype,
                )
                for block in self.blocks
            ]
        return [block.attention.get_kv_cache_speaker(speaker_state) for block in self.blocks]

    def encode_speaker(
        self,
        speaker_latent: torch.Tensor,
        speaker_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode normalized speaker state for KV projection or auxiliary heads."""
        if not self.use_speaker_conditioning:
            if speaker_mask is not None:
                raise ValueError("speaker_mask requires a speaker-conditioned checkpoint.")
            return torch.empty(
                (speaker_latent.shape[0], 0, self.speaker_model_size),
                device=speaker_latent.device,
                dtype=self.in_proj.weight.dtype,
            )
        assert self.speaker_encoder is not None
        assert self.speaker_norm is not None
        speaker_state = self.speaker_encoder(speaker_latent, speaker_mask)
        if self.speaker_resampler is not None:
            # SpeakerEncoder always prepends its contextual global-timbre state.
            # Make that state valid when callers provide the patch-only mask, then
            # consume the variable mask here. Every emitted summary query is valid.
            resampler_mask = speaker_mask
            if resampler_mask is not None:
                patch_mask_shape = (speaker_state.shape[0], speaker_state.shape[1] - 1)
                state_mask_shape = tuple(speaker_state.shape[:2])
                if tuple(resampler_mask.shape) == patch_mask_shape:
                    resampler_mask = torch.cat(
                        (
                            torch.ones(
                                (speaker_state.shape[0], 1),
                                dtype=torch.bool,
                                device=speaker_state.device,
                            ),
                            resampler_mask.to(device=speaker_state.device, dtype=torch.bool),
                        ),
                        dim=1,
                    )
                elif tuple(resampler_mask.shape) != state_mask_shape:
                    # SpeakerEncoder should already reject this; retain a local
                    # fail-closed check in case a compatible encoder is substituted.
                    raise RuntimeError(
                        "Speaker resampler mask must match encoder patches or states."
                    )
                else:
                    resampler_mask = resampler_mask.to(
                        device=speaker_state.device,
                        dtype=torch.bool,
                    ).clone()
                    # The leading state is model-owned, not input padding. Keep
                    # the contextual global-timbre channel available even if a
                    # caller supplied an encoded-state-shaped padding mask.
                    resampler_mask[:, 0] = True
            speaker_state = self.speaker_resampler(speaker_state, resampler_mask)
        return self.speaker_norm(speaker_state)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Checkpoint encoder and DiT blocks with PyTorch's non-reentrant implementation."""
        options = _non_reentrant_checkpoint_kwargs(gradient_checkpointing_kwargs)
        self.gradient_checkpointing = True
        self._gradient_checkpointing_kwargs = options
        encoders = [self.text_encoder]
        if self.speaker_encoder is not None:
            encoders.append(self.speaker_encoder)
        for encoder in encoders:
            encoder.gradient_checkpointing = True
            encoder._gradient_checkpointing_kwargs = dict(options)

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing throughout the backbone."""
        self.gradient_checkpointing = False
        encoders = [self.text_encoder]
        if self.speaker_encoder is not None:
            encoders.append(self.speaker_encoder)
        for encoder in encoders:
            encoder.gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
