"""Reproducible DACVAE posterior sampling without codec-side changes.

The bundled DACVAE's public ``encode`` method samples from its learned
posterior even in evaluation mode.  This module preserves that sampling
contract while using a call-local :class:`torch.Generator`, so dataset
preparation and reference encoding do not consume or rewind process-global
random state.

Nothing in this module changes DACVAE parameters or source.  In particular,
posterior-mean encoding is intentionally not offered: a mean-only latent has a
different distribution from the samples on which the decoder was trained.
"""

from __future__ import annotations

import hashlib
import re
from numbers import Integral
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

DACVAE_POSTERIOR_SAMPLING_POLICY = "posterior_sample_seeded_v1"
DACVAE_POSTERIOR_SEED_DOMAIN = "nar_vae.dacvae.posterior_sample_seeded_v1"
TORCH_UINT64_SEED_MAX = 2**64 - 1

_SHA256 = re.compile(r"[0-9a-f]{64}")


class DACVAEEncodingError(ValueError):
    """Raised when deterministic DACVAE encoding inputs are invalid."""


def validate_torch_seed(seed: Any) -> int:
    """Return a torch-safe unsigned 64-bit seed.

    ``torch.Generator.manual_seed`` accepts unsigned 64-bit values, but Python
    booleans are integers and values outside that range fail inconsistently at
    the C++ boundary.  Validate the public contract before constructing the
    generator.
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise DACVAEEncodingError("seed must be an unsigned 64-bit integer.")
    result = int(seed)
    if result < 0 or result > TORCH_UINT64_SEED_MAX:
        raise DACVAEEncodingError("seed must be an unsigned 64-bit integer.")
    return result


def _validated_codec_sha256(codec_sha256: Any) -> str:
    if not isinstance(codec_sha256, str) or _SHA256.fullmatch(codec_sha256) is None:
        raise DACVAEEncodingError("codec_sha256 must be a lowercase SHA-256 digest.")
    return codec_sha256


def canonical_mono_float32_pcm(audio: Any) -> bytes:
    """Return stable little-endian bytes for a non-empty ``[T]``/``[C,T]`` waveform.

    Two-dimensional input is always channel-first.  Callers must resample to
    the codec rate before this function; sample-rate conversion is deliberately
    outside the seed policy.  Signed zero is normalized so numerically equal
    PCM does not acquire two content identities.
    """

    if isinstance(audio, torch.Tensor):
        samples = audio.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        try:
            samples = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DACVAEEncodingError(
                "audio must be convertible to a finite float32 waveform."
            ) from exc
    if samples.ndim == 2:
        if samples.shape[0] == 0:
            raise DACVAEEncodingError("audio must contain at least one channel.")
        samples = samples.astype(np.float32, copy=False).mean(axis=0, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise DACVAEEncodingError(
            "audio must be a non-empty mono [T] or channel-first [C, T] waveform."
        )
    if not np.isfinite(samples).all():
        raise DACVAEEncodingError("audio must contain only finite samples.")
    samples = np.ascontiguousarray(samples, dtype="<f4")
    # Canonicalize the otherwise observable sign bit of IEEE-754 zero.
    samples[samples == 0] = 0.0
    return samples.tobytes()


def derive_dacvae_posterior_seed(
    audio: Any,
    *,
    codec_sha256: str,
) -> int:
    """Derive a stable uint64 seed from policy, codec artifact, and canonical PCM.

    ``audio`` must already be the exact mono/resampled signal passed to DACVAE.
    Binding the immutable codec artifact prevents two different posterior
    parameterizations from silently sharing an encoding identity.
    """

    codec_sha256 = _validated_codec_sha256(codec_sha256)
    pcm = canonical_mono_float32_pcm(audio)
    digest = hashlib.sha256()
    digest.update(DACVAE_POSTERIOR_SEED_DOMAIN.encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(codec_sha256))
    digest.update(len(pcm).to_bytes(8, "big", signed=False))
    digest.update(pcm)
    return validate_torch_seed(int.from_bytes(digest.digest()[:8], "big", signed=False))


@torch.inference_mode()
def encode_dacvae_posterior_seeded(
    codec: Any,
    audio_data: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Encode audio with DACVAE's unchanged sampled-posterior formula.

    This is the public ``DACVAE.encode`` computation with only its random draw
    made explicit and call-local::

        mean, scale = quantizer.in_proj(encoder(_pad(audio))).chunk(2, dim=1)
        stdev = softplus(scale) + 1e-4
        latent = randn(generator=local_generator) * stdev + mean

    The helper intentionally uses the codec's existing encoder, padding method,
    and posterior projection. It neither calls nor modifies global RNG state. The
    policy guarantees exact repetition on the same device/backend; PyTorch does
    not promise identical CPU and CUDA random-number streams for one seed.
    """

    seed = validate_torch_seed(seed)
    if not isinstance(audio_data, torch.Tensor):
        raise TypeError("audio_data must be a torch.Tensor.")
    if audio_data.ndim != 3 or audio_data.shape[0] != 1 or audio_data.shape[1] != 1:
        raise DACVAEEncodingError("audio_data must have shape [1, 1, T].")
    if audio_data.shape[-1] <= 0:
        raise DACVAEEncodingError("audio_data must contain at least one sample.")
    if not torch.is_floating_point(audio_data):
        raise DACVAEEncodingError("audio_data must use a floating-point dtype.")
    if not torch.isfinite(audio_data).all():
        raise DACVAEEncodingError("audio_data must contain only finite samples.")

    try:
        padded = codec._pad(audio_data)
        encoded = codec.encoder(padded)
        mean, scale = codec.quantizer.in_proj(encoded).chunk(2, dim=1)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise DACVAEEncodingError(
            "codec must expose a compatible bundled DACVAE _pad, encoder, and "
            "quantizer.in_proj API."
        ) from exc

    if not torch.is_floating_point(mean) or mean.device.type not in {"cpu", "cuda"}:
        raise DACVAEEncodingError(
            "seeded DACVAE posterior sampling supports floating-point CPU/CUDA tensors only."
        )
    try:
        generator = torch.Generator(device=mean.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            layout=mean.layout,
            device=mean.device,
            generator=generator,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DACVAEEncodingError(
            f"Could not construct a local posterior RNG on device {mean.device}."
        ) from exc
    stdev = F.softplus(scale) + 1e-4
    return noise * stdev + mean


__all__ = [
    "DACVAEEncodingError",
    "DACVAE_POSTERIOR_SAMPLING_POLICY",
    "DACVAE_POSTERIOR_SEED_DOMAIN",
    "TORCH_UINT64_SEED_MAX",
    "canonical_mono_float32_pcm",
    "derive_dacvae_posterior_seed",
    "encode_dacvae_posterior_seeded",
    "validate_torch_seed",
]
