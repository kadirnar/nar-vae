"""Versioned latent-normalization statistics without codec-side changes.

The contract in this module is intentionally independent from DACVAE.  It records
the provenance needed to reproduce a set of statistics and applies only an
invertible transform at the model boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

LATENT_NORMALIZATION_SCHEMA_VERSION = 1
LATENT_NORMALIZATION_MODES = ("none", "per_channel_v1")
LatentNormalizationMode = Literal["none", "per_channel_v1"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "codec",
        "sample_rate",
        "hop_length",
        "latent_dim",
        "frame_count",
        "dataset_fingerprint",
        "mean",
        "std",
    }
)
_CODEC_FIELDS = frozenset({"source", "revision", "sha256"})


class LatentNormalizationError(ValueError):
    """Raised when latent statistics or their provenance are invalid."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LatentNormalizationError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LatentNormalizationError(f"{name} must be a non-negative integer.")
    return value


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise LatentNormalizationError(
            f"{name} must be a non-empty string without surrounding whitespace."
        )
    return value


def _float_tuple(value: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise LatentNormalizationError(f"{name} must be a sequence of finite numbers.")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LatentNormalizationError(f"{name} must be a sequence of finite numbers.") from exc
    if not all(math.isfinite(item) for item in result):
        raise LatentNormalizationError(f"{name} must contain only finite numbers.")
    return result


def _strict_mapping(value: Any, fields: frozenset[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LatentNormalizationError(f"{name} must be a JSON object.")
    result = dict(value)
    if set(result) != fields:
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields)
        raise LatentNormalizationError(
            f"{name} fields are invalid: missing={missing}, unknown={unknown}."
        )
    return result


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LatentNormalizationError(f"Duplicate JSON field {key!r}.")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise LatentNormalizationError(f"Non-finite JSON number {value!r} is not permitted.")


@dataclass(frozen=True, slots=True)
class LatentNormalizationContract:
    """Immutable normalization parameters and the data/codec identity that produced them.

    ``per_channel_v1`` stores population statistics over valid latent frames.
    ``none`` is a first-class mode and therefore cannot be confused with missing
    metadata or silently enabled normalization.
    """

    mode: LatentNormalizationMode
    codec_source: str
    codec_revision: str | None
    codec_sha256: str
    sample_rate: int
    hop_length: int
    latent_dim: int
    frame_count: int
    dataset_fingerprint: str
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    schema_version: int = field(
        default=LATENT_NORMALIZATION_SCHEMA_VERSION,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.mode not in LATENT_NORMALIZATION_MODES:
            raise LatentNormalizationError(
                f"mode must be one of {LATENT_NORMALIZATION_MODES}, got {self.mode!r}."
            )
        _nonempty_string(self.codec_source, name="codec_source")
        if self.codec_revision is not None:
            _nonempty_string(self.codec_revision, name="codec_revision")
        if not isinstance(self.codec_sha256, str) or _SHA256.fullmatch(self.codec_sha256) is None:
            raise LatentNormalizationError("codec_sha256 must be a lowercase SHA-256 digest.")
        _positive_integer(self.sample_rate, name="sample_rate")
        _positive_integer(self.hop_length, name="hop_length")
        _positive_integer(self.latent_dim, name="latent_dim")
        _nonnegative_integer(self.frame_count, name="frame_count")
        _nonempty_string(self.dataset_fingerprint, name="dataset_fingerprint")

        mean = _float_tuple(self.mean, name="mean")
        std = _float_tuple(self.std, name="std")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

        if self.mode == "none":
            if mean or std:
                raise LatentNormalizationError("mode='none' must not contain mean or std values.")
            return

        if self.frame_count <= 0:
            raise LatentNormalizationError("per_channel_v1 requires at least one valid frame.")
        if len(mean) != self.latent_dim or len(std) != self.latent_dim:
            raise LatentNormalizationError(
                "per_channel_v1 mean and std lengths must equal latent_dim."
            )
        if any(value <= 0.0 for value in std):
            raise LatentNormalizationError(
                "per_channel_v1 std values must be strictly positive; constant channels "
                "must be diagnosed rather than silently clamped."
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "codec": {
                "source": self.codec_source,
                "revision": self.codec_revision,
                "sha256": self.codec_sha256,
            },
            "sample_rate": self.sample_rate,
            "hop_length": self.hop_length,
            "latent_dim": self.latent_dim,
            "frame_count": self.frame_count,
            "dataset_fingerprint": self.dataset_fingerprint,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    @property
    def checksum(self) -> str:
        """SHA-256 of the canonical JSON payload, excluding the checksum itself."""
        return _payload_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation including its checksum."""
        result = self._payload()
        result["checksum"] = self.checksum
        return result

    def to_json(self) -> str:
        """Serialize deterministically as compact, canonical UTF-8 JSON text."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LatentNormalizationContract:
        """Parse and verify a serialized contract without accepting unknown fields."""
        if not isinstance(value, Mapping):
            raise LatentNormalizationError("Latent normalization value must be an object.")
        raw = dict(value)
        expected_fields = _PAYLOAD_FIELDS | {"checksum"}
        if set(raw) != expected_fields:
            missing = sorted(expected_fields - set(raw))
            unknown = sorted(set(raw) - expected_fields)
            raise LatentNormalizationError(
                f"Latent normalization fields are invalid: missing={missing}, unknown={unknown}."
            )

        checksum = raw.pop("checksum")
        if not isinstance(checksum, str) or _SHA256.fullmatch(checksum) is None:
            raise LatentNormalizationError("checksum must be a lowercase SHA-256 digest.")
        expected_checksum = _payload_checksum(raw)
        if not hmac.compare_digest(checksum, expected_checksum):
            raise LatentNormalizationError(
                "Latent normalization checksum does not match its canonical payload."
            )

        schema_version = raw["schema_version"]
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != LATENT_NORMALIZATION_SCHEMA_VERSION
        ):
            raise LatentNormalizationError(
                f"Unsupported latent normalization schema version {schema_version!r}; "
                f"expected {LATENT_NORMALIZATION_SCHEMA_VERSION}."
            )
        codec = _strict_mapping(raw["codec"], _CODEC_FIELDS, name="codec")
        return cls(
            mode=raw["mode"],
            codec_source=codec["source"],
            codec_revision=codec["revision"],
            codec_sha256=codec["sha256"],
            sample_rate=raw["sample_rate"],
            hop_length=raw["hop_length"],
            latent_dim=raw["latent_dim"],
            frame_count=raw["frame_count"],
            dataset_fingerprint=raw["dataset_fingerprint"],
            mean=raw["mean"],
            std=raw["std"],
        )

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> LatentNormalizationContract:
        """Parse canonical or ordinary JSON and reject duplicate object fields."""
        try:
            raw = json.loads(
                value,
                object_pairs_hook=_json_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except LatentNormalizationError:
            raise
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            raise LatentNormalizationError("Invalid latent normalization JSON.") from exc
        if not isinstance(raw, Mapping):
            raise LatentNormalizationError("Latent normalization JSON must be an object.")
        return cls.from_dict(raw)

    def _broadcast_statistics(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor.")
        if latents.ndim not in (2, 3):
            raise LatentNormalizationError("latents must have shape [C, T] or [B, C, T].")
        channel_axis = 0 if latents.ndim == 2 else 1
        if latents.shape[channel_axis] != self.latent_dim:
            raise LatentNormalizationError(
                f"Latent channel dimension is {latents.shape[channel_axis]}, expected "
                f"{self.latent_dim}."
            )
        if not latents.is_floating_point():
            raise LatentNormalizationError("latents must use a floating-point dtype.")

        if self.mode == "none":
            empty = latents.new_empty((0,))
            return empty, empty
        shape = (self.latent_dim, 1) if latents.ndim == 2 else (1, self.latent_dim, 1)
        mean = latents.new_tensor(self.mean).view(shape)
        std = latents.new_tensor(self.std).view(shape)
        return mean, std

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize a ``[C,T]`` or ``[B,C,T]`` tensor without changing its dtype/device."""
        mean, std = self._broadcast_statistics(latents)
        if self.mode == "none":
            return latents
        return (latents - mean) / std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`normalize` for a ``[C,T]`` or ``[B,C,T]`` tensor."""
        mean, std = self._broadcast_statistics(latents)
        if self.mode == "none":
            return latents
        return latents * std + mean


class LatentStatisticsAccumulator:
    """Float64 streaming population statistics for valid ``[C,T]`` chunks.

    Each chunk is summarized independently, then combined with the parallel
    Welford formula.  ``merge`` applies the same formula to worker-local
    accumulators, so callers never need to retain the full latent dataset.
    """

    def __init__(self, latent_dim: int) -> None:
        self._latent_dim = _positive_integer(latent_dim, name="latent_dim")
        self._frame_count = 0
        self._mean = torch.zeros(self._latent_dim, dtype=torch.float64)
        self._m2 = torch.zeros(self._latent_dim, dtype=torch.float64)

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _combine(self, count: int, mean: torch.Tensor, m2: torch.Tensor) -> None:
        if count == 0:
            return
        if self._frame_count == 0:
            self._frame_count = count
            self._mean.copy_(mean)
            self._m2.copy_(m2)
            return

        previous_count = self._frame_count
        total_count = previous_count + count
        delta = mean - self._mean
        self._mean.add_(delta * (count / total_count))
        self._m2.add_(m2 + delta.square() * (previous_count * count / total_count))
        self._frame_count = total_count

    def update(
        self,
        latents: torch.Tensor,
        *,
        valid_frames: int | None = None,
    ) -> LatentStatisticsAccumulator:
        """Add the valid prefix of one ``[C,T]`` latent sequence.

        The selected frames are detached and transferred to CPU float64 before
        reduction.  ``valid_frames=0`` is a validated no-op, which is convenient
        for padded or filtered records.
        """
        if not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor.")
        if latents.ndim != 2:
            raise LatentNormalizationError("Streaming updates require latents shaped [C, T].")
        if latents.shape[0] != self._latent_dim:
            raise LatentNormalizationError(
                f"Latent channel dimension is {latents.shape[0]}, expected {self._latent_dim}."
            )
        if not latents.is_floating_point():
            raise LatentNormalizationError("latents must use a floating-point dtype.")

        total_frames = latents.shape[1]
        count = total_frames if valid_frames is None else valid_frames
        if isinstance(count, bool) or not isinstance(count, int):
            raise LatentNormalizationError("valid_frames must be an integer or None.")
        if count < 0 or count > total_frames:
            raise LatentNormalizationError(
                f"valid_frames must be between 0 and {total_frames}, got {count}."
            )
        if count == 0:
            return self

        values = latents[:, :count].detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise LatentNormalizationError("Valid latent frames must contain only finite values.")
        chunk_mean = values.mean(dim=1)
        centered = values - chunk_mean.unsqueeze(1)
        chunk_m2 = centered.square().sum(dim=1)
        self._combine(count, chunk_mean, chunk_m2)
        return self

    def merge(self, other: LatentStatisticsAccumulator) -> LatentStatisticsAccumulator:
        """Merge worker-local statistics using the parallel Welford formula."""
        if not isinstance(other, LatentStatisticsAccumulator):
            raise TypeError("other must be a LatentStatisticsAccumulator.")
        if other.latent_dim != self.latent_dim:
            raise LatentNormalizationError(
                f"Cannot merge latent_dim={other.latent_dim} into latent_dim={self.latent_dim}."
            )
        self._combine(other._frame_count, other._mean, other._m2)
        return self

    def finalize(
        self,
        *,
        mode: LatentNormalizationMode,
        codec_source: str,
        codec_revision: str | None,
        codec_sha256: str,
        sample_rate: int,
        hop_length: int,
        dataset_fingerprint: str,
    ) -> LatentNormalizationContract:
        """Freeze accumulated values into a provenance-bound contract.

        The standard deviation is the population statistic ``sqrt(M2 / N)``.
        The caller must choose ``none`` or ``per_channel_v1`` explicitly.
        """
        if mode == "none":
            return LatentNormalizationContract(
                mode=mode,
                codec_source=codec_source,
                codec_revision=codec_revision,
                codec_sha256=codec_sha256,
                sample_rate=sample_rate,
                hop_length=hop_length,
                latent_dim=self.latent_dim,
                frame_count=self.frame_count,
                dataset_fingerprint=dataset_fingerprint,
            )
        if mode != "per_channel_v1":
            raise LatentNormalizationError(
                f"mode must be one of {LATENT_NORMALIZATION_MODES}, got {mode!r}."
            )
        if self.frame_count == 0:
            raise LatentNormalizationError("Cannot finalize per-channel statistics without frames.")

        variance = (self._m2 / self.frame_count).clamp_min(0.0)
        std = variance.sqrt()
        return LatentNormalizationContract(
            mode=mode,
            codec_source=codec_source,
            codec_revision=codec_revision,
            codec_sha256=codec_sha256,
            sample_rate=sample_rate,
            hop_length=hop_length,
            latent_dim=self.latent_dim,
            frame_count=self.frame_count,
            dataset_fingerprint=dataset_fingerprint,
            mean=tuple(self._mean.tolist()),
            std=tuple(std.tolist()),
        )


__all__ = [
    "LATENT_NORMALIZATION_MODES",
    "LATENT_NORMALIZATION_SCHEMA_VERSION",
    "LatentNormalizationContract",
    "LatentNormalizationError",
    "LatentNormalizationMode",
    "LatentStatisticsAccumulator",
]
