"""Dependency-free request and shape metadata for Echo serving experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from nar_vae.languages import LanguagePair


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class ShapeBucketKey:
    """Exact compatibility key for one step-synchronous model invocation.

    Values are bucket identities rather than raw, unpadded shapes. Two work items
    may share a batch only when the complete key compares equal.
    """

    checkpoint: str
    generation_profile: str
    precision: str
    text_bucket: int
    reference_bucket: int
    latent_bucket: int
    block_bucket: int
    solver: str
    step_index: int
    step_count: int
    cfg_layout: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint",
            "generation_profile",
            "precision",
            "solver",
            "cfg_layout",
        ):
            _require_non_empty(getattr(self, name), name)
        for name in (
            "text_bucket",
            "reference_bucket",
            "latent_bucket",
            "block_bucket",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.step_count <= 0:
            raise ValueError("step_count must be positive.")
        if not 0 <= self.step_index < self.step_count:
            raise ValueError("step_index must satisfy 0 <= step_index < step_count.")

    def sort_key(self) -> tuple[str | int, ...]:
        """Return a stable ordering key for deterministic scheduling and reports."""
        return (
            self.checkpoint,
            self.generation_profile,
            self.precision,
            self.text_bucket,
            self.reference_bucket,
            self.latent_bucket,
            self.block_bucket,
            self.solver,
            self.step_index,
            self.step_count,
            self.cfg_layout,
        )

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-compatible representation with stable field names."""
        return {
            "checkpoint": self.checkpoint,
            "generation_profile": self.generation_profile,
            "precision": self.precision,
            "text_bucket": self.text_bucket,
            "reference_bucket": self.reference_bucket,
            "latent_bucket": self.latent_bucket,
            "block_bucket": self.block_bucket,
            "solver": self.solver,
            "step_index": self.step_index,
            "step_count": self.step_count,
            "cfg_layout": self.cfg_layout,
        }

    @property
    def stable_id(self) -> str:
        """Return the canonical JSON identity used in batch distributions."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    """Immutable serving metadata with independent target/reference languages."""

    request_id: str
    client_id: str
    arrival_time_s: float
    first_audio_deadline_s: float
    bucket_key: ShapeBucketKey
    target_language: str = "en"
    reference_language: str | None = None
    total_blocks: int = 1
    continuation_interval_s: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.request_id, "request_id")
        _require_non_empty(self.client_id, "client_id")
        _require_finite_non_negative(self.arrival_time_s, "arrival_time_s")
        _require_finite_non_negative(self.first_audio_deadline_s, "first_audio_deadline_s")
        if self.first_audio_deadline_s <= self.arrival_time_s:
            raise ValueError("first_audio_deadline_s must be later than arrival_time_s.")
        if self.total_blocks <= 0:
            raise ValueError("total_blocks must be positive.")
        if self.total_blocks > 1:
            if self.continuation_interval_s is None or self.continuation_interval_s <= 0:
                raise ValueError(
                    "continuation_interval_s must be positive when total_blocks is greater than one."
                )
        elif self.continuation_interval_s is not None and self.continuation_interval_s <= 0:
            raise ValueError("continuation_interval_s must be positive when supplied.")

        pair = LanguagePair.resolve(
            self.target_language,
            self.reference_language,
            has_reference=self.reference_language is not None,
        )
        object.__setattr__(self, "target_language", pair.target)
        object.__setattr__(self, "reference_language", pair.reference)

    @property
    def language_pair(self) -> LanguagePair:
        """Return source and target roles without inferring one from the other."""
        return LanguagePair(
            target=self.target_language,
            reference=self.reference_language,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible request metadata."""
        pair = self.language_pair
        return {
            "request_id": self.request_id,
            "client_id": self.client_id,
            "arrival_time_s": self.arrival_time_s,
            "first_audio_deadline_s": self.first_audio_deadline_s,
            "target_language": pair.target,
            "reference_language": pair.reference,
            "language_pair": {
                "target": pair.target,
                "reference": pair.reference,
                "cross_lingual": pair.is_cross_lingual,
            },
            "total_blocks": self.total_blocks,
            "continuation_interval_s": self.continuation_interval_s,
            "bucket_key": self.bucket_key.to_dict(),
        }


__all__ = ["RequestMetadata", "ShapeBucketKey"]
