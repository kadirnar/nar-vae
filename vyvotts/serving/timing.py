"""Stage timing and JSON-safe descriptive statistics for serving measurements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

STAGE_NAMES = (
    "queue",
    "conditioning",
    "generation",
    "decode",
    "transfer",
    "packetization",
    "ttfa",
)


def percentile(values: Iterable[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for finite values."""
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must satisfy 0 <= fraction <= 1.")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile for an empty series.")
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Percentile values must be finite.")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_percentiles(values: Iterable[float]) -> dict[str, float | int | None]:
    """Return p50/p95/p99 and basic statistics, including an empty-series record."""
    selected = [float(value) for value in values]
    if not selected:
        return {
            "count": 0,
            "mean_s": None,
            "p50_s": None,
            "p95_s": None,
            "p99_s": None,
            "min_s": None,
            "max_s": None,
        }
    return {
        "count": len(selected),
        "mean_s": statistics.fmean(selected),
        "p50_s": percentile(selected, 0.5),
        "p95_s": percentile(selected, 0.95),
        "p99_s": percentile(selected, 0.99),
        "min_s": min(selected),
        "max_s": max(selected),
    }


@dataclass(frozen=True, slots=True)
class StageTiming:
    """One request's queue-to-first-audio stage durations in seconds."""

    queue_s: float
    conditioning_s: float
    generation_s: float
    decode_s: float
    transfer_s: float
    packetization_s: float
    ttfa_s: float
    result_kind: str

    def __post_init__(self) -> None:
        for name in (
            "queue_s",
            "conditioning_s",
            "generation_s",
            "decode_s",
            "transfer_s",
            "packetization_s",
            "ttfa_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not self.result_kind.strip():
            raise ValueError("result_kind must be a non-empty string.")

    @classmethod
    def from_complete_waveform_timings(cls, timings: Mapping[str, float]) -> "StageTiming":
        """Adapt existing benchmark names without changing their values or semantics."""
        return cls(
            queue_s=float(timings.get("queue", 0.0)),
            conditioning_s=float(timings["conditioning"]),
            generation_s=float(timings["ode_sampling"]),
            decode_s=float(timings["decoding"]),
            transfer_s=float(timings["output_transfer"]),
            packetization_s=0.0,
            ttfa_s=float(timings["ttfa"]),
            result_kind="complete_waveform",
        )

    def to_dict(self) -> dict[str, float | str]:
        """Return stable JSON-compatible stage names."""
        return {
            "queue_s": self.queue_s,
            "conditioning_s": self.conditioning_s,
            "generation_s": self.generation_s,
            "decode_s": self.decode_s,
            "transfer_s": self.transfer_s,
            "packetization_s": self.packetization_s,
            "ttfa_s": self.ttfa_s,
            "result_kind": self.result_kind,
        }


def summarize_stage_timings(timings: Iterable[StageTiming]) -> dict[str, object]:
    """Aggregate every timing stage with the same percentile definition."""
    selected = list(timings)
    return {
        "queue": summarize_percentiles(timing.queue_s for timing in selected),
        "conditioning": summarize_percentiles(timing.conditioning_s for timing in selected),
        "generation": summarize_percentiles(timing.generation_s for timing in selected),
        "decode": summarize_percentiles(timing.decode_s for timing in selected),
        "transfer": summarize_percentiles(timing.transfer_s for timing in selected),
        "packetization": summarize_percentiles(timing.packetization_s for timing in selected),
        "ttfa": summarize_percentiles(timing.ttfa_s for timing in selected),
    }


def non_claim_evidence(
    *,
    result_kind: str,
    synthetic: bool,
    hardware_measured: bool | None = None,
) -> dict[str, object]:
    """Mark a result that cannot support named-GPU streaming or latency claims."""
    return {
        "result_kind": result_kind,
        "synthetic": synthetic,
        "model_streaming": False,
        "independently_playable_audio": False,
        "hardware_measured": False if synthetic else hardware_measured,
        "named_gpu_streaming_evidence": False,
        "claim_eligible": False,
    }


__all__ = [
    "STAGE_NAMES",
    "StageTiming",
    "non_claim_evidence",
    "percentile",
    "summarize_percentiles",
    "summarize_stage_timings",
]
