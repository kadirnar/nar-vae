"""Deterministic synthetic burst and steady-stream load simulation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

from nar_vae.languages import LanguagePair
from nar_vae.serving.metadata import RequestMetadata, ShapeBucketKey
from nar_vae.serving.scheduler import (
    DeadlineBatchScheduler,
    RequestStatus,
    SchedulerConfig,
    WorkKind,
)
from nar_vae.serving.timing import StageTiming, non_claim_evidence, summarize_stage_timings

DEFAULT_CLIENT_COUNTS = (1, 8, 16, 32, 50)


class ArrivalPattern(str, Enum):
    """Synthetic arrival modes required by the load contract."""

    SYNCHRONIZED_BURST = "synchronized_burst"
    STEADY_STREAM = "steady_stream"


class ManualClock:
    """Monotonic injected clock used without wall-clock sleeps."""

    def __init__(self, initial_time_s: float = 0.0) -> None:
        if not math.isfinite(initial_time_s) or initial_time_s < 0:
            raise ValueError("initial_time_s must be finite and non-negative.")
        self._time_s = float(initial_time_s)

    def __call__(self) -> float:
        return self._time_s

    def advance(self, duration_s: float) -> float:
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and non-negative.")
        self._time_s += duration_s
        return self._time_s

    def advance_to(self, time_s: float) -> float:
        if not math.isfinite(time_s) or time_s < self._time_s:
            raise ValueError("ManualClock cannot move backwards or to a non-finite value.")
        self._time_s = float(time_s)
        return self._time_s


@dataclass(frozen=True, slots=True)
class SyntheticServiceTimes:
    """Fixed synthetic batch durations; these are never GPU measurements."""

    conditioning_s: float = 0.004
    first_generation_s: float = 0.010
    continuation_generation_s: float = 0.006
    decode_s: float = 0.003
    transfer_s: float = 0.001
    packetization_s: float = 0.001

    def __post_init__(self) -> None:
        for name in (
            "conditioning_s",
            "first_generation_s",
            "continuation_generation_s",
            "decode_s",
            "transfer_s",
            "packetization_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")

    def batch_duration_s(self, kind: WorkKind) -> float:
        generation = (
            self.first_generation_s
            if kind == WorkKind.FIRST_BLOCK
            else self.continuation_generation_s
        )
        conditioning = self.conditioning_s if kind == WorkKind.FIRST_BLOCK else 0.0
        return conditioning + generation + self.decode_s + self.transfer_s + self.packetization_s

    def to_dict(self) -> dict[str, float]:
        return {
            "conditioning_s": self.conditioning_s,
            "first_generation_s": self.first_generation_s,
            "continuation_generation_s": self.continuation_generation_s,
            "decode_s": self.decode_s,
            "transfer_s": self.transfer_s,
            "packetization_s": self.packetization_s,
        }


def _default_bucket_key() -> ShapeBucketKey:
    return ShapeBucketKey(
        checkpoint="synthetic-untrained-contract",
        generation_profile="synthetic",
        precision="synthetic",
        text_bucket=64,
        reference_bucket=0,
        latent_bucket=256,
        block_bucket=32,
        solver="synthetic",
        step_index=0,
        step_count=1,
        cfg_layout="synthetic-single",
    )


@dataclass(frozen=True, slots=True)
class SyntheticWorkload:
    """One deterministic load scenario with explicit language roles and provenance."""

    pattern: ArrivalPattern
    clients: int
    blocks_per_request: int
    first_audio_budget_s: float = 0.1
    continuation_interval_s: float = 0.04
    arrival_interval_s: float = 0.002
    target_language: str = "en"
    reference_language: str | None = None
    bucket_key: ShapeBucketKey = field(default_factory=_default_bucket_key)
    seed: int = 0
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.clients <= 0:
            raise ValueError("clients must be positive.")
        if self.blocks_per_request <= 0:
            raise ValueError("blocks_per_request must be positive.")
        if not math.isfinite(self.first_audio_budget_s) or self.first_audio_budget_s <= 0:
            raise ValueError("first_audio_budget_s must be finite and positive.")
        if not math.isfinite(self.continuation_interval_s) or self.continuation_interval_s <= 0:
            raise ValueError("continuation_interval_s must be finite and positive.")
        if not math.isfinite(self.arrival_interval_s) or self.arrival_interval_s < 0:
            raise ValueError("arrival_interval_s must be finite and non-negative.")
        json.dumps(dict(self.provenance), allow_nan=False)
        pair = LanguagePair.resolve(
            self.target_language,
            self.reference_language,
            has_reference=self.reference_language is not None,
        )
        object.__setattr__(self, "target_language", pair.target)
        object.__setattr__(self, "reference_language", pair.reference)
        if pair.reference is not None and self.bucket_key == _default_bucket_key():
            object.__setattr__(
                self,
                "bucket_key",
                replace(self.bucket_key, reference_bucket=128),
            )

    def arrival_offsets(self) -> tuple[float, ...]:
        if self.pattern == ArrivalPattern.SYNCHRONIZED_BURST:
            return (0.0,) * self.clients
        return tuple(index * self.arrival_interval_s for index in range(self.clients))

    def to_dict(self) -> dict[str, object]:
        return {
            "pattern": self.pattern.value,
            "clients": self.clients,
            "blocks_per_request": self.blocks_per_request,
            "first_audio_budget_s": self.first_audio_budget_s,
            "continuation_interval_s": self.continuation_interval_s,
            "arrival_interval_s": self.arrival_interval_s,
            "target_language": self.target_language,
            "reference_language": self.reference_language,
            "language_pair": {
                "target": self.target_language,
                "reference": self.reference_language,
                "cross_lingual": (
                    self.reference_language is not None
                    and self.reference_language != self.target_language
                ),
            },
            "bucket_key": self.bucket_key.to_dict(),
            "seed": self.seed,
            "provenance": dict(self.provenance),
        }


class SyntheticLoadHarness:
    """Run event-driven scheduling simulations against an injected manual clock."""

    def __init__(
        self,
        *,
        clock: ManualClock,
        scheduler_config: SchedulerConfig | None = None,
        service_times: SyntheticServiceTimes | None = None,
    ) -> None:
        self.clock = clock
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.service_times = service_times or SyntheticServiceTimes()

    def _requests(self, workload: SyntheticWorkload) -> list[RequestMetadata]:
        start = self.clock()
        requests = []
        for index, offset in enumerate(workload.arrival_offsets()):
            arrival = start + offset
            requests.append(
                RequestMetadata(
                    request_id=f"{workload.pattern.value}-{index:03d}",
                    client_id=f"client-{index:03d}",
                    arrival_time_s=arrival,
                    first_audio_deadline_s=arrival + workload.first_audio_budget_s,
                    bucket_key=workload.bucket_key,
                    target_language=workload.target_language,
                    reference_language=workload.reference_language,
                    total_blocks=workload.blocks_per_request,
                    continuation_interval_s=(
                        workload.continuation_interval_s
                        if workload.blocks_per_request > 1
                        else None
                    ),
                )
            )
        return requests

    def run(self, workload: SyntheticWorkload) -> dict[str, object]:
        """Return one JSON-compatible synthetic report without sleeping or model work."""
        started_at_s = self.clock()
        scheduler = DeadlineBatchScheduler(self.scheduler_config, clock=self.clock)
        requests = self._requests(workload)
        pending = list(requests)
        admitted = 0
        first_timings: dict[str, StageTiming] = {}
        batch_history: list[dict[str, object]] = []
        batch_sizes: list[int] = []

        while pending or scheduler.pending_count:
            now = self.clock()
            while pending and pending[0].arrival_time_s <= now:
                decision = scheduler.admit(pending.pop(0))
                admitted += int(decision.admitted)

            batch = scheduler.next_batch()
            if batch is not None:
                selected_at_s = self.clock()
                duration_s = self.service_times.batch_duration_s(batch.kind)
                self.clock.advance(duration_s)
                completed_at_s = self.clock()
                if batch.kind == WorkKind.FIRST_BLOCK:
                    for item in batch.items:
                        metadata = scheduler.state(item.request_id).metadata
                        first_timings[item.request_id] = StageTiming(
                            queue_s=selected_at_s - metadata.arrival_time_s,
                            conditioning_s=self.service_times.conditioning_s,
                            generation_s=self.service_times.first_generation_s,
                            decode_s=self.service_times.decode_s,
                            transfer_s=self.service_times.transfer_s,
                            packetization_s=self.service_times.packetization_s,
                            ttfa_s=completed_at_s - metadata.arrival_time_s,
                            result_kind="synthetic_schedule_simulation",
                        )
                scheduler.complete_batch(batch)
                batch_sizes.append(len(batch.items))
                batch_row = batch.to_dict()
                batch_row["completed_at_s"] = completed_at_s
                batch_row["synthetic_duration_s"] = duration_s
                batch_history.append(batch_row)
                continue

            event_times = []
            if pending:
                event_times.append(pending[0].arrival_time_s)
            ready_at = scheduler.next_ready_time()
            if ready_at is not None:
                event_times.append(ready_at)
            deadline = scheduler.next_deadline()
            if deadline is not None:
                event_times.append(deadline)
            future = [event_time for event_time in event_times if event_time > now]
            if not future:
                scheduler.expire()
                if not pending and not scheduler.pending_count:
                    break
                raise RuntimeError("Synthetic scheduler made no event-loop progress.")
            self.clock.advance_to(min(future))

        finished_at_s = self.clock()
        states = scheduler.states
        status_counts = Counter(state.status.value for state in states)
        rejection_reasons = Counter(
            state.reason
            for state in states
            if state.status == RequestStatus.REJECTED and state.reason is not None
        )
        failure_reasons = Counter(
            state.reason
            for state in states
            if state.status in (RequestStatus.TIMED_OUT, RequestStatus.FAILED)
            and state.reason is not None
        )
        successful_timings = [
            first_timings[state.metadata.request_id]
            for state in states
            if state.status == RequestStatus.COMPLETED
            and state.metadata.request_id in first_timings
        ]
        elapsed_s = finished_at_s - started_at_s
        completed = status_counts[RequestStatus.COMPLETED.value]
        size_distribution = Counter(batch_sizes)

        request_results = []
        for state in states:
            row = state.to_dict()
            timing = first_timings.get(state.metadata.request_id)
            row["first_audio_timing"] = timing.to_dict() if timing is not None else None
            request_results.append(row)

        report = {
            "schema_version": 1,
            "evidence": non_claim_evidence(
                result_kind="synthetic_schedule_simulation",
                synthetic=True,
            ),
            "workload": workload.to_dict(),
            "provenance": {
                "clock": "injected_manual_clock",
                "network_used": False,
                "gpu_used": False,
                "checkpoint_loaded": False,
                "audio_generated": False,
                "service_times": self.service_times.to_dict(),
                "scheduler": {
                    "max_batch_size": self.scheduler_config.max_batch_size,
                    "max_active_requests": self.scheduler_config.max_active_requests,
                    "max_queue_delay_s": self.scheduler_config.max_queue_delay_s,
                    "max_first_block_batches": self.scheduler_config.max_first_block_batches,
                },
            },
            "counts": {
                "submitted": len(requests),
                "admitted": admitted,
                "completed": completed,
                "rejected": status_counts[RequestStatus.REJECTED.value],
                "timed_out": status_counts[RequestStatus.TIMED_OUT.value],
                "failed": status_counts[RequestStatus.FAILED.value],
            },
            "failures": {
                "total": (
                    status_counts[RequestStatus.TIMED_OUT.value]
                    + status_counts[RequestStatus.FAILED.value]
                ),
                "reasons": dict(sorted(failure_reasons.items())),
            },
            "rejections": {
                "total": status_counts[RequestStatus.REJECTED.value],
                "reasons": dict(sorted(rejection_reasons.items())),
            },
            "latency": summarize_stage_timings(successful_timings),
            "throughput": {
                "elapsed_s": elapsed_s,
                "completed_requests_per_s": completed / elapsed_s if elapsed_s else None,
                "completed_blocks": sum(state.completed_blocks for state in states),
            },
            "batches": {
                "count": len(batch_sizes),
                "size_distribution": {
                    str(size): count for size, count in sorted(size_distribution.items())
                },
                "maximum_size": max(batch_sizes, default=0),
                "history": batch_history,
            },
            "resources": {
                "gpu_peak_memory_bytes": None,
                "gpu_utilization": None,
            },
            "request_results": request_results,
        }
        json.dumps(report, allow_nan=False)
        return report


def run_synthetic_load_suite(
    *,
    client_counts: Sequence[int] = DEFAULT_CLIENT_COUNTS,
    scheduler_config: SchedulerConfig | None = None,
    service_times: SyntheticServiceTimes | None = None,
    clock_factory: Callable[[], ManualClock] = ManualClock,
    target_language: str = "en",
    reference_language: str | None = None,
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run synchronized-burst and steady-stream matrices at standard client counts."""
    counts = tuple(int(value) for value in client_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("client_counts must contain positive integers.")
    scenarios: dict[str, dict[str, object]] = {
        ArrivalPattern.SYNCHRONIZED_BURST.value: {},
        ArrivalPattern.STEADY_STREAM.value: {},
    }
    for pattern in ArrivalPattern:
        for clients in counts:
            workload = SyntheticWorkload(
                pattern=pattern,
                clients=clients,
                blocks_per_request=(1 if pattern == ArrivalPattern.SYNCHRONIZED_BURST else 4),
                target_language=target_language,
                reference_language=reference_language,
                provenance=provenance or {},
            )
            report = SyntheticLoadHarness(
                clock=clock_factory(),
                scheduler_config=scheduler_config,
                service_times=service_times,
            ).run(workload)
            scenarios[pattern.value][str(clients)] = report

    result = {
        "schema_version": 1,
        "evidence": non_claim_evidence(
            result_kind="synthetic_schedule_suite",
            synthetic=True,
        ),
        "client_counts": list(counts),
        "scenarios": scenarios,
    }
    json.dumps(result, allow_nan=False)
    return result


def write_json_result(result: Mapping[str, object], output: str | Path) -> Path:
    """Write a validated JSON-compatible serving result and return its path."""
    output_path = Path(output)
    serialized = json.dumps(dict(result), indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_CLIENT_COUNTS",
    "ArrivalPattern",
    "ManualClock",
    "SyntheticLoadHarness",
    "SyntheticServiceTimes",
    "SyntheticWorkload",
    "run_synthetic_load_suite",
    "write_json_result",
]
