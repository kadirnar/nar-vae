"""Deterministic admission, deadline, and batch scheduling contracts."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from vyvotts.serving.metadata import RequestMetadata, ShapeBucketKey


class RequestStatus(str, Enum):
    """Lifecycle states recorded by the dependency-free scheduler."""

    ADMITTED = "admitted"
    COMPLETED = "completed"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class WorkKind(str, Enum):
    """Priority classes for first-audio and continuation work."""

    FIRST_BLOCK = "first_block"
    CONTINUATION = "continuation"


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Bounded scheduler limits with deterministic continuation fairness."""

    max_batch_size: int = 16
    max_active_requests: int = 50
    max_queue_delay_s: float = 0.1
    max_first_block_batches: int = 4

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive.")
        if self.max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive.")
        if not math.isfinite(self.max_queue_delay_s) or self.max_queue_delay_s <= 0:
            raise ValueError("max_queue_delay_s must be finite and positive.")
        if self.max_first_block_batches <= 0:
            raise ValueError("max_first_block_batches must be positive.")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Serializable result of one admission attempt."""

    request_id: str
    admitted: bool
    status: RequestStatus
    decided_at_s: float
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "admitted": self.admitted,
            "status": self.status.value,
            "decided_at_s": self.decided_at_s,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScheduledWork:
    """One immutable unit selected for an exact-compatible batch."""

    sequence: int
    request_id: str
    kind: WorkKind
    block_index: int
    bucket_key: ShapeBucketKey
    ready_at_s: float
    deadline_s: float
    queue_deadline_s: float

    def priority_key(self) -> tuple[float, float, int, str]:
        return (self.deadline_s, self.ready_at_s, self.sequence, self.request_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "request_id": self.request_id,
            "kind": self.kind.value,
            "block_index": self.block_index,
            "ready_at_s": self.ready_at_s,
            "deadline_s": self.deadline_s,
            "queue_deadline_s": self.queue_deadline_s,
            "bucket_key": self.bucket_key.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ScheduledBatch:
    """One deterministic batch containing only an exact shape key and work class."""

    selected_at_s: float
    kind: WorkKind
    bucket_key: ShapeBucketKey
    items: tuple[ScheduledWork, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("A scheduled batch must contain at least one work item.")
        if any(item.kind != self.kind for item in self.items):
            raise ValueError("A scheduled batch cannot mix first and continuation work.")
        if any(item.bucket_key != self.bucket_key for item in self.items):
            raise ValueError("A scheduled batch cannot mix shape bucket keys.")

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_at_s": self.selected_at_s,
            "kind": self.kind.value,
            "size": len(self.items),
            "bucket_key": self.bucket_key.to_dict(),
            "request_ids": [item.request_id for item in self.items],
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(slots=True)
class RequestState:
    """Mutable admission and progress state owned by a scheduler instance."""

    metadata: RequestMetadata
    status: RequestStatus
    admitted_at_s: float | None = None
    completed_blocks: int = 0
    first_audio_at_s: float | None = None
    finished_at_s: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        pair = self.metadata.language_pair
        return {
            "request": self.metadata.to_dict(),
            "request_id": self.metadata.request_id,
            "client_id": self.metadata.client_id,
            "target_language": pair.target,
            "reference_language": pair.reference,
            "language_pair": {
                "target": pair.target,
                "reference": pair.reference,
                "cross_lingual": pair.is_cross_lingual,
            },
            "status": self.status.value,
            "admitted_at_s": self.admitted_at_s,
            "completed_blocks": self.completed_blocks,
            "first_audio_at_s": self.first_audio_at_s,
            "finished_at_s": self.finished_at_s,
            "reason": self.reason,
        }


class DeadlineBatchScheduler:
    """Clock-injected scheduler for exact-key batches and bounded queue delay.

    First blocks are preferred while both work classes are ready. After
    ``max_first_block_batches`` consecutive first-block batches, one continuation
    batch is selected so admitted streams make deterministic progress.
    """

    def __init__(
        self,
        config: SchedulerConfig | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or SchedulerConfig()
        self._clock = clock
        self._states: dict[str, RequestState] = {}
        self._queue: list[ScheduledWork] = []
        self._next_sequence = 0
        self._first_block_streak = 0

    def _now(self) -> float:
        value = float(self._clock())
        if not math.isfinite(value) or value < 0:
            raise ValueError("The scheduler clock must return a finite non-negative value.")
        return value

    @property
    def states(self) -> tuple[RequestState, ...]:
        """Return states in request insertion order for deterministic serialization."""
        return tuple(self._states.values())

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return sum(state.status == RequestStatus.ADMITTED for state in self._states.values())

    def state(self, request_id: str) -> RequestState:
        try:
            return self._states[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request_id {request_id!r}.") from exc

    def _append_work(
        self,
        metadata: RequestMetadata,
        *,
        kind: WorkKind,
        block_index: int,
        ready_at_s: float,
        deadline_s: float,
        queue_deadline_s: float | None = None,
    ) -> None:
        self._queue.append(
            ScheduledWork(
                sequence=self._next_sequence,
                request_id=metadata.request_id,
                kind=kind,
                block_index=block_index,
                bucket_key=metadata.bucket_key,
                ready_at_s=ready_at_s,
                deadline_s=deadline_s,
                queue_deadline_s=(deadline_s if queue_deadline_s is None else queue_deadline_s),
            )
        )
        self._next_sequence += 1

    def admit(self, metadata: RequestMetadata) -> AdmissionDecision:
        """Admit a request or reject it before work enters the batch queue."""
        now = self._now()
        if metadata.request_id in self._states:
            raise ValueError(f"Duplicate request_id {metadata.request_id!r}.")
        if metadata.arrival_time_s > now:
            raise ValueError("A request cannot be admitted before its arrival_time_s.")

        bounded_deadline = min(
            metadata.first_audio_deadline_s,
            metadata.arrival_time_s + self.config.max_queue_delay_s,
        )
        if bounded_deadline <= now:
            reason = (
                "first_audio_deadline_expired"
                if metadata.first_audio_deadline_s <= now
                else "queue_delay_limit_expired"
            )
            state = RequestState(
                metadata=metadata,
                status=RequestStatus.REJECTED,
                finished_at_s=now,
                reason=reason,
            )
            self._states[metadata.request_id] = state
            return AdmissionDecision(
                request_id=metadata.request_id,
                admitted=False,
                status=state.status,
                decided_at_s=now,
                reason=state.reason,
            )
        if self.active_count >= self.config.max_active_requests:
            state = RequestState(
                metadata=metadata,
                status=RequestStatus.REJECTED,
                finished_at_s=now,
                reason="capacity_exhausted",
            )
            self._states[metadata.request_id] = state
            return AdmissionDecision(
                request_id=metadata.request_id,
                admitted=False,
                status=state.status,
                decided_at_s=now,
                reason=state.reason,
            )

        state = RequestState(
            metadata=metadata,
            status=RequestStatus.ADMITTED,
            admitted_at_s=now,
        )
        self._states[metadata.request_id] = state
        self._append_work(
            metadata,
            kind=WorkKind.FIRST_BLOCK,
            block_index=0,
            ready_at_s=now,
            deadline_s=metadata.first_audio_deadline_s,
            queue_deadline_s=bounded_deadline,
        )
        return AdmissionDecision(
            request_id=metadata.request_id,
            admitted=True,
            status=state.status,
            decided_at_s=now,
        )

    def _timeout_request(self, request_id: str, *, now: float, reason: str) -> RequestState:
        state = self.state(request_id)
        if state.status != RequestStatus.ADMITTED:
            return state
        state.status = RequestStatus.TIMED_OUT
        state.finished_at_s = now
        state.reason = reason
        self._queue = [item for item in self._queue if item.request_id != request_id]
        return state

    def expire(self) -> tuple[RequestState, ...]:
        """Expire queued work whose absolute deadline has been reached."""
        now = self._now()
        expired: list[RequestState] = []
        for item in sorted(self._queue, key=ScheduledWork.priority_key):
            if item.queue_deadline_s > now:
                continue
            if item.kind == WorkKind.FIRST_BLOCK:
                reason = (
                    "first_audio_deadline_expired"
                    if item.deadline_s <= now
                    else "queue_delay_limit_expired"
                )
            else:
                reason = "continuation_deadline_expired"
            state = self._timeout_request(item.request_id, now=now, reason=reason)
            if state not in expired:
                expired.append(state)
        return tuple(expired)

    def next_batch(self, *, limit: int | None = None) -> ScheduledBatch | None:
        """Select one exact-key batch using deadline order and stable tie-breaking."""
        now = self._now()
        self.expire()
        ready = [item for item in self._queue if item.ready_at_s <= now]
        if not ready:
            return None

        first = [item for item in ready if item.kind == WorkKind.FIRST_BLOCK]
        continuation = [item for item in ready if item.kind == WorkKind.CONTINUATION]
        if first and continuation:
            kind = (
                WorkKind.CONTINUATION
                if self._first_block_streak >= self.config.max_first_block_batches
                else WorkKind.FIRST_BLOCK
            )
        elif first:
            kind = WorkKind.FIRST_BLOCK
        else:
            kind = WorkKind.CONTINUATION

        candidates = [item for item in ready if item.kind == kind]
        leader = min(candidates, key=ScheduledWork.priority_key)
        compatible = sorted(
            (item for item in candidates if item.bucket_key == leader.bucket_key),
            key=ScheduledWork.priority_key,
        )
        batch_limit = self.config.max_batch_size if limit is None else limit
        if batch_limit <= 0:
            raise ValueError("limit must be positive when supplied.")
        selected = tuple(compatible[: min(batch_limit, self.config.max_batch_size)])
        selected_sequences = {item.sequence for item in selected}
        self._queue = [item for item in self._queue if item.sequence not in selected_sequences]

        if kind == WorkKind.FIRST_BLOCK:
            self._first_block_streak += 1
        else:
            self._first_block_streak = 0
        return ScheduledBatch(
            selected_at_s=now,
            kind=kind,
            bucket_key=leader.bucket_key,
            items=selected,
        )

    def complete_batch(self, batch: ScheduledBatch) -> tuple[RequestState, ...]:
        """Record successful work and enqueue the next continuation when required."""
        now = self._now()
        completed: list[RequestState] = []
        for item in batch.items:
            state = self.state(item.request_id)
            if state.status != RequestStatus.ADMITTED:
                continue
            if item.deadline_s < now:
                reason = (
                    "first_audio_deadline_missed"
                    if item.kind == WorkKind.FIRST_BLOCK
                    else "continuation_deadline_missed"
                )
                completed.append(self._timeout_request(item.request_id, now=now, reason=reason))
                continue

            state.completed_blocks += 1
            if item.kind == WorkKind.FIRST_BLOCK:
                state.first_audio_at_s = now
            if state.completed_blocks >= state.metadata.total_blocks:
                state.status = RequestStatus.COMPLETED
                state.finished_at_s = now
                completed.append(state)
                continue

            assert state.first_audio_at_s is not None
            assert state.metadata.continuation_interval_s is not None
            next_block = state.completed_blocks
            continuation_deadline = (
                state.first_audio_at_s + next_block * state.metadata.continuation_interval_s
            )
            self._append_work(
                state.metadata,
                kind=WorkKind.CONTINUATION,
                block_index=next_block,
                ready_at_s=now,
                deadline_s=continuation_deadline,
            )
        return tuple(completed)

    def fail_request(self, request_id: str, reason: str) -> RequestState:
        """Record an execution failure and remove all queued work for the request."""
        if not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        now = self._now()
        state = self.state(request_id)
        if state.status == RequestStatus.ADMITTED:
            state.status = RequestStatus.FAILED
            state.finished_at_s = now
            state.reason = reason
            self._queue = [item for item in self._queue if item.request_id != request_id]
        return state

    def next_ready_time(self) -> float | None:
        """Return the next queued readiness time for an event-driven harness."""
        if not self._queue:
            return None
        return min(item.ready_at_s for item in self._queue)

    def next_deadline(self) -> float | None:
        """Return the next queued expiration time for an event-driven harness."""
        if not self._queue:
            return None
        return min(item.queue_deadline_s for item in self._queue)


__all__ = [
    "AdmissionDecision",
    "DeadlineBatchScheduler",
    "RequestState",
    "RequestStatus",
    "ScheduledBatch",
    "ScheduledWork",
    "SchedulerConfig",
    "WorkKind",
]
