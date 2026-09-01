"""Shared torchrun rank and device setup for training and dataset preparation."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterator, TypeVar

import torch
import torch.distributed as dist

_T = TypeVar("_T")


def _environment_integer(
    environment: Mapping[str, str],
    name: str,
    default: int | None = None,
) -> int:
    raw = environment.get(name)
    if raw is None:
        if default is None:
            raise ValueError(f"{name} is required for a distributed launch.")
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; received {raw!r}.") from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative; received {value}.")
    return value


@dataclass(frozen=True)
class DistributedContext:
    """Environment-derived identity for one torchrun process."""

    local_rank: int
    rank: int
    world_size: int
    local_world_size: int

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "DistributedContext":
        """Parse torchrun variables without initializing CUDA or a process group."""
        values = os.environ if environment is None else environment
        world_size = _environment_integer(values, "WORLD_SIZE", 1)
        if world_size == 0:
            raise ValueError("WORLD_SIZE must be positive.")
        distributed = world_size > 1
        local_rank = _environment_integer(
            values,
            "LOCAL_RANK",
            None if distributed else 0,
        )
        rank = _environment_integer(values, "RANK", None if distributed else 0)
        local_world_size = _environment_integer(values, "LOCAL_WORLD_SIZE", world_size)
        if local_world_size == 0:
            raise ValueError("LOCAL_WORLD_SIZE must be positive.")
        if rank >= world_size:
            raise ValueError(f"RANK {rank} must be smaller than WORLD_SIZE {world_size}.")
        if local_rank >= local_world_size:
            raise ValueError(
                f"LOCAL_RANK {local_rank} must be smaller than LOCAL_WORLD_SIZE {local_world_size}."
            )
        return cls(
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            local_world_size=local_world_size,
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def trainer_local_rank(self) -> int:
        """Return the rank convention expected by Transformers TrainingArguments."""
        return self.local_rank if self.is_distributed else -1

    def device(self) -> torch.device:
        """Return this process's device without mutating global CUDA state."""
        if self.is_distributed:
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_distributed(
    context: DistributedContext | None = None,
    *,
    timeout_seconds: int = 7200,
) -> DistributedContext:
    """Select the local GPU and initialize NCCL once for a torchrun process."""
    context = context or DistributedContext.from_environment()
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if not context.is_distributed:
        return context
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training and data preparation require CUDA.")
    device_count = torch.cuda.device_count()
    if context.local_rank >= device_count:
        raise RuntimeError(
            f"LOCAL_RANK {context.local_rank} cannot select one of {device_count} CUDA devices."
        )

    torch.cuda.set_device(context.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds),
        )
    actual_rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    if actual_rank != context.rank or actual_world_size != context.world_size:
        raise RuntimeError(
            "Initialized process-group identity does not match torchrun environment: "
            f"rank/world={actual_rank}/{actual_world_size}, "
            f"environment={context.rank}/{context.world_size}."
        )
    return context


def cleanup_distributed(*, barrier: bool = True) -> None:
    """Synchronize an initialized group when requested, then release it."""
    if not dist.is_initialized():
        return
    try:
        if barrier:
            dist.barrier()
    finally:
        dist.destroy_process_group()


@contextmanager
def distributed_cleanup_guard() -> Iterator[None]:
    """Release a process group created inside the guarded training call.

    A rank-local exception must not enter a collective barrier while its peers
    are still elsewhere in startup or training. Successful calls retain the
    orderly barrier, while failures destroy the newly created group directly.
    Process groups owned by an embedding application are left untouched.
    """
    group_was_initialized = dist.is_initialized()
    completed = False
    try:
        yield
        completed = True
    finally:
        if not group_was_initialized and dist.is_initialized():
            cleanup_distributed(barrier=completed)


def resolve_node_consistent_value(
    context: DistributedContext,
    resolver: Callable[[], _T],
    *,
    description: str,
) -> _T:
    """Resolve once per node and require every node leader to return the same value."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string.")
    if not context.is_distributed:
        return resolver()

    local_value = None
    local_error = None
    local_exception = None
    if context.local_rank == 0:
        try:
            local_value = resolver()
        except Exception as exc:  # pragma: no cover - exercised in a real process group
            local_exception = exc
            local_error = repr(exc)

    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(
        gathered,
        {
            "rank": context.rank,
            "is_node_leader": context.local_rank == 0,
            "value": local_value,
            "error": local_error,
        },
    )
    if not all(isinstance(item, Mapping) for item in gathered):
        raise RuntimeError(f"Distributed {description} exchange returned malformed payloads.")
    leader_results = [item for item in gathered if item.get("is_node_leader") is True]
    errors = [item for item in leader_results if item.get("error") is not None]
    if errors:
        if local_exception is not None:
            raise local_exception
        first = errors[0]
        raise RuntimeError(
            f"Node leader rank {first.get('rank')} rejected {description}: {first.get('error')}"
        )
    values = [item.get("value") for item in leader_results]
    if not values or any(value is None for value in values):
        raise RuntimeError(f"Every node leader must resolve {description}.")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{description.capitalize()} differs between node-local filesystems.")
    return values[0]


def propagate_distributed_error(
    context: DistributedContext,
    error: BaseException | None,
    *,
    description: str,
) -> None:
    """Make every rank raise before peers enter a later collective or DDP backward."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string.")
    if not context.is_distributed:
        if error is not None:
            raise error
        return

    local_message = None if error is None else f"{type(error).__name__}: {error}"
    gathered: list[str | None] = [None] * context.world_size
    dist.all_gather_object(gathered, local_message)
    failures = [(rank, message) for rank, message in enumerate(gathered) if message is not None]
    if not failures:
        return
    if error is not None:
        raise error
    rank, message = failures[0]
    raise RuntimeError(f"Rank {rank} failed during {description}: {message}")


def propagate_process_group_error(
    error: BaseException | None,
    *,
    description: str,
) -> None:
    """Propagate a rank-local failure through the currently initialized group.

    Trainer checkpoint hooks do not own the :class:`DistributedContext` used to
    launch the run, but every rank still has to reach the same verdict before a
    later barrier.  This variant derives rank/world size from the live process
    group and also preserves the original exception on the rank that raised it.
    """
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string.")
    if not dist.is_available() or not dist.is_initialized():
        if error is not None:
            raise error
        return

    local_message = None if error is None else f"{type(error).__name__}: {error}"
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_message)
    failures = [(rank, message) for rank, message in enumerate(gathered) if message is not None]
    if not failures:
        return
    if error is not None:
        raise error
    rank, message = failures[0]
    raise RuntimeError(f"Rank {rank} failed during {description}: {message}")


def run_distributed_operation(
    context: DistributedContext,
    operation: Callable[[], _T],
    *,
    description: str,
) -> _T:
    """Run local startup work and make its exception a rank-consistent verdict."""
    result: _T | None = None
    error: BaseException | None = None
    try:
        result = operation()
    except Exception as exc:  # pragma: no cover - collective failure path
        error = exc
    propagate_distributed_error(context, error, description=description)
    return result  # type: ignore[return-value]


def require_process_group_consistent_value(
    value: _T,
    *,
    description: str,
) -> _T:
    """Require all ranks in the live group to present the same serializable value."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string.")
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        raise ValueError(f"{description.capitalize()} differs between distributed ranks.")
    return value


def shard_indices(total_items: int, *, rank: int, world_size: int) -> range:
    """Return a deterministic, disjoint strided shard for one global rank."""
    if total_items < 0:
        raise ValueError("total_items must be non-negative.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size.")
    return range(rank, total_items, world_size)


__all__ = [
    "DistributedContext",
    "cleanup_distributed",
    "distributed_cleanup_guard",
    "initialize_distributed",
    "propagate_distributed_error",
    "propagate_process_group_error",
    "require_process_group_consistent_value",
    "resolve_node_consistent_value",
    "run_distributed_operation",
    "shard_indices",
]
