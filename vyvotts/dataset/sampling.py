"""Deterministic variable-length batching under a per-batch frame budget."""

from __future__ import annotations

import hashlib
import json
import operator
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

_STATE_VERSION = 1
LATENT_NUM_FRAMES_COLUMN = "latent_num_frames"


def _positive_integer(value: object, *, name: str) -> int:
    try:
        normalized = operator.index(value) if not isinstance(value, bool) else 0
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    try:
        normalized = operator.index(value) if not isinstance(value, bool) else -1
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _stable_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_latent_num_frames(value: object, *, name: str = LATENT_NUM_FRAMES_COLUMN) -> int:
    """Validate one persisted latent frame count without loading its latent array."""
    return _positive_integer(value, name=name)


def infer_latent_num_frames(latents: object, *, name: str = "latents") -> int:
    """Infer a legacy row's frame count from an already loaded ``[D, T]`` value."""
    shape = getattr(latents, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f"{name} must have shape [channels, frames].")
        return validate_latent_num_frames(shape[-1], name=f"{name} frame dimension")

    if isinstance(latents, (str, bytes)) or not isinstance(latents, Sequence) or not latents:
        raise ValueError(f"{name} must have shape [channels, frames].")
    try:
        frame_count = len(latents[0])
    except (TypeError, IndexError) as exc:
        raise ValueError(f"{name} must have shape [channels, frames].") from exc
    frame_count = validate_latent_num_frames(frame_count, name=f"{name} frame dimension")
    if any(not hasattr(channel, "__len__") or len(channel) != frame_count for channel in latents):
        raise ValueError(f"{name} channels must have a consistent positive frame dimension.")
    return frame_count


def read_dataset_frame_lengths(
    dataset: object,
    *,
    allow_legacy_inference: bool = False,
) -> list[int]:
    """Read validated frame metadata without materializing latent arrays.

    Prepared datasets should expose a column named ``latent_num_frames``. The
    legacy fallback deliberately requires an opt-in because indexing each row can
    deserialize the full latent array and make sampler construction expensive.
    """
    if not isinstance(allow_legacy_inference, bool):
        raise TypeError("allow_legacy_inference must be a boolean.")
    try:
        dataset_size = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("dataset must be sized and indexable.") from exc
    if dataset_size == 0:
        raise ValueError("Cannot read frame lengths from an empty dataset.")

    column_names = getattr(dataset, "column_names", None)
    if column_names is None:
        first_row = dataset[0]  # type: ignore[index]
        column_names = tuple(first_row.keys()) if isinstance(first_row, Mapping) else ()
    has_persisted_lengths = LATENT_NUM_FRAMES_COLUMN in column_names

    if has_persisted_lengths:
        try:
            raw_lengths = dataset[LATENT_NUM_FRAMES_COLUMN]  # type: ignore[index]
        except (IndexError, KeyError, TypeError):
            raw_lengths = []
            for index in range(dataset_size):
                row = dataset[index]  # type: ignore[index]
                if not isinstance(row, Mapping) or LATENT_NUM_FRAMES_COLUMN not in row:
                    raise ValueError(
                        f"Dataset row {index} has no {LATENT_NUM_FRAMES_COLUMN!r} value, "
                        "despite advertising that column."
                    )
                raw_lengths.append(row[LATENT_NUM_FRAMES_COLUMN])
        lengths = [
            validate_latent_num_frames(value, name=f"{LATENT_NUM_FRAMES_COLUMN}[{index}]")
            for index, value in enumerate(raw_lengths)
        ]
        if len(lengths) != dataset_size:
            raise ValueError(
                f"{LATENT_NUM_FRAMES_COLUMN} contains {len(lengths)} values for "
                f"a dataset of size {dataset_size}."
            )
        return lengths

    if not allow_legacy_inference:
        raise ValueError(
            f"Dataset has no {LATENT_NUM_FRAMES_COLUMN!r} column. Re-prepare it for efficient "
            "frame-budget batching, or set allow_legacy_inference=True to inspect latent rows."
        )

    lengths = []
    for index in range(dataset_size):
        row = dataset[index]  # type: ignore[index]
        if not isinstance(row, Mapping) or "latents" not in row:
            raise ValueError(f"Legacy dataset row {index} has no latents value.")
        lengths.append(infer_latent_num_frames(row["latents"], name=f"row {index} latents"))
    return lengths


class FrameBudgetBatchSampler:
    """Build deterministic sortish batches that stay within a frame budget.

    The global batch plan contains each dataset index exactly once. Distributed
    ranks take disjoint strided batches from that plan. If its batch count is not
    divisible by ``world_size``, ``drop_last=True`` removes the trailing remainder;
    ``pad_to_world_size=True`` instead repeats leading batches explicitly. With
    neither option, ranks remain duplicate-free but may receive one different
    number of batches.

    ``state_dict`` records the next batch to dispatch. This supports exact sampler
    resume when the data loader does not prefetch ahead of the training step being
    checkpointed. Call ``set_epoch`` at a new epoch boundary to reset that cursor.
    """

    def __init__(
        self,
        lengths: Iterable[int],
        *,
        max_frames: int,
        max_examples: int | None = None,
        bucket_size: int = 128,
        seed: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
        pad_to_world_size: bool = False,
    ) -> None:
        if isinstance(lengths, (str, bytes)):
            raise TypeError("lengths must be a non-empty sequence of positive frame counts.")
        try:
            raw_lengths = tuple(lengths)
        except TypeError as exc:
            raise TypeError(
                "lengths must be a non-empty sequence of positive frame counts."
            ) from exc
        if not raw_lengths:
            raise ValueError("lengths must contain at least one frame count.")

        self.max_frames = _positive_integer(max_frames, name="max_frames")
        self.max_examples = (
            None if max_examples is None else _positive_integer(max_examples, name="max_examples")
        )
        self.bucket_size = _positive_integer(bucket_size, name="bucket_size")
        self.seed = _nonnegative_integer(seed, name="seed")
        self._epoch = _nonnegative_integer(epoch, name="epoch")
        self.world_size = _positive_integer(world_size, name="world_size")
        self.rank = _nonnegative_integer(rank, name="rank")
        if self.rank >= self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a boolean.")
        if not isinstance(pad_to_world_size, bool):
            raise TypeError("pad_to_world_size must be a boolean.")
        if drop_last and pad_to_world_size:
            raise ValueError("drop_last and pad_to_world_size are mutually exclusive.")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.pad_to_world_size = pad_to_world_size

        normalized_lengths = []
        for index, length in enumerate(raw_lengths):
            try:
                normalized = _positive_integer(length, name=f"lengths[{index}]")
            except ValueError as exc:
                raise ValueError(
                    f"lengths[{index}] must be a positive integer frame count."
                ) from exc
            if normalized > self.max_frames:
                raise ValueError(
                    f"lengths[{index}]={normalized} exceeds max_frames={self.max_frames}."
                )
            normalized_lengths.append(normalized)
        self.lengths = tuple(normalized_lengths)

        self._config_digest = _stable_digest(
            {
                "lengths": self.lengths,
                "max_frames": self.max_frames,
                "max_examples": self.max_examples,
                "bucket_size": self.bucket_size,
                "seed": self.seed,
                "shuffle": self.shuffle,
                "rank": self.rank,
                "world_size": self.world_size,
                "drop_last": self.drop_last,
                "pad_to_world_size": self.pad_to_world_size,
            }
        )
        self._batches: tuple[tuple[int, ...], ...] = ()
        self._plan_digest = ""
        self._next_batch = 0
        self._rebuild_plan()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def sampler(self) -> "FrameBudgetBatchSampler":
        """Expose epoch control through Accelerate's nested batch-sampler contract."""
        return self

    @property
    def next_batch(self) -> int:
        """Index of the next local batch that will be dispatched."""
        return self._next_batch

    @property
    def remaining_batches(self) -> int:
        return len(self._batches) - self._next_batch

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic epoch plan and rewind its dispatch cursor."""
        self._epoch = _nonnegative_integer(epoch, name="epoch")
        self._next_batch = 0
        self._rebuild_plan()

    def _epoch_seed(self) -> int:
        digest = hashlib.sha256(f"{self.seed}:{self._epoch}".encode("ascii")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _ordered_indices(self, rng: random.Random) -> list[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            return indices

        # Sort random mega-buckets by descending length. This keeps nearby items
        # padding-friendly while changing bucket membership and ties each epoch.
        rng.shuffle(indices)
        ordered = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=self.lengths.__getitem__, reverse=True)
            ordered.extend(bucket)
        return ordered

    def _pack_batches(self, ordered_indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_frames = 0
        for index in ordered_indices:
            length = self.lengths[index]
            exceeds_frames = current_frames + length > self.max_frames
            exceeds_examples = self.max_examples is not None and len(current) >= self.max_examples
            if current and (exceeds_frames or exceeds_examples):
                batches.append(current)
                current = []
                current_frames = 0
            current.append(index)
            current_frames += length
        if current:
            batches.append(current)
        return batches

    def _rebuild_plan(self) -> None:
        rng = random.Random(self._epoch_seed())
        global_batches = self._pack_batches(self._ordered_indices(rng))
        if self.shuffle:
            rng.shuffle(global_batches)

        remainder = len(global_batches) % self.world_size
        if remainder and self.drop_last:
            global_batches = global_batches[:-remainder]
        elif remainder and self.pad_to_world_size:
            required = self.world_size - remainder
            original = tuple(global_batches)
            global_batches.extend(
                list(original[index % len(original)]) for index in range(required)
            )

        local_batches = global_batches[self.rank :: self.world_size]
        self._batches = tuple(tuple(batch) for batch in local_batches)
        self._plan_digest = _stable_digest(self._batches)

    def __iter__(self) -> Iterator[list[int]]:
        while self._next_batch < len(self._batches):
            batch = list(self._batches[self._next_batch])
            # Advance before yielding so a checkpoint taken immediately after
            # receipt resumes at the following batch.
            self._next_batch += 1
            yield batch

    def __len__(self) -> int:
        """Return total local batches in the epoch, independent of resume cursor."""
        return len(self._batches)

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable exact-dispatch resume record."""
        return {
            "version": _STATE_VERSION,
            "config_digest": self._config_digest,
            "epoch": self._epoch,
            "next_batch": self._next_batch,
            "plan_digest": self._plan_digest,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore an exact local dispatch cursor after validating the plan."""
        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping.")
        if state.get("version") != _STATE_VERSION:
            raise ValueError(f"Unsupported sampler state version: {state.get('version')!r}.")
        if state.get("config_digest") != self._config_digest:
            raise ValueError("Sampler state does not match this dataset or batching configuration.")

        epoch = _nonnegative_integer(state.get("epoch"), name="state epoch")
        next_batch = _nonnegative_integer(state.get("next_batch"), name="state next_batch")
        previous = (self._epoch, self._batches, self._plan_digest, self._next_batch)
        self._epoch = epoch
        try:
            self._rebuild_plan()
            if state.get("plan_digest") != self._plan_digest:
                raise ValueError("Sampler state plan does not match the deterministic epoch plan.")
            if next_batch > len(self._batches):
                raise ValueError(
                    f"state next_batch={next_batch} exceeds local batch count {len(self._batches)}."
                )
        except BaseException:
            self._epoch, self._batches, self._plan_digest, self._next_batch = previous
            raise
        self._next_batch = next_batch


__all__ = [
    "LATENT_NUM_FRAMES_COLUMN",
    "FrameBudgetBatchSampler",
    "infer_latent_num_frames",
    "read_dataset_frame_lengths",
    "validate_latent_num_frames",
]
