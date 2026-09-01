"""Training-only DataLoader integration for deterministic frame-budget batches."""

from __future__ import annotations

import hashlib
import json
import math
import operator
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from nar_vae.configuration import (
    FrameBudgetBatchingSettings,
    resolve_frame_budget_batching,
)
from nar_vae.dataset.sampling import (
    LATENT_NUM_FRAMES_COLUMN,
    FrameBudgetBatchSampler,
    read_dataset_frame_lengths,
)
from nar_vae.dataset.utterance_store import (
    CONDITIONING_NUM_TOKENS_COLUMN,
    SPEAKER_NUM_FRAMES_COLUMN,
)
from nar_vae.losses.flow_matching_loss import AccumulationLossNormalization


def _batch_valid_counts(batch: dict[str, Any]) -> torch.Tensor:
    """Return packed velocity, example, alignment-token, and frame counts."""
    latents = batch.get("latents")
    conditioning_ids = batch.get("conditioning_ids")
    if not isinstance(latents, torch.Tensor) or latents.ndim != 3:
        raise ValueError("Training batches must contain latents with shape [B, D, T].")
    if not isinstance(conditioning_ids, torch.Tensor) or conditioning_ids.ndim != 2:
        raise ValueError("Training batches must contain conditioning_ids with shape [B, L].")
    batch_size, latent_size, latent_frames = latents.shape
    if batch_size <= 0 or latent_size <= 0 or latent_frames <= 0:
        raise ValueError("Training batches cannot contain empty latent dimensions.")
    if conditioning_ids.shape[0] != batch_size or conditioning_ids.shape[1] <= 0:
        raise ValueError("conditioning_ids must have the same non-empty batch dimension.")

    latent_mask = batch.get("latent_mask")
    if latent_mask is None:
        valid_frames = torch.tensor(
            batch_size * latent_frames,
            device=latents.device,
            dtype=torch.long,
        )
    else:
        if not isinstance(latent_mask, torch.Tensor) or tuple(latent_mask.shape) != (
            batch_size,
            latent_frames,
        ):
            raise ValueError("latent_mask must match the batch and latent-frame dimensions.")
        valid_frames = latent_mask.to(device=latents.device, dtype=torch.bool).sum(dtype=torch.long)

    conditioning_mask = batch.get("conditioning_mask")
    if conditioning_mask is not None:
        if not isinstance(conditioning_mask, torch.Tensor) or tuple(
            conditioning_mask.shape
        ) != tuple(conditioning_ids.shape):
            raise ValueError("conditioning_mask must have the conditioning_ids shape.")
        conditioning_mask = conditioning_mask.to(device=latents.device, dtype=torch.bool)

    alignment_mask = batch.get("alignment_mask")
    if alignment_mask is None:
        alignment_mask = conditioning_mask
    else:
        if not isinstance(alignment_mask, torch.Tensor) or tuple(alignment_mask.shape) != tuple(
            conditioning_ids.shape
        ):
            raise ValueError("alignment_mask must have the conditioning_ids shape.")
        alignment_mask = alignment_mask.to(device=latents.device, dtype=torch.bool)
        if conditioning_mask is not None:
            torch._assert_async(
                ~(alignment_mask & ~conditioning_mask).any(),
                "alignment_mask cannot select padded conditioning tokens.",
            )

    valid_tokens = (
        torch.tensor(conditioning_ids.numel(), device=latents.device, dtype=torch.long)
        if alignment_mask is None
        else alignment_mask.sum(dtype=torch.long)
    )

    counts = torch.stack(
        (
            valid_frames * latent_size,
            torch.tensor(batch_size, device=latents.device, dtype=torch.long),
            valid_tokens,
            valid_frames,
        )
    )
    torch._assert_async(
        (counts > 0).all(),
        "Every optimizer window requires valid audio frames and alignment tokens.",
    )
    return counts


def build_accumulation_loss_normalization(
    batch_samples: list[dict[str, Any]],
    device: torch.device,
) -> AccumulationLossNormalization:
    """Count each objective's valid items over an optimizer accumulation window."""
    if not batch_samples:
        raise ValueError("Cannot normalize an empty gradient-accumulation window.")
    totals = torch.stack([_batch_valid_counts(batch) for batch in batch_samples]).sum(dim=0)
    totals = totals.to(device=device, dtype=torch.float32)
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    world_size = dist.get_world_size() if distributed else 1
    if distributed:
        # One packed collective per optimizer window replaces two collectives for
        # every objective in every microbatch. The globally reduced denominator
        # is sufficient for exact gradients after DDP's gradient average.
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    torch._assert_async(
        torch.isfinite(totals).all() & (totals > 0).all(),
        "Accumulation-window valid-item totals must be finite and positive.",
    )
    return AccumulationLossNormalization(
        *totals.unbind(),
        globally_reduced=True,
        world_size=world_size,
    )


def _stable_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _positive_integer(value: object, *, name: str) -> int:
    try:
        normalized = operator.index(value) if not isinstance(value, bool) else 0
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return normalized


def _validated_lengths(
    values: Iterable[object],
    *,
    expected_size: int,
    name: str,
    allow_zero: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain one integer length per dataset row.")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain one integer length per dataset row.") from exc
    if len(raw_values) != expected_size:
        raise ValueError(
            f"{name} contains {len(raw_values)} values for a dataset of size {expected_size}."
        )
    normalized = []
    minimum = 0 if allow_zero else 1
    for index, value in enumerate(raw_values):
        try:
            length = operator.index(value) if not isinstance(value, bool) else -1
        except TypeError as exc:
            raise ValueError(f"{name}[{index}] must be an integer.") from exc
        if length < minimum:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name}[{index}] must be a {qualifier} integer.")
        normalized.append(length)
    return tuple(normalized)


def _read_dataset_length_metadata(
    dataset: object,
    column: str,
    *,
    allow_zero: bool = False,
) -> tuple[int, ...]:
    """Read one cheap length vector, including from dynamic dataset wrappers.

    Wrappers that choose a reference per epoch can expose
    ``get_length_metadata(column)``. The sampler calls that hook again after
    ``set_epoch`` so its cost plan follows the selected references without
    materializing codec latents.
    """
    try:
        dataset_size = len(dataset)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("dataset must be sized and indexable.") from exc
    if dataset_size <= 0:
        raise ValueError("Cannot build a batch plan for an empty dataset.")

    metadata_hook = getattr(dataset, "get_length_metadata", None)
    if callable(metadata_hook):
        try:
            values = metadata_hook(column)
        except (KeyError, NotImplementedError) as exc:
            raise ValueError(f"Dataset length-metadata hook does not provide {column!r}.") from exc
    else:
        column_names = getattr(dataset, "column_names", ())
        if column not in column_names:
            raise ValueError(
                f"Padded-attention batching requires a persisted {column!r} column or a "
                "get_length_metadata(column) dataset hook."
            )
        try:
            values = dataset[column]  # type: ignore[index]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"Could not read dataset length metadata {column!r}.") from exc
    return _validated_lengths(
        values,
        expected_size=dataset_size,
        name=column,
        allow_zero=allow_zero,
    )


def _set_dataset_epoch(dataset: object, epoch: int) -> bool:
    """Propagate an epoch to a dynamic wrapper without depending on its class."""
    set_epoch = getattr(dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
        return True
    nested = getattr(dataset, "dataset", None)
    if nested is not None and nested is not dataset:
        return _set_dataset_epoch(nested, epoch)
    return False


class EpochAwareFrameBudgetBatchSampler(FrameBudgetBatchSampler):
    """Legacy frame-budget sampler that also advances dynamic reference data."""

    def __init__(self, *args, epoch_target: object | None = None, **kwargs) -> None:
        self._epoch_target = epoch_target
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        if self._epoch_target is not None:
            _set_dataset_epoch(self._epoch_target, epoch)
        super().set_epoch(epoch)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        epoch = state.get("epoch") if isinstance(state, Mapping) else None
        previous_dataset_epoch = (
            getattr(self._epoch_target, "epoch", None) if self._epoch_target is not None else None
        )
        if (
            self._epoch_target is not None
            and isinstance(epoch, int)
            and not isinstance(epoch, bool)
            and epoch >= 0
        ):
            _set_dataset_epoch(self._epoch_target, epoch)
        try:
            super().load_state_dict(state)
        except BaseException:
            if (
                self._epoch_target is not None
                and isinstance(previous_dataset_epoch, int)
                and not isinstance(previous_dataset_epoch, bool)
                and previous_dataset_epoch >= 0
            ):
                _set_dataset_epoch(self._epoch_target, previous_dataset_epoch)
            raise


class PaddedAttentionBatchSampler(EpochAwareFrameBudgetBatchSampler):
    """Pack examples under both frame and padded transformer-attention budgets.

    The estimate covers target self-attention, target-to-text/reference cross
    attention, and the two conditioning encoders for the padded batch shape::

        B * (T^2 + T*L + T*S + L^2 + S^2)

    ``T`` and ``S`` are padded target/reference lengths after their configured
    patching. This is a conservative device-independent proxy rather than a
    promise about wall time.
    Consecutive groups of ``step_world_size`` batches have adjacent estimated
    costs, which reduces stragglers after Accelerate assigns one batch per rank.
    """

    def __init__(
        self,
        target_lengths: Iterable[int],
        *,
        text_lengths: Iterable[int],
        speaker_frame_lengths: Iterable[int],
        max_attention_cost: int,
        speaker_patch_size: int,
        target_patch_size: int = 1,
        step_world_size: int = 1,
        metadata_provider: Callable[[], tuple[Iterable[int], Iterable[int]]] | None = None,
        **kwargs,
    ) -> None:
        target_lengths = tuple(target_lengths)
        dataset_size = len(target_lengths)
        if dataset_size == 0:
            raise ValueError("target_lengths must contain at least one frame count.")
        self.max_attention_cost = _positive_integer(
            max_attention_cost,
            name="max_attention_cost",
        )
        self.speaker_patch_size = _positive_integer(
            speaker_patch_size,
            name="speaker_patch_size",
        )
        self.target_patch_size = _positive_integer(
            target_patch_size,
            name="target_patch_size",
        )
        self.step_world_size = _positive_integer(
            step_world_size,
            name="step_world_size",
        )
        self._metadata_provider = metadata_provider
        self._metadata_refresh_enabled = False
        self._metadata_size = dataset_size
        self.text_lengths = _validated_lengths(
            text_lengths,
            expected_size=dataset_size,
            name=CONDITIONING_NUM_TOKENS_COLUMN,
            allow_zero=False,
        )
        self.speaker_frame_lengths = _validated_lengths(
            speaker_frame_lengths,
            expected_size=dataset_size,
            name=SPEAKER_NUM_FRAMES_COLUMN,
            allow_zero=True,
        )
        super().__init__(target_lengths, **kwargs)
        self._config_digest = _stable_digest(
            {
                "frame_budget_config": self._config_digest,
                "text_lengths": self.text_lengths,
                "max_attention_cost": self.max_attention_cost,
                "speaker_patch_size": self.speaker_patch_size,
                "target_patch_size": self.target_patch_size,
                "step_world_size": self.step_world_size,
            }
        )
        self._metadata_refresh_enabled = True

    def _refresh_length_metadata(self) -> None:
        if not self._metadata_refresh_enabled or self._metadata_provider is None:
            return
        text_lengths, speaker_frame_lengths = self._metadata_provider()
        self.text_lengths = _validated_lengths(
            text_lengths,
            expected_size=self._metadata_size,
            name=CONDITIONING_NUM_TOKENS_COLUMN,
            allow_zero=False,
        )
        self.speaker_frame_lengths = _validated_lengths(
            speaker_frame_lengths,
            expected_size=self._metadata_size,
            name=SPEAKER_NUM_FRAMES_COLUMN,
            allow_zero=True,
        )

    def _speaker_tokens(self, index: int) -> int:
        return math.ceil(self.speaker_frame_lengths[index] / self.speaker_patch_size)

    def batch_cost(self, indices: Sequence[int]) -> int:
        """Return the integer padded-attention proxy for one non-empty batch."""
        if not indices:
            raise ValueError("Cannot estimate the cost of an empty batch.")
        batch_size = len(indices)
        target = math.ceil(max(self.lengths[index] for index in indices) / self.target_patch_size)
        text = max(self.text_lengths[index] for index in indices)
        speaker = max(self._speaker_tokens(index) for index in indices)
        return batch_size * (
            target * target + target * text + target * speaker + text * text + speaker * speaker
        )

    def _ordered_indices(self, rng: random.Random) -> list[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            return indices
        rng.shuffle(indices)
        ordered = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.batch_cost((index,)), reverse=True)
            ordered.extend(bucket)
        return ordered

    def _pack_batches(self, ordered_indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_frames = 0
        for index in ordered_indices:
            if self.batch_cost((index,)) > self.max_attention_cost:
                raise ValueError(
                    f"Example {index} has padded attention cost {self.batch_cost((index,))}, "
                    f"which exceeds max_attention_cost={self.max_attention_cost}."
                )
            candidate = [*current, index]
            exceeds_frames = current_frames + self.lengths[index] > self.max_frames
            exceeds_examples = self.max_examples is not None and len(current) >= self.max_examples
            exceeds_attention = self.batch_cost(candidate) > self.max_attention_cost
            if current and (exceeds_frames or exceeds_examples or exceeds_attention):
                batches.append(current)
                current = [index]
                current_frames = self.lengths[index]
            else:
                current = candidate
                current_frames += self.lengths[index]
        if current:
            batches.append(current)
        return batches

    def _rebuild_plan(self) -> None:
        self._refresh_length_metadata()
        super()._rebuild_plan()
        if not self.shuffle or self.step_world_size == 1 or len(self._batches) < 2:
            return

        batches = sorted(self._batches, key=self.batch_cost, reverse=True)
        full_count = len(batches) - (len(batches) % self.step_world_size)
        full_groups = [
            list(batches[start : start + self.step_world_size])
            for start in range(0, full_count, self.step_world_size)
        ]
        remainder = list(batches[full_count:])
        rng = random.Random(self._epoch_seed() ^ 0x4E4152564145)
        for group in full_groups:
            rng.shuffle(group)
        rng.shuffle(full_groups)
        balanced = [batch for group in full_groups for batch in group]
        balanced.extend(remainder)
        self._batches = tuple(tuple(batch) for batch in balanced)
        self._plan_digest = _stable_digest(self._batches)


def _dataset_and_collator(trainer: Any) -> tuple[object, Any]:
    """Mirror Trainer's unused-column behavior before constructing our DataLoader."""
    dataset = trainer.train_dataset
    data_collator = trainer.data_collator
    if not trainer.args.remove_unused_columns:
        return dataset, data_collator

    from transformers.utils import is_datasets_available

    if is_datasets_available():
        import datasets

        if isinstance(dataset, datasets.Dataset):
            return (
                trainer._remove_unused_columns(dataset, description="Training"),
                data_collator,
            )
    return (
        dataset,
        trainer._get_collator_with_removed_columns(
            data_collator,
            description="Training",
        ),
    )


def build_frame_budget_train_dataloader(
    trainer: Any,
    settings: FrameBudgetBatchingSettings,
) -> Any:
    """Build one global batch plan and let Accelerate shard its batches.

    The sampler deliberately uses rank zero and world size one on every
    process. Accelerator.prepare then wraps it in BatchSamplerShard and
    performs the only distributed sharding step. The sampler's drop_last
    attribute is copied from TrainingArguments so Accelerate can trim an
    incomplete group of global batches and keep DDP ranks step-aligned.
    """
    if not settings.enabled:
        raise ValueError("Frame-budget batching settings are disabled.")
    if trainer.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")
    if trainer.args.world_size > 1 and not trainer.args.dataloader_drop_last:
        raise ValueError(
            "Distributed frame-budget batching requires dataloader_drop_last: true "
            "so every rank executes the same number of optimizer steps."
        )

    _set_dataset_epoch(trainer.train_dataset, 0)
    if callable(getattr(trainer.train_dataset, "get_length_metadata", None)):
        frame_lengths = list(
            _read_dataset_length_metadata(
                trainer.train_dataset,
                LATENT_NUM_FRAMES_COLUMN,
            )
        )
    else:
        frame_lengths = read_dataset_frame_lengths(
            trainer.train_dataset,
            allow_legacy_inference=settings.allow_legacy_frame_length_inference,
        )
    data_seed = trainer.args.data_seed
    if data_seed is None:
        data_seed = trainer.args.seed
    sampler_kwargs = {
        "max_frames": settings.max_frames_per_batch,
        "max_examples": settings.max_examples_per_batch,
        "bucket_size": settings.frame_bucket_size,
        "seed": data_seed,
        "rank": 0,
        "world_size": 1,
        "drop_last": trainer.args.dataloader_drop_last,
        "epoch_target": trainer.train_dataset,
    }
    training_config = getattr(trainer, "training_config", {})
    batching_cost = training_config.get("batching_cost", "frames")
    if batching_cost == "frames":
        batch_sampler = EpochAwareFrameBudgetBatchSampler(
            frame_lengths,
            **sampler_kwargs,
        )
    elif batching_cost == "padded_attention":
        max_attention_cost = training_config.get("max_attention_cost_per_batch")
        if (
            isinstance(max_attention_cost, bool)
            or not isinstance(max_attention_cost, int)
            or max_attention_cost <= 0
        ):
            raise ValueError(
                "padded_attention batching requires a positive max_attention_cost_per_batch."
            )
        use_speaker_conditioning = training_config.get(
            "use_speaker_conditioning",
            True,
        )
        text_lengths = _read_dataset_length_metadata(
            trainer.train_dataset,
            CONDITIONING_NUM_TOKENS_COLUMN,
        )

        def attention_metadata() -> tuple[tuple[int, ...], tuple[int, ...]]:
            if use_speaker_conditioning:
                speaker_lengths = _read_dataset_length_metadata(
                    trainer.train_dataset,
                    SPEAKER_NUM_FRAMES_COLUMN,
                    allow_zero=True,
                )
            else:
                speaker_lengths = (0,) * len(frame_lengths)
            return text_lengths, speaker_lengths

        _, speaker_lengths = attention_metadata()
        batch_sampler = PaddedAttentionBatchSampler(
            frame_lengths,
            text_lengths=text_lengths,
            speaker_frame_lengths=speaker_lengths,
            max_attention_cost=max_attention_cost,
            speaker_patch_size=training_config.get("speaker_patch_size", 4),
            target_patch_size=training_config.get("target_patch_size", 1),
            step_world_size=trainer.args.world_size,
            metadata_provider=attention_metadata,
            **sampler_kwargs,
        )
    else:
        raise ValueError("batching_cost must be 'frames' or 'padded_attention'.")
    trainer._frame_budget_batch_sampler = batch_sampler

    dataset, data_collator = _dataset_and_collator(trainer)
    from transformers.trainer_utils import seed_worker

    dataloader_params = {
        "batch_sampler": batch_sampler,
        "collate_fn": data_collator,
        "num_workers": trainer.args.dataloader_num_workers,
        "pin_memory": trainer.args.dataloader_pin_memory,
        "persistent_workers": trainer.args.dataloader_persistent_workers,
        "worker_init_fn": partial(
            seed_worker,
            num_workers=trainer.args.dataloader_num_workers,
            rank=trainer.args.process_index,
        ),
    }
    prefetch_factor = trainer.args.dataloader_prefetch_factor
    if prefetch_factor is not None:
        dataloader_params["prefetch_factor"] = prefetch_factor
    dataloader = DataLoader(dataset, **dataloader_params)

    # Accelerate <1.4 rejects variable-size batch samplers while even_batches is
    # enabled. drop_last on our sampler already gives BatchSamplerShard an exact
    # no-duplication rule for incomplete groups of global batches.
    previous_even_batches = trainer.accelerator.even_batches
    trainer.accelerator.even_batches = False
    try:
        prepared = trainer.accelerator.prepare(dataloader)
    finally:
        trainer.accelerator.even_batches = previous_even_batches

    # Accelerate 1.3 does not descend through BatchSamplerShard.batch_sampler
    # when propagating set_epoch. Attach the same sampler through its public
    # sampler convention so all supported Accelerate versions rebuild plans.
    prepared_batch_sampler = getattr(prepared, "batch_sampler", None)
    if getattr(prepared_batch_sampler, "batch_sampler", None) is batch_sampler:
        prepared_batch_sampler.sampler = batch_sampler
    return prepared


class FrameBudgetTrainerMixin:
    """Use frame-budget batches when enabled, otherwise retain Trainer behavior."""

    training_config: dict

    def get_batch_samples(self, epoch_iterator, num_batches, device=None):
        """Collect one optimizer window with exact TTS denominators on Transformers 4.x.

        Transformers 4.49 computes its built-in item count directly from a
        language-model ``labels`` field, while later 4.x releases delegate to
        ``_get_num_items_in_batch``.  TTS has no such labels and needs multiple
        denominators, so implement the complete stable behavior here rather than
        relying on either private implementation.
        """
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        if not batch_samples:
            return batch_samples, None
        normalization_device = self.args.device if device is None else device
        normalization = build_accumulation_loss_normalization(
            batch_samples,
            normalization_device,
        )
        return batch_samples, normalization

    def _get_num_items_in_batch(self, batch_samples, device):
        # Current 4.x prediction/evaluation paths also call this hook directly.
        return build_accumulation_loss_normalization(batch_samples, device)

    def get_train_dataloader(self):
        settings = resolve_frame_budget_batching(self.training_config)
        if not settings.enabled:
            return super().get_train_dataloader()
        return build_frame_budget_train_dataloader(self, settings)


__all__ = [
    "CONDITIONING_NUM_TOKENS_COLUMN",
    "EpochAwareFrameBudgetBatchSampler",
    "FrameBudgetTrainerMixin",
    "PaddedAttentionBatchSampler",
    "SPEAKER_NUM_FRAMES_COLUMN",
    "build_accumulation_loss_normalization",
    "build_frame_budget_train_dataloader",
]
