"""Training-only DataLoader integration for deterministic frame-budget batches."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader

from nar_vae.configuration import (
    FrameBudgetBatchingSettings,
    resolve_frame_budget_batching,
)
from nar_vae.dataset.sampling import (
    FrameBudgetBatchSampler,
    read_dataset_frame_lengths,
)
from nar_vae.losses.flow_matching_loss import AccumulationLossNormalization


def _batch_valid_counts(batch: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return velocity, example, text-token, and alignment counts for one batch."""
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
        valid_frames = batch_size * latent_frames
    else:
        if not isinstance(latent_mask, torch.Tensor) or tuple(latent_mask.shape) != (
            batch_size,
            latent_frames,
        ):
            raise ValueError("latent_mask must match the batch and latent-frame dimensions.")
        valid_frames = int(latent_mask.to(dtype=torch.bool).sum().item())

    conditioning_mask = batch.get("conditioning_mask")
    if conditioning_mask is None:
        valid_tokens = conditioning_ids.numel()
    else:
        if not isinstance(conditioning_mask, torch.Tensor) or tuple(
            conditioning_mask.shape
        ) != tuple(conditioning_ids.shape):
            raise ValueError("conditioning_mask must have the conditioning_ids shape.")
        valid_tokens = int(conditioning_mask.to(dtype=torch.bool).sum().item())

    if valid_frames <= 0 or valid_tokens <= 0:
        raise ValueError("Every optimizer window requires valid audio frames and text tokens.")
    return valid_frames * latent_size, batch_size, valid_tokens, valid_frames


def build_accumulation_loss_normalization(
    batch_samples: list[dict[str, Any]],
    device: torch.device,
) -> AccumulationLossNormalization:
    """Count each objective's valid items over an optimizer accumulation window."""
    if not batch_samples:
        raise ValueError("Cannot normalize an empty gradient-accumulation window.")
    totals = [0, 0, 0, 0]
    for batch in batch_samples:
        counts = _batch_valid_counts(batch)
        totals = [total + count for total, count in zip(totals, counts)]
    values = [torch.tensor(total, device=device, dtype=torch.float32) for total in totals]
    return AccumulationLossNormalization(*values)


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

    frame_lengths = read_dataset_frame_lengths(
        trainer.train_dataset,
        allow_legacy_inference=settings.allow_legacy_frame_length_inference,
    )
    data_seed = trainer.args.data_seed
    if data_seed is None:
        data_seed = trainer.args.seed
    batch_sampler = FrameBudgetBatchSampler(
        frame_lengths,
        max_frames=settings.max_frames_per_batch,
        max_examples=settings.max_examples_per_batch,
        bucket_size=settings.frame_bucket_size,
        seed=data_seed,
        rank=0,
        world_size=1,
        drop_last=trainer.args.dataloader_drop_last,
    )
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
    "FrameBudgetTrainerMixin",
    "build_accumulation_loss_normalization",
    "build_frame_budget_train_dataloader",
]
