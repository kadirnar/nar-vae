"""CPU-only tests for Trainer frame-budget DataLoader integration."""

from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from vyvotts.configuration import resolve_frame_budget_batching
from vyvotts.dataset.sampling import FrameBudgetBatchSampler
from vyvotts.training_data import (
    FrameBudgetTrainerMixin,
    build_accumulation_loss_normalization,
    build_frame_budget_train_dataloader,
)

_TRAINING_EXTRAS_AVAILABLE = True
try:
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
    from torch import nn
    from transformers import TrainingArguments

    from vyvotts.finetune import EchoDiTFineTuner
    from vyvotts.train import EchoDiTTrainer
except (ImportError, RuntimeError):
    _TRAINING_EXTRAS_AVAILABLE = False
    BatchSamplerShard = None
    DataLoaderShard = None
    EchoDiTFineTuner = None
    EchoDiTTrainer = None
    TrainingArguments = None
    nn = None


class MetadataDataset:
    column_names = ("latent_num_frames", "sample_id")

    def __init__(self, lengths):
        self.lengths = list(lengths)
        self.row_reads = 0

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, key):
        if key == "latent_num_frames":
            return self.lengths
        if isinstance(key, int):
            self.row_reads += 1
            return {
                "latent_num_frames": self.lengths[key],
                "sample_id": key,
            }
        raise KeyError(key)


class CapturingAccelerator:
    def __init__(self):
        self.input_loader = None
        self.even_batches = True
        self.even_batches_during_prepare = None

    def prepare(self, dataloader):
        self.input_loader = dataloader
        self.even_batches_during_prepare = self.even_batches
        return dataloader


def collate_ids(rows):
    return [row["sample_id"] for row in rows]


@unittest.skipUnless(_TRAINING_EXTRAS_AVAILABLE, "training extras are not installed")
class TrainingDataLoaderTest(unittest.TestCase):
    def _trainer(self, lengths):
        dataset = MetadataDataset(lengths)
        accelerator = CapturingAccelerator()
        args = SimpleNamespace(
            data_seed=23,
            seed=99,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            dataloader_persistent_workers=False,
            dataloader_prefetch_factor=None,
            process_index=3,
            world_size=1,
            remove_unused_columns=False,
        )
        return (
            SimpleNamespace(
                train_dataset=dataset,
                data_collator=collate_ids,
                args=args,
                accelerator=accelerator,
            ),
            dataset,
            accelerator,
        )

    def test_both_training_entry_points_share_the_frame_budget_mixin(self):
        self.assertTrue(issubclass(EchoDiTTrainer, FrameBudgetTrainerMixin))
        self.assertTrue(issubclass(EchoDiTFineTuner, FrameBudgetTrainerMixin))

    def test_accumulation_normalization_counts_each_objective_over_full_window(self):
        first = {
            "latents": torch.zeros(1, 2, 4),
            "latent_mask": torch.tensor([[True, True, False, False]]),
            "conditioning_ids": torch.ones(1, 3, dtype=torch.long),
            "conditioning_mask": torch.tensor([[True, True, False]]),
        }
        second = {
            "latents": torch.zeros(2, 2, 4),
            "latent_mask": torch.tensor([[True, True, True, False], [True, True, True, True]]),
            "conditioning_ids": torch.ones(2, 4, dtype=torch.long),
            "conditioning_mask": torch.tensor(
                [[True, True, True, False], [True, True, False, False]]
            ),
        }

        counts = build_accumulation_loss_normalization([first, second], torch.device("cpu"))

        self.assertEqual(counts.velocity_elements.item(), 18)
        self.assertEqual(counts.examples.item(), 3)
        self.assertEqual(counts.text_tokens.item(), 7)
        self.assertEqual(counts.alignment_frames.item(), 9)

    def test_optimizer_window_collection_supports_old_and_new_transformers_signatures(self):
        batches = [
            {
                "latents": torch.zeros(1, 2, 3),
                "latent_mask": torch.tensor([[True, True, False]]),
                "conditioning_ids": torch.ones(1, 2, dtype=torch.long),
                "conditioning_mask": torch.tensor([[True, False]]),
            },
            {
                "latents": torch.zeros(1, 2, 4),
                "latent_mask": torch.tensor([[True, True, True, True]]),
                "conditioning_ids": torch.ones(1, 2, dtype=torch.long),
                "conditioning_mask": torch.tensor([[True, True]]),
            },
        ]

        class Trainer(FrameBudgetTrainerMixin):
            args = SimpleNamespace(device=torch.device("cpu"))

        old_batches, old_counts = Trainer().get_batch_samples(iter(batches), 2)
        new_batches, new_counts = Trainer().get_batch_samples(
            iter(batches),
            2,
            torch.device("cpu"),
        )

        self.assertEqual(len(old_batches), 2)
        self.assertEqual(len(new_batches), 2)
        self.assertEqual(old_counts.velocity_elements.item(), 12)
        self.assertEqual(new_counts.velocity_elements.item(), 12)
        self.assertEqual(old_counts.examples.item(), 2)
        self.assertEqual(old_counts.text_tokens.item(), 3)
        self.assertEqual(old_counts.alignment_frames.item(), 6)

    def test_both_training_entry_points_build_the_custom_loader(self):
        config = {
            "batch_size": 2,
            "max_frames_per_batch": 10,
            "frame_bucket_size": 4,
        }
        for trainer_type, config_key in (
            (EchoDiTTrainer, "training_config"),
            (EchoDiTFineTuner, "config"),
        ):
            with self.subTest(trainer=trainer_type.__name__), TemporaryDirectory() as directory:
                arguments = TrainingArguments(
                    output_dir=directory,
                    per_device_train_batch_size=2,
                    remove_unused_columns=False,
                    dataloader_pin_memory=False,
                    report_to=[],
                )
                trainer = trainer_type(
                    **{
                        config_key: config,
                        "model": nn.Linear(1, 1),
                        "args": arguments,
                        "train_dataset": MetadataDataset([3, 4]),
                        "data_collator": collate_ids,
                    }
                )

                batches = list(trainer.get_train_dataloader())

                self.assertTrue(trainer.model_accepts_loss_kwargs)
                self.assertEqual(len(batches), 1)
                self.assertEqual(sorted(batches[0]), [0, 1])
                self.assertIsInstance(
                    trainer._frame_budget_batch_sampler,
                    FrameBudgetBatchSampler,
                )

    def test_builder_uses_metadata_and_constructs_one_global_batch_plan(self):
        trainer, dataset, accelerator = self._trainer([7, 5, 4, 3, 2])
        settings = resolve_frame_budget_batching(
            {
                "batch_size": 3,
                "max_frames_per_batch": 10,
                "frame_bucket_size": 4,
            }
        )

        dataloader = build_frame_budget_train_dataloader(trainer, settings)
        sampler = trainer._frame_budget_batch_sampler

        self.assertIs(dataloader, accelerator.input_loader)
        self.assertTrue(accelerator.even_batches)
        self.assertFalse(accelerator.even_batches_during_prepare)
        self.assertEqual(dataset.row_reads, 0)
        self.assertIs(dataloader.batch_sampler, sampler)
        self.assertIsNone(dataloader.batch_size)
        self.assertEqual(sampler.rank, 0)
        self.assertEqual(sampler.world_size, 1)
        self.assertTrue(sampler.drop_last)
        self.assertEqual(sampler.max_examples, 3)
        self.assertEqual(sampler.seed, 23)

        batches = list(dataloader)
        self.assertEqual(dataset.row_reads, len(dataset))
        self.assertEqual(sorted(index for batch in batches for index in batch), list(range(5)))
        for batch in batches:
            self.assertLessEqual(sum(dataset.lengths[index] for index in batch), 10)
            self.assertLessEqual(len(batch), 3)

    def test_disabled_setting_delegates_to_the_normal_trainer_loader(self):
        sentinel = object()

        class NormalLoader:
            def get_train_dataloader(self):
                return sentinel

        class Trainer(FrameBudgetTrainerMixin, NormalLoader):
            training_config = {"max_frames_per_batch": 0}

        self.assertIs(Trainer().get_train_dataloader(), sentinel)

    def test_distributed_loader_requires_drop_last_for_equal_steps(self):
        trainer, _, _ = self._trainer([2, 3, 4])
        trainer.args.world_size = 2
        trainer.args.dataloader_drop_last = False
        settings = resolve_frame_budget_batching(
            {
                "batch_size": 2,
                "max_frames_per_batch": 6,
            }
        )

        with self.assertRaisesRegex(ValueError, "dataloader_drop_last: true"):
            build_frame_budget_train_dataloader(trainer, settings)

    def test_accelerate_is_the_only_distributed_sharder(self):
        def batches_for_rank(rank):
            global_sampler = FrameBudgetBatchSampler(
                [5, 3, 2, 2, 1, 1],
                max_frames=5,
                shuffle=False,
                rank=0,
                world_size=1,
                drop_last=True,
            )
            shard = BatchSamplerShard(
                global_sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=False,
            )
            return global_sampler, list(shard)

        left_sampler, left = batches_for_rank(0)
        right_sampler, right = batches_for_rank(1)

        self.assertEqual((left_sampler.rank, left_sampler.world_size), (0, 1))
        self.assertEqual((right_sampler.rank, right_sampler.world_size), (0, 1))
        self.assertEqual(left, [[0]])
        self.assertEqual(right, [[1, 2]])
        self.assertTrue(set(sum(left, [])).isdisjoint(sum(right, [])))
        self.assertEqual(set(sum(left, []) + sum(right, [])), {0, 1, 2})
        self.assertTrue({3, 4, 5}.isdisjoint(sum(left, []) + sum(right, [])))

    def test_accelerate_set_epoch_reaches_the_nested_sampler(self):
        sampler = FrameBudgetBatchSampler(
            [2, 3, 4, 5],
            max_frames=6,
            seed=17,
            drop_last=True,
        )
        shard = BatchSamplerShard(
            sampler,
            num_processes=2,
            process_index=0,
            split_batches=False,
            even_batches=False,
        )
        shard.sampler = sampler
        loader = DataLoaderShard(
            list(range(4)),
            batch_sampler=shard,
            collate_fn=lambda rows: rows,
        )

        loader.set_epoch(5)

        self.assertIs(sampler.sampler, sampler)
        self.assertEqual(sampler.epoch, 5)
        self.assertEqual(sampler.next_batch, 0)


if __name__ == "__main__":
    unittest.main()
