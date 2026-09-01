"""CPU-only tests for Trainer frame-budget DataLoader integration."""

from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nar_vae.configuration import resolve_frame_budget_batching
from nar_vae.dataset.sampling import FrameBudgetBatchSampler
from nar_vae.training_data import (
    EpochAwareFrameBudgetBatchSampler,
    FrameBudgetTrainerMixin,
    PaddedAttentionBatchSampler,
    build_accumulation_loss_normalization,
    build_frame_budget_train_dataloader,
)

_TRAINING_DEPENDENCIES_AVAILABLE = True
try:
    from accelerate.data_loader import BatchSamplerShard, DataLoaderShard
    from torch import nn
    from transformers import TrainingArguments

    from nar_vae.finetune import EchoDiTFineTuner
    from nar_vae.train import EchoDiTTrainer, _materialize_flow_model_weights
except (ImportError, RuntimeError):
    _TRAINING_DEPENDENCIES_AVAILABLE = False
    BatchSamplerShard = None
    DataLoaderShard = None
    EchoDiTFineTuner = None
    EchoDiTTrainer = None
    _materialize_flow_model_weights = None
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


class AttentionMetadataDataset(MetadataDataset):
    column_names = (
        "latent_num_frames",
        "conditioning_num_tokens",
        "speaker_num_frames",
        "sample_id",
    )

    def __init__(self, lengths, text_lengths, speaker_lengths):
        super().__init__(lengths)
        self.text_lengths = list(text_lengths)
        self.speaker_lengths = list(speaker_lengths)

    def __getitem__(self, key):
        if key == "conditioning_num_tokens":
            return self.text_lengths
        if key == "speaker_num_frames":
            return self.speaker_lengths
        return super().__getitem__(key)


class PretrainingContractLinear(torch.nn.Linear):
    text_conditioning_mode = "frozen_features"
    generative_objective = "vp_diffusion_v"


class TrainingModelWrapper(torch.nn.Module):
    text_conditioning_mode = "frozen_features"
    generative_objective = "vp_diffusion_v"

    def __init__(self, module):
        super().__init__()
        self.module = module


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


@unittest.skipUnless(_TRAINING_DEPENDENCIES_AVAILABLE, "training dependencies are not installed")
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

    def test_flow_export_hardlinks_the_existing_trainer_state_dict(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            flow_output = output / "flow_model"
            flow_output.mkdir()
            trainer_weights = output / "pytorch_model.bin"
            trainer_weights.write_bytes(b"one serialized state dict")

            flow_weights = _materialize_flow_model_weights(output, flow_output, object())

            self.assertTrue(os.path.samefile(trainer_weights, flow_weights))
            self.assertEqual(flow_weights.read_bytes(), trainer_weights.read_bytes())

    def test_sft_export_reuses_trainer_weights_before_manifest_sealing(self):
        import nar_vae.finetune as finetune_module

        with TemporaryDirectory() as directory:
            output = Path(directory) / "sft-export"
            trainer = object.__new__(EchoDiTFineTuner)
            trainer.config = {}
            trainer.flow_model = object()
            trainer.ema_model = None
            trainer.parent_lineage = None
            trainer.parent_model_manifest = None
            trainer.is_world_process_zero = lambda: True
            root_payload = b"one exact SFT Trainer state dict"

            def save_trainer_weights(_trainer, output_dir, _internal_call=False):
                del _trainer, _internal_call
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "pytorch_model.bin").write_bytes(root_payload)

            def assert_manifest_input(flow_model_dir, *_args, **kwargs):
                root_weights = output / "pytorch_model.bin"
                flow_weights = Path(flow_model_dir) / "pytorch_model.bin"
                self.assertTrue(os.path.samefile(root_weights, flow_weights))
                self.assertEqual(
                    hashlib.sha256(flow_weights.read_bytes()).hexdigest(),
                    hashlib.sha256(root_payload).hexdigest(),
                )
                self.assertEqual(kwargs["checkpoint_files"], ["pytorch_model.bin"])

            with (
                patch.object(finetune_module.Trainer, "save_model", new=save_trainer_weights),
                patch.object(finetune_module, "write_training_lineage"),
                patch.object(
                    finetune_module,
                    "write_model_manifest",
                    side_effect=assert_manifest_input,
                ) as write_manifest,
                patch.object(finetune_module, "write_training_checkpoint_manifest"),
            ):
                trainer.save_model(output)

            write_manifest.assert_called_once()
            self.assertTrue(
                os.path.samefile(
                    output / "pytorch_model.bin",
                    output / "flow_model" / "pytorch_model.bin",
                )
            )

    def test_sft_export_keeps_cpu_state_dict_fallback_without_root_pytorch_artifact(self):
        import nar_vae.finetune as finetune_module

        with TemporaryDirectory() as directory:
            output = Path(directory) / "sft-safetensors-export"
            model = nn.Linear(3, 2)
            trainer = object.__new__(EchoDiTFineTuner)
            trainer.config = {}
            trainer.flow_model = model
            trainer.ema_model = None
            trainer.parent_lineage = None
            trainer.parent_model_manifest = None
            trainer.is_world_process_zero = lambda: True

            def save_without_pytorch_artifact(_trainer, output_dir, _internal_call=False):
                del _trainer, _internal_call
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            with (
                patch.object(
                    finetune_module.Trainer,
                    "save_model",
                    new=save_without_pytorch_artifact,
                ),
                patch.object(finetune_module, "write_training_lineage"),
                patch.object(finetune_module, "write_model_manifest"),
                patch.object(finetune_module, "write_training_checkpoint_manifest"),
            ):
                trainer.save_model(output)

            exported = torch.load(
                output / "flow_model" / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            expected = model.state_dict()
            self.assertEqual(exported.keys(), expected.keys())
            for name in expected:
                torch.testing.assert_close(exported[name], expected[name])

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
        self.assertTrue(counts.globally_reduced)
        self.assertEqual(counts.world_size, 1)

    def test_accumulation_normalization_uses_alignment_tokens_and_one_collective(self):
        batch = {
            "latents": torch.zeros(1, 2, 4),
            "latent_mask": torch.tensor([[True, True, False, False]]),
            "conditioning_ids": torch.ones(1, 4, dtype=torch.long),
            "conditioning_mask": torch.tensor([[True, True, True, False]]),
            "alignment_mask": torch.tensor([[False, True, False, False]]),
        }

        def add_remote_counts(totals, op):
            self.assertIs(op, torch.distributed.ReduceOp.SUM)
            totals.add_(torch.tensor([8.0, 2.0, 3.0, 4.0]))

        with (
            patch("nar_vae.training_data.dist.is_available", return_value=True),
            patch("nar_vae.training_data.dist.is_initialized", return_value=True),
            patch("nar_vae.training_data.dist.get_world_size", return_value=2),
            patch(
                "nar_vae.training_data.dist.all_reduce",
                side_effect=add_remote_counts,
            ) as all_reduce,
        ):
            counts = build_accumulation_loss_normalization([batch], torch.device("cpu"))

        all_reduce.assert_called_once()
        torch.testing.assert_close(
            torch.stack(
                (
                    counts.velocity_elements,
                    counts.examples,
                    counts.text_tokens,
                    counts.alignment_frames,
                )
            ),
            torch.tensor([12.0, 3.0, 4.0, 6.0]),
        )
        self.assertTrue(counts.globally_reduced)
        self.assertEqual(counts.world_size, 2)

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
                trainer_config = dict(config)
                model = nn.Linear(1, 1)
                if trainer_type is EchoDiTTrainer:
                    trainer_config.update(
                        text_conditioning_mode="frozen_features",
                        generative_objective="vp_diffusion_v",
                    )
                    model = PretrainingContractLinear(1, 1)
                trainer = trainer_type(
                    **{
                        config_key: trainer_config,
                        "model": model,
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

    def test_pretraining_trainer_rejects_config_or_unwrapped_model_contract_mismatch(self):
        with TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                per_device_train_batch_size=1,
                remove_unused_columns=False,
                report_to=[],
            )
            valid_config = {
                "text_conditioning_mode": "frozen_features",
                "generative_objective": "vp_diffusion_v",
            }
            valid_model = PretrainingContractLinear(1, 1)
            rectified_model = PretrainingContractLinear(1, 1)
            rectified_model.generative_objective = "rectified_flow"
            cases = (
                (
                    {**valid_config, "text_conditioning_mode": "scratch_tokens"},
                    valid_model,
                    "requires text_conditioning_mode: frozen_features",
                ),
                (
                    {**valid_config, "generative_objective": "rectified_flow"},
                    valid_model,
                    "requires generative_objective: vp_diffusion_v",
                ),
                (
                    valid_config,
                    TrainingModelWrapper(nn.Linear(1, 1)),
                    "cannot relabel a scratch-token model",
                ),
                (
                    valid_config,
                    TrainingModelWrapper(rectified_model),
                    "cannot relabel a rectified-flow model",
                ),
            )

            for config, model, message in cases:
                with (
                    self.subTest(message=message),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    EchoDiTTrainer(
                        training_config=config,
                        model=model,
                        args=arguments,
                        train_dataset=MetadataDataset([1]),
                        data_collator=collate_ids,
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

    def test_padded_attention_budget_accounts_for_text_and_reference_padding(self):
        sampler = PaddedAttentionBatchSampler(
            [8, 8],
            text_lengths=[2, 20],
            speaker_frame_lengths=[4, 40],
            max_attention_cost=1_500,
            speaker_patch_size=2,
            max_frames=100,
            max_examples=2,
            shuffle=False,
        )

        batches = list(sampler)

        self.assertEqual(batches, [[0], [1]])
        self.assertTrue(all(sampler.batch_cost(batch) <= 1_500 for batch in batches))
        patched = PaddedAttentionBatchSampler(
            [8],
            text_lengths=[2],
            speaker_frame_lengths=[4],
            max_attention_cost=1_500,
            speaker_patch_size=2,
            target_patch_size=2,
            max_frames=100,
            max_examples=1,
            shuffle=False,
        )
        self.assertLess(patched.batch_cost((0,)), sampler.batch_cost((0,)))

    def test_builder_selects_padded_attention_mode_from_training_config(self):
        trainer, _, _ = self._trainer([8, 8])
        dataset = AttentionMetadataDataset([8, 8], [2, 20], [4, 40])
        trainer.train_dataset = dataset
        trainer.training_config = {
            "batching_cost": "padded_attention",
            "max_attention_cost_per_batch": 1_500,
            "speaker_patch_size": 2,
            "use_speaker_conditioning": True,
        }
        settings = resolve_frame_budget_batching(
            {
                "batch_size": 2,
                "max_frames_per_batch": 100,
            }
        )

        dataloader = build_frame_budget_train_dataloader(trainer, settings)

        self.assertIsInstance(trainer._frame_budget_batch_sampler, PaddedAttentionBatchSampler)
        self.assertEqual(sorted(tuple(batch) for batch in dataloader), [(0,), (1,)])

    def test_padded_attention_batches_are_balanced_in_ddp_step_groups(self):
        sampler = PaddedAttentionBatchSampler(
            [8, 7, 6, 5, 4, 3, 2, 1],
            text_lengths=[1] * 8,
            speaker_frame_lengths=[0] * 8,
            max_attention_cost=1_000,
            speaker_patch_size=2,
            step_world_size=2,
            max_frames=10,
            max_examples=1,
            bucket_size=8,
            seed=31,
            shuffle=True,
        )

        batches = list(sampler)
        cost_order = sorted(
            range(8),
            key=lambda index: sampler.batch_cost((index,)),
            reverse=True,
        )
        expected_groups = {frozenset(cost_order[start : start + 2]) for start in range(0, 8, 2)}
        actual_groups = {
            frozenset(batch[0] for batch in batches[start : start + 2]) for start in range(0, 8, 2)
        }

        self.assertEqual(actual_groups, expected_groups)

    def test_epoch_aware_sampler_refreshes_dynamic_reference_metadata(self):
        class DynamicReferences:
            def __init__(self):
                self.epoch = 0

            def set_epoch(self, epoch):
                self.epoch = epoch

            def metadata(self):
                return [2, 2], [4 + self.epoch, 6 + self.epoch]

        dataset = DynamicReferences()
        sampler = PaddedAttentionBatchSampler(
            [4, 4],
            text_lengths=[2, 2],
            speaker_frame_lengths=[4, 6],
            max_attention_cost=1_000,
            speaker_patch_size=2,
            metadata_provider=dataset.metadata,
            epoch_target=dataset,
            max_frames=10,
            max_examples=2,
        )

        sampler.set_epoch(3)

        self.assertEqual(dataset.epoch, 3)
        self.assertEqual(sampler.speaker_frame_lengths, (7, 9))
        self.assertIsInstance(sampler, EpochAwareFrameBudgetBatchSampler)

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
