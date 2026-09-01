"""CPU-only tests for architecture-aware Muon training support."""

from __future__ import annotations

import copy
import io
import unittest
from collections.abc import Mapping, Sequence
from unittest.mock import patch

import torch
import torch.nn as nn

from nar_vae.finetune import EchoDiTFineTuner
from nar_vae.model_presets import get_model_preset
from nar_vae.models.flow_matching import create_flow_matching_echodit
from nar_vae.train import EchoDiTTrainer
from nar_vae.training_optimizers import build_muon_optimizer, partition_muon_parameters


def _conditioned_preset_model(name: str):
    """Construct a real packaged topology without allocating its parameter storage."""
    with torch.device("meta"):
        return create_flow_matching_echodit(
            latent_size=128,
            text_vocab_size=530,
            speaker_patch_size=4,
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=("en", "tr"),
            supported_language_pairs=(("en", "en"), ("tr", "tr"), ("tr", "en")),
            use_duration_predictor=True,
            duration_predictor_use_speaker=True,
            use_mas_duration=True,
            **get_model_preset(name).model_kwargs(),
        )


def _small_cpu_model():
    """Return a cheap EchoDiT that retains every optimizer-relevant component."""
    return create_flow_matching_echodit(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=16,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        timestep_embed_size=8,
        adaln_rank=4,
        cfg_dropout=0.0,
        use_speaker_conditioning=True,
        use_language_conditioning=True,
        supported_languages=("en", "tr"),
        supported_language_pairs=(("en", "en"), ("tr", "en")),
        use_duration_predictor=True,
        duration_predictor_hidden_size=8,
        duration_predictor_num_layers=1,
        duration_predictor_use_speaker=True,
        use_mas_duration=True,
        duration_alignment_hidden_size=4,
    )


def _muon_config() -> dict[str, object]:
    return {
        "optimizer": "muon",
        "learning_rate": 2e-3,
        "weight_decay": 0.01,
        "muon_learning_rate": 2e-3,
        "muon_weight_decay": 0.02,
        "muon_momentum": 0.9,
        "muon_nesterov": True,
        "muon_ns_steps": 2,
        "muon_epsilon": 1e-7,
        "muon_adjust_lr_fn": "match_rms_adamw",
        "adam_beta1": 0.8,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
    }


def _selected_optimizer_names(model: nn.Module) -> tuple[str, str, str]:
    partition = partition_muon_parameters(model)
    return (
        partition.muon_names[0],
        partition.adamw_decay_names[0],
        partition.adamw_no_decay_names[0],
    )


def _set_gradients(model: nn.Module, names: Sequence[str], *, offset: float) -> None:
    parameters = dict(model.named_parameters())
    for parameter in model.parameters():
        parameter.grad = None
    for index, name in enumerate(names):
        parameter = parameters[name]
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        gradient = ((values.remainder(11) - 5.0) * 0.01) + offset + index * 0.005
        parameter.grad = gradient.to(dtype=parameter.dtype)


class MuonParameterRoutingTest(unittest.TestCase):
    def _assert_complete_disjoint_partition(self, model, partition) -> None:
        groups = (partition.muon, partition.adamw_decay, partition.adamw_no_decay)
        routed_ids = [id(parameter) for group in groups for parameter in group]
        trainable_ids = [
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        ]
        routed_names = (
            partition.muon_names + partition.adamw_decay_names + partition.adamw_no_decay_names
        )
        trainable_names = tuple(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )

        self.assertEqual(len(routed_ids), len(set(routed_ids)))
        self.assertEqual(set(routed_ids), set(trainable_ids))
        self.assertEqual(len(routed_names), len(set(routed_names)))
        self.assertEqual(set(routed_names), set(trainable_names))

        name_order = {name: index for index, name in enumerate(trainable_names)}
        for names in (
            partition.muon_names,
            partition.adamw_decay_names,
            partition.adamw_no_decay_names,
        ):
            indices = [name_order[name] for name in names]
            self.assertEqual(indices, sorted(indices))

    def test_nano_and_tiny_route_only_hidden_linear_weights_through_muon(self):
        expected_muon = {
            "dit.text_encoder.blocks.0.attention.wq.weight",
            "dit.speaker_encoder.blocks.0.mlp.w1.weight",
            "dit.cond_module.2.weight",
            "dit.blocks.0.attention.wq.weight",
            "dit.blocks.0.attention_adaln.shift_down.weight",
            "duration_predictor.blocks.0.in_projection.weight",
        }
        expected_auxiliary_decay = {
            "dit.text_encoder.text_embedding.weight",
            "dit.text_encoder.language_embedding.weight",
            "dit.speaker_encoder.in_proj.weight",
            "dit.cond_module.0.weight",
            "dit.cond_module.4.weight",
            "dit.in_proj.weight",
            "dit.out_proj.weight",
            "duration_predictor.output_projection.weight",
            "duration_alignment.text_statistics.weight",
        }
        expected_auxiliary_no_decay = {
            "dit.text_encoder.blocks.0.attention.q_norm.weight",
            "dit.blocks.0.attention.q_norm.weight",
            "dit.text_norm.weight",
            "dit.in_proj.bias",
            "dit.out_proj.bias",
        }

        for preset in ("nano", "tiny"):
            with self.subTest(preset=preset):
                model = _conditioned_preset_model(preset)
                partition = partition_muon_parameters(model)
                modules = dict(model.named_modules())
                parameters = dict(model.named_parameters())

                self.assertTrue(expected_muon.issubset(partition.muon_names))
                self.assertTrue(expected_auxiliary_decay.issubset(partition.adamw_decay_names))
                self.assertTrue(
                    expected_auxiliary_no_decay.issubset(partition.adamw_no_decay_names)
                )
                for name in partition.muon_names:
                    owner_name, _, parameter_name = name.rpartition(".")
                    self.assertEqual(parameter_name, "weight")
                    self.assertIsInstance(modules[owner_name], nn.Linear)
                    self.assertEqual(parameters[name].ndim, 2)

                self._assert_complete_disjoint_partition(model, partition)
                repeated = partition_muon_parameters(model)
                self.assertEqual(repeated.muon_names, partition.muon_names)
                self.assertEqual(repeated.adamw_decay_names, partition.adamw_decay_names)
                self.assertEqual(repeated.adamw_no_decay_names, partition.adamw_no_decay_names)

                frozen_names = (
                    "dit.blocks.0.attention.wq.weight",
                    "dit.text_encoder.text_embedding.weight",
                )
                for name in frozen_names:
                    parameters[name].requires_grad_(False)
                frozen_partition = partition_muon_parameters(model)
                self._assert_complete_disjoint_partition(model, frozen_partition)
                for name in frozen_names:
                    self.assertNotIn(name, frozen_partition.trainable_names)

                same_topology = _conditioned_preset_model(preset)
                same_parameters = dict(same_topology.named_parameters())
                for name in frozen_names:
                    same_parameters[name].requires_grad_(False)
                same_partition = partition_muon_parameters(same_topology)
                self.assertEqual(same_partition.muon_names, frozen_partition.muon_names)
                self.assertEqual(
                    same_partition.adamw_decay_names,
                    frozen_partition.adamw_decay_names,
                )
                self.assertEqual(
                    same_partition.adamw_no_decay_names,
                    frozen_partition.adamw_no_decay_names,
                )


@unittest.skipUnless(hasattr(torch.optim, "Muon"), "PyTorch 2.9+ Muon is required")
class HybridMuonAdamWTest(unittest.TestCase):
    def test_pretrain_and_sft_trainers_select_the_shared_muon_builder(self):
        for trainer_class in (EchoDiTTrainer, EchoDiTFineTuner):
            with self.subTest(trainer=trainer_class.__name__):
                trainer = object.__new__(trainer_class)
                trainer.optimizer = None
                trainer.model = nn.Linear(2, 2)
                trainer.training_config = {"optimizer": "muon"}
                expected = object()
                with patch(
                    "nar_vae.training_optimizers.build_muon_optimizer",
                    return_value=expected,
                ) as builder:
                    self.assertIs(trainer.create_optimizer(), expected)
                builder.assert_called_once_with(trainer.model, trainer.training_config)

    def test_cpu_step_is_finite_and_updates_muon_and_adamw_branches(self):
        torch.manual_seed(7)
        model = _small_cpu_model()
        optimizer = build_muon_optimizer(model, _muon_config())
        selected_names = _selected_optimizer_names(model)
        parameters = dict(model.named_parameters())
        before = {name: parameters[name].detach().clone() for name in selected_names}

        _set_gradients(model, selected_names, offset=0.03)
        optimizer.step()

        self.assertEqual(
            [group["optimizer_role"] for group in optimizer.param_groups],
            ["muon", "adamw_decay", "adamw_no_decay"],
        )
        for name in selected_names:
            self.assertTrue(torch.isfinite(parameters[name]).all(), name)
            self.assertFalse(torch.equal(parameters[name], before[name]), name)

    def test_scheduler_changes_every_live_optimizer_group_learning_rate(self):
        torch.manual_seed(11)
        model = _small_cpu_model()
        optimizer = build_muon_optimizer(model, _muon_config())
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 / (step + 1),
        )
        initial = [float(group["lr"]) for group in optimizer.param_groups]

        _set_gradients(model, _selected_optimizer_names(model), offset=0.02)
        optimizer.step()
        scheduler.step()

        updated = [float(group["lr"]) for group in optimizer.param_groups]
        self.assertEqual(len(updated), 3)
        for before, after in zip(initial, updated):
            self.assertAlmostEqual(after, before * 0.5)
        self.assertEqual(updated, scheduler.get_last_lr())
        self.assertEqual(
            optimizer.muon_optimizer.param_groups[0]["lr"],
            updated[0],
        )
        self.assertEqual(
            [group["lr"] for group in optimizer.adamw_optimizer.param_groups],
            updated[1:],
        )

    def test_state_dict_round_trip_preserves_the_exact_next_step(self):
        torch.manual_seed(19)
        uninterrupted = _small_cpu_model()
        torch.manual_seed(19)
        resumed = _small_cpu_model()
        uninterrupted_optimizer = build_muon_optimizer(uninterrupted, _muon_config())
        resumed_optimizer = build_muon_optimizer(resumed, _muon_config())
        selected_names = _selected_optimizer_names(uninterrupted)

        _set_gradients(uninterrupted, selected_names, offset=0.01)
        uninterrupted_optimizer.step()
        resumed.load_state_dict(copy.deepcopy(uninterrupted.state_dict()), strict=True)

        buffer = io.BytesIO()
        torch.save(uninterrupted_optimizer.state_dict(), buffer)
        buffer.seek(0)
        optimizer_state = torch.load(buffer, map_location="cpu", weights_only=True)
        self.assertEqual(optimizer_state["hybrid_muon_adamw_version"], 1)
        self.assertIsNotNone(optimizer_state["adamw"])
        resumed_optimizer.load_state_dict(optimizer_state)
        self._assert_nested_equal(
            uninterrupted_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
        )

        _set_gradients(uninterrupted, selected_names, offset=0.04)
        _set_gradients(resumed, selected_names, offset=0.04)
        uninterrupted_optimizer.step()
        resumed_optimizer.step()

        uninterrupted_state = uninterrupted.state_dict()
        resumed_state = resumed.state_dict()
        self.assertEqual(set(uninterrupted_state), set(resumed_state))
        for name in uninterrupted_state:
            torch.testing.assert_close(
                uninterrupted_state[name],
                resumed_state[name],
                rtol=0,
                atol=0,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )
        self._assert_nested_equal(
            uninterrupted_optimizer.state_dict(),
            resumed_optimizer.state_dict(),
        )

    def _assert_nested_equal(self, left, right) -> None:
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            return
        if isinstance(left, Mapping):
            self.assertIsInstance(right, Mapping)
            self.assertEqual(set(left), set(right))
            for key in left:
                self._assert_nested_equal(left[key], right[key])
            return
        if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
            self.assertIsInstance(right, Sequence)
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self._assert_nested_equal(left_item, right_item)
            return
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
