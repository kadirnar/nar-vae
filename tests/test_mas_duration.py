"""CPU tests for versioned MAS duration training and scratch adaLN-Zero."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nar_vae.checkpoint import (
    CheckpointProvenance,
    DurationCheckpointInfo,
    FlowCheckpoint,
    LanguageCheckpointInfo,
    LegacyMonotonicAlignmentCheckpointError,
    MonotonicAlignmentCheckpointInfo,
    ReferenceLanguageCheckpointInfo,
    inspect_monotonic_alignment_capability,
    load_pretrained_checkpoint,
)
from nar_vae.dacvae import HubDACVAESource
from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.losses.flow_matching_loss import FlowMatchingLoss, _global_valid_mean
from nar_vae.model_manifest import text_conditioning_from_config
from nar_vae.models.dit import LowRankAdaLN
from nar_vae.models.duration import (
    DurationAlignmentOutput,
    EchoDurationAlignment,
    allocate_positive_token_durations,
    expand_text_by_durations,
)
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.solvers.ode_solver import ODESolver


def tiny_model(*, mas: bool) -> FlowMatchingEchoDiT:
    return FlowMatchingEchoDiT(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=20,
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
        cfg_dropout_text=0.0,
        cfg_dropout_speaker=0.0,
        use_duration_predictor=True,
        duration_predictor_hidden_size=6,
        duration_predictor_num_layers=1,
        use_mas_duration=mas,
        duration_alignment_hidden_size=3,
    )


class DurationAlignmentHeadTest(unittest.TestCase):
    def test_clean_latents_are_fixed_targets_but_text_prior_receives_gradients(self):
        torch.manual_seed(3)
        head = EchoDurationAlignment(text_size=5, latent_size=4, hidden_size=3)
        text_state = torch.randn(2, 3, 5, requires_grad=True)
        clean_latents = torch.randn(2, 4, 6, requires_grad=True)

        likelihoods = head(text_state, clean_latents)
        self.assertEqual(likelihoods.shape, (2, 3, 6))
        self.assertTrue(torch.isfinite(likelihoods).all())
        (-likelihoods.mean()).backward()

        self.assertIsNotNone(text_state.grad)
        self.assertIsNotNone(head.text_statistics.weight.grad)
        self.assertIsNone(clean_latents.grad)

    def test_positive_integer_allocation_exactly_matches_requested_total(self):
        expected = torch.tensor([[1.0, 2.0, 7.0, 999.0], [0.0, 0.0, 0.0, 999.0]])
        mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
        allocated = allocate_positive_token_durations(
            expected,
            torch.tensor([13, 8]),
            mask,
        )

        torch.testing.assert_close(allocated.sum(dim=1), torch.tensor([13, 8]))
        self.assertTrue((allocated[mask] >= 1).all())
        self.assertTrue((allocated[~mask] == 0).all())
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            allocate_positive_token_durations(expected[:1], 2, mask[:1])

    def test_positive_integer_allocation_preserves_exact_integral_predictions(self):
        expected = torch.tensor(
            [
                [1.0, 9.0, 999.0],
                [1.0, 1.0, 8.0],
            ]
        )
        mask = torch.tensor(
            [
                [True, True, False],
                [True, True, True],
            ]
        )

        allocated = allocate_positive_token_durations(expected, 10, mask)

        torch.testing.assert_close(
            allocated,
            torch.tensor([[1, 9, 0], [1, 1, 8]]),
        )

    def test_duration_expansion_maps_each_frame_to_its_token(self):
        text_state = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
        expanded = expand_text_by_durations(
            text_state,
            torch.tensor([[2, 1, 3]]),
            target_frames=6,
        )

        torch.testing.assert_close(
            expanded,
            torch.tensor(
                [[[1.0, 10.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [3.0, 30.0], [3.0, 30.0]]]
            ),
        )
        with self.assertRaisesRegex(ValueError, "sum exactly"):
            expand_text_by_durations(text_state, torch.tensor([[2, 1, 2]]), target_frames=6)


class MASFlowTrainingTest(unittest.TestCase):
    def test_one_invariant_encoder_batch_serves_duration_mas_cfg_and_trajectory(self):
        model = FlowMatchingEchoDiT(
            latent_size=4,
            model_size=8,
            num_layers=1,
            num_heads=2,
            intermediate_size=16,
            text_vocab_size=20,
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
            use_speaker_conditioning=True,
            use_duration_predictor=True,
            duration_predictor_hidden_size=6,
            duration_predictor_num_layers=1,
            duration_predictor_use_speaker=True,
            use_mas_duration=True,
            duration_alignment_hidden_size=3,
        ).eval()
        conditioning_ids = torch.tensor([[1, 2, 3]])
        speaker_latent = torch.randn(1, 4, 4)

        with (
            patch.object(model.dit, "encode_text", wraps=model.dit.encode_text) as encode_text,
            patch.object(
                model.dit,
                "encode_speaker",
                wraps=model.dit.encode_speaker,
            ) as encode_speaker,
        ):
            encoded = model.encode_inference_conditioning(
                conditioning_ids,
                speaker_latent=speaker_latent,
                cfg_mode="alternating",
            )
            _, token_weights = model.predict_duration_frames_and_token_weights(encoded)
            assert token_weights is not None
            token_durations = model.allocate_token_duration_frames(
                token_weights,
                total_frames=6,
            )
            prepared, prepared_cfg = model.finalize_inference_conditioning(
                encoded,
                token_durations=token_durations,
            )
            ODESolver.sample(
                model=model,
                conditioning_ids=conditioning_ids,
                token_durations=token_durations,
                prepared_conditioning=prepared,
                prepared_cfg_conditioning=prepared_cfg,
                num_steps=2,
                latent_shape=(1, 4, 6),
                solver="heun",
                cfg_scale=2.0,
                cfg_mode="alternating",
                speaker_latent=speaker_latent,
                device=torch.device("cpu"),
            )

        self.assertEqual(encode_text.call_count, 1)
        self.assertEqual(encode_speaker.call_count, 1)
        self.assertEqual(token_durations.sum().item(), 6)

    def test_model_returns_versioned_alignment_output_and_exact_inference_allocation(self):
        torch.manual_seed(5)
        model = tiny_model(mas=True).eval()
        token_mask = torch.tensor([[True, True, True, False]])
        velocity, prediction = model(
            latents=torch.randn(1, 4, 6),
            conditioning_ids=torch.tensor([[1, 2, 3, 0]]),
            timesteps=torch.tensor([0.5]),
            attention_mask=token_mask,
            use_cfg_dropout=False,
            return_duration_prediction=True,
            return_duration_alignment=True,
            duration_target_latents=torch.randn(1, 4, 6),
        )

        self.assertEqual(velocity.shape, (1, 4, 6))
        self.assertIsInstance(prediction, DurationAlignmentOutput)
        self.assertEqual(prediction.token_durations.shape, token_mask.shape)
        self.assertEqual(prediction.log_likelihoods.shape, (1, 4, 6))
        self.assertEqual(prediction.hard_alignment.shape, (1, 4, 6))
        self.assertEqual(prediction.hard_alignment.dtype, torch.bool)
        allocated = model.predict_token_duration_frames(
            torch.tensor([[1, 2, 3, 0]]),
            token_mask,
            total_frames=11,
        )
        self.assertEqual(allocated.sum().item(), 11)
        self.assertTrue((allocated[token_mask] >= 1).all())
        self.assertEqual(allocated[~token_mask].count_nonzero().item(), 0)

    def test_inference_duration_allocation_changes_frame_conditioning_and_velocity(self):
        torch.manual_seed(17)
        model = tiny_model(mas=True).eval()
        with torch.no_grad():
            model.dit.out_proj.weight.normal_(std=0.2)
            model.dit.frame_text_proj.weight.normal_(std=0.2)
        conditioning_ids = torch.tensor([[1, 2]])
        early_switch = model.prepare_inference_conditioning(
            conditioning_ids,
            token_durations=torch.tensor([[1, 5]]),
        )
        late_switch = model.prepare_inference_conditioning(
            conditioning_ids,
            token_durations=torch.tensor([[5, 1]]),
        )
        latents = torch.randn(1, 4, 6)
        timesteps = torch.tensor([0.5])

        early_velocity = model.forward_prepared(latents, timesteps, early_switch)
        late_velocity = model.forward_prepared(latents, timesteps, late_switch)

        self.assertFalse(torch.allclose(early_velocity, late_velocity))

    def test_joint_flow_mas_objective_backpropagates_to_both_duration_heads(self):
        torch.manual_seed(7)
        model = tiny_model(mas=True).train()
        with torch.no_grad():
            model.dit.out_proj.weight.normal_(std=0.1)
        objective = FlowMatchingLoss(
            timestep_distribution="uniform",
            duration_loss_weight=0.1,
            mas_duration_loss_weight=0.2,
            mas_alignment_loss_weight=0.03,
        )
        loss = objective(
            model,
            torch.randn(2, 4, 6),
            torch.tensor([[1, 2, 3], [2, 3, 4]]),
            conditioning_mask=torch.ones(2, 3, dtype=torch.bool),
            latent_mask=torch.tensor(
                [[True, True, True, True, True, True], [True, True, True, True, False, False]]
            ),
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.duration_predictor.output_projection.weight.grad)
        self.assertIsNotNone(model.duration_alignment.text_statistics.weight.grad)
        self.assertGreater(
            model.duration_alignment.text_statistics.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertIsNotNone(model.dit.frame_text_proj.weight.grad)
        self.assertGreater(model.dit.frame_text_proj.weight.grad.abs().sum().item(), 0.0)

    def test_mas_rejects_more_valid_tokens_than_frames(self):
        model = tiny_model(mas=True).train()
        objective = FlowMatchingLoss(
            timestep_distribution="uniform",
            mas_duration_loss_weight=0.1,
            mas_alignment_loss_weight=0.01,
        )
        with self.assertRaisesRegex(ValueError, "at least one frame per token"):
            objective(
                model,
                torch.randn(1, 4, 2),
                torch.tensor([[1, 2, 3]]),
            )

    def test_global_valid_mean_uses_global_count_and_ddp_gradient_scale(self):
        numerator = torch.tensor(6.0, requires_grad=True)

        def add_remote_rank(statistics, op):
            del op
            if statistics.numel() == 2:
                statistics.add_(torch.tensor([4.0, 8.0]))

        with (
            patch("nar_vae.losses.flow_matching_loss.dist.is_available", return_value=True),
            patch("nar_vae.losses.flow_matching_loss.dist.is_initialized", return_value=True),
            patch("nar_vae.losses.flow_matching_loss.dist.get_world_size", return_value=2),
            patch(
                "nar_vae.losses.flow_matching_loss.dist.all_reduce",
                side_effect=add_remote_rank,
            ),
        ):
            result = _global_valid_mean(numerator, torch.tensor(2))
        torch.testing.assert_close(result, torch.tensor(1.0))
        result.backward()
        torch.testing.assert_close(numerator.grad, torch.tensor(0.2))

    def test_pre_reduced_window_count_avoids_per_objective_collectives(self):
        numerator = torch.tensor(6.0, requires_grad=True)
        with (
            patch("nar_vae.losses.flow_matching_loss.dist.is_available", return_value=True),
            patch("nar_vae.losses.flow_matching_loss.dist.is_initialized", return_value=True),
            patch("nar_vae.losses.flow_matching_loss.dist.get_world_size", return_value=2),
            patch("nar_vae.losses.flow_matching_loss.dist.all_reduce") as all_reduce,
        ):
            result = _global_valid_mean(
                numerator,
                torch.tensor(2),
                accumulation_count=torch.tensor(10),
                accumulation_is_global=True,
                normalization_world_size=2,
            )

        all_reduce.assert_not_called()
        torch.testing.assert_close(result, torch.tensor(1.2))
        result.backward()
        torch.testing.assert_close(numerator.grad, torch.tensor(0.2))

    def test_accumulation_window_denominator_weights_ragged_microbatches_exactly(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        first = _global_valid_mean(
            parameter * 2,
            torch.tensor(2),
            accumulation_count=torch.tensor(10),
        )
        second = _global_valid_mean(
            parameter * 8,
            torch.tensor(8),
            accumulation_count=torch.tensor(10),
        )

        torch.testing.assert_close(first + second, parameter)
        (first + second).backward()
        torch.testing.assert_close(parameter.grad, torch.tensor(1.0))


class MASCheckpointTest(unittest.TestCase):
    def test_inference_reconstructs_the_mas_head_from_checkpoint_capability(self):
        checkpoint_path = Path("mas.bin")
        checkpoint = Mock()
        checkpoint.path = checkpoint_path
        checkpoint.provenance = CheckpointProvenance(
            kind="local",
            source=str(checkpoint_path),
            requested_revision=None,
            resolved_revision=None,
            base_filename=checkpoint_path.name,
            ema_filename=None,
            selected_filename=checkpoint_path.name,
            path=checkpoint_path,
            base_path=checkpoint_path,
        )
        checkpoint.infer_text_vocab_size.return_value = 20
        checkpoint.infer_speaker_conditioning.return_value = False
        checkpoint.language_capability.return_value = LanguageCheckpointInfo(False)
        checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(
            False
        )
        checkpoint.duration_capability.return_value = DurationCheckpointInfo(True, 6, 1, False)
        checkpoint.monotonic_alignment_capability.return_value = MonotonicAlignmentCheckpointInfo(
            True, 3, 1
        )
        model = tiny_model(mas=True)
        codec = SimpleNamespace(
            sample_rate=16,
            hop_length=4,
            decode=lambda latent: latent,
        )
        codec_id = "facebook/dacvae-watermarked"
        codec_revision = "a" * 40

        with (
            patch("nar_vae.inference.FlowCheckpoint.load", return_value=checkpoint),
            patch(
                "nar_vae.inference.create_flow_matching_echodit",
                return_value=model,
            ) as factory,
            patch(
                "nar_vae.inference.load_model_manifest",
                return_value=SimpleNamespace(
                    text_conditioning=text_conditioning_from_config({}),
                    representation={
                        "codec_source": codec_id,
                        "codec_revision": codec_revision,
                        "codec_filename": "weights.pth",
                        "codec_sha256": "a" * 64,
                    },
                ),
            ),
            patch("nar_vae.inference.validate_inference_manifest") as validate_manifest,
            patch("nar_vae.inference.load_dacvae", return_value=codec) as load_codec,
            patch("nar_vae.inference.validate_loaded_codec"),
        ):
            runtime = FlowMatchingTTSInference(
                checkpoint_path,
                dacvae_model=codec_id,
                device="cpu",
            )

        self.assertTrue(runtime.uses_mas_duration)
        self.assertTrue(factory.call_args.kwargs["use_mas_duration"])
        self.assertEqual(factory.call_args.kwargs["duration_alignment_hidden_size"], 3)
        checkpoint.load_into.assert_called_once_with(model)
        resolved_source = load_codec.call_args.args[0]
        self.assertIsInstance(resolved_source, HubDACVAESource)
        self.assertEqual(resolved_source.repo_id, codec_id)
        self.assertEqual(resolved_source.revision, codec_revision)
        self.assertEqual(resolved_source.filename, "weights.pth")
        self.assertIs(validate_manifest.call_args.kwargs["codec_source"], resolved_source)

    def test_capability_is_inferred_from_complete_checkpoint_state(self):
        model = tiny_model(mas=True)
        capability = inspect_monotonic_alignment_capability(model.state_dict())
        self.assertEqual(capability, MonotonicAlignmentCheckpointInfo(True, 3, 1))
        self.assertFalse(
            inspect_monotonic_alignment_capability(tiny_model(mas=False).state_dict()).enabled
        )

        partial = dict(model.state_dict())
        partial.pop("duration_alignment_version")
        with self.assertRaisesRegex(LegacyMonotonicAlignmentCheckpointError, "complete"):
            inspect_monotonic_alignment_capability(partial)

        missing_regulator = dict(model.state_dict())
        missing_regulator.pop("dit.frame_text_proj.weight")
        with self.assertRaisesRegex(LegacyMonotonicAlignmentCheckpointError, "frame-regulator"):
            inspect_monotonic_alignment_capability(missing_regulator)

        metadata_only = dict(tiny_model(mas=False).state_dict())
        metadata_only["duration_alignment_version"] = torch.tensor(1, dtype=torch.int32)
        metadata_only["duration_alignment_hidden_size_metadata"] = torch.tensor(
            3, dtype=torch.int32
        )
        with self.assertRaisesRegex(
            LegacyMonotonicAlignmentCheckpointError,
            "metadata is present without",
        ):
            inspect_monotonic_alignment_capability(metadata_only)

        unsupported = dict(model.state_dict())
        unsupported["duration_alignment_version"] = torch.tensor(99, dtype=torch.int32)
        with self.assertRaisesRegex(LegacyMonotonicAlignmentCheckpointError, "Unsupported"):
            inspect_monotonic_alignment_capability(unsupported)

    def test_sft_loading_requires_matching_parent_capability(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mas_path = directory / "mas.bin"
            legacy_path = directory / "duration-only.bin"
            mas_model = tiny_model(mas=True)
            duration_only = tiny_model(mas=False)
            torch.save(mas_model.state_dict(), mas_path)
            torch.save(duration_only.state_dict(), legacy_path)

            loaded = tiny_model(mas=True)
            result = load_pretrained_checkpoint(loaded, mas_path)
            self.assertEqual(result.missing_keys, [])
            self.assertEqual(result.unexpected_keys, [])
            with self.assertRaisesRegex(RuntimeError, "parent checkpoint does not contain"):
                load_pretrained_checkpoint(tiny_model(mas=True), legacy_path)
            with self.assertRaisesRegex(RuntimeError, "without that alignment capability"):
                load_pretrained_checkpoint(tiny_model(mas=False), mas_path)

    def test_partial_ema_capability_comes_from_full_base_state(self):
        model = tiny_model(mas=True)
        checkpoint = FlowCheckpoint(
            path=Path("pytorch_model_ema.bin"),
            state_dict={"dit.out_proj.weight": model.dit.out_proj.weight.detach().clone()},
            base_state_dict=model.state_dict(),
            is_ema=True,
        )
        self.assertEqual(
            checkpoint.monotonic_alignment_capability(),
            MonotonicAlignmentCheckpointInfo(True, 3, 1),
        )


class AdaLNZeroInitializationTest(unittest.TestCase):
    def test_scratch_modulation_is_zero_and_strict_load_overwrites_it(self):
        scratch = tiny_model(mas=False)
        self.assertEqual(scratch.dit.cond_module[-1].weight.count_nonzero().item(), 0)
        adaln_modules = [module for module in scratch.modules() if isinstance(module, LowRankAdaLN)]
        self.assertTrue(adaln_modules)
        for module in adaln_modules:
            for projection in (module.shift_up, module.scale_up, module.gate_up):
                self.assertEqual(projection.weight.count_nonzero().item(), 0)
                self.assertEqual(projection.bias.count_nonzero().item(), 0)

        source = tiny_model(mas=False)
        with torch.no_grad():
            source.dit.cond_module[-1].weight.fill_(0.25)
            for module in source.modules():
                if isinstance(module, LowRankAdaLN):
                    for projection in (module.shift_up, module.scale_up, module.gate_up):
                        projection.weight.fill_(0.5)
                        projection.bias.fill_(0.75)
        result = scratch.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        torch.testing.assert_close(
            scratch.dit.cond_module[-1].weight,
            source.dit.cond_module[-1].weight,
        )
        for target_module, source_module in zip(
            adaln_modules,
            (module for module in source.modules() if isinstance(module, LowRankAdaLN)),
        ):
            torch.testing.assert_close(target_module.shift_up.weight, source_module.shift_up.weight)
            torch.testing.assert_close(target_module.scale_up.bias, source_module.scale_up.bias)
            torch.testing.assert_close(target_module.gate_up.weight, source_module.gate_up.weight)


if __name__ == "__main__":
    unittest.main()
