"""Tests for EchoDiT checkpoint compatibility inference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nar_vae.checkpoint import (
    GLOBAL_LANGUAGE_EMBEDDING_KEY,
    LANGUAGE_EMBEDDING_KEY,
    FlowCheckpoint,
    GenerativeObjectiveCheckpointError,
    LegacyArchitectureCheckpointError,
    LegacyLanguageCheckpointError,
    LegacySpeakerCheckpointError,
    inspect_language_conditioning,
    inspect_speaker_num_summary_tokens,
    inspect_target_patch_size,
    load_pretrained_checkpoint,
    resolve_flow_checkpoint,
)
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.objectives import VP_DIFFUSION_OBJECTIVE
from nar_vae.voice import SPEAKER_CONDITIONING_VERSION, SPEAKER_PATCH_LAYOUT_VERSION


def _tiny_flow_model(
    *,
    use_speaker_conditioning: bool = False,
    use_language_conditioning: bool = False,
    speaker_num_summary_tokens: int = 0,
    generative_objective: str = "rectified_flow",
    diffusion_schedule_shift: float = 1.0,
) -> FlowMatchingEchoDiT:
    """Build the real topology at a CPU-test scale."""
    return FlowMatchingEchoDiT(
        latent_size=2,
        model_size=8,
        num_layers=2,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=32,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        target_patch_size=2,
        timestep_embed_size=8,
        adaln_rank=2,
        use_speaker_conditioning=use_speaker_conditioning,
        use_language_conditioning=use_language_conditioning,
        speaker_num_summary_tokens=speaker_num_summary_tokens,
        supported_languages=("en", "tr") if use_language_conditioning else None,
        generative_objective=generative_objective,
        diffusion_schedule_shift=diffusion_schedule_shift,
    )


def _tiny_fully_conditioned_model() -> FlowMatchingEchoDiT:
    """Build every immutable topology branch for EMA-overlay regression tests."""
    return FlowMatchingEchoDiT(
        latent_size=2,
        model_size=8,
        num_layers=2,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=32,
        text_model_size=8,
        text_num_layers=0,
        text_num_heads=2,
        text_intermediate_size=16,
        text_conditioning_mode="frozen_features",
        conditioning_feature_size=6,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        target_patch_size=2,
        timestep_embed_size=8,
        adaln_rank=2,
        use_speaker_conditioning=True,
        speaker_num_summary_tokens=3,
        use_language_conditioning=True,
        supported_languages=("en", "tr"),
        supported_language_pairs=(("en", "en"), ("tr", "tr")),
        use_duration_predictor=True,
        duration_predictor_hidden_size=8,
        duration_predictor_num_layers=1,
        duration_predictor_use_speaker=True,
        use_mas_duration=True,
        duration_alignment_hidden_size=2,
        generative_objective=VP_DIFFUSION_OBJECTIVE,
    )


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


class OrigModuleWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self._orig_mod = module


class FlowCheckpointTest(unittest.TestCase):
    def test_training_preload_validator_runs_immediately_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pytorch_model.bin"
            torch.save({"weight": torch.ones(1, 1)}, checkpoint_path)
            model = torch.nn.Linear(1, 1, bias=False)
            original_load = torch.load
            events = []

            def validate(path):
                events.append(("validate", path))

            def deserialize(*args, **kwargs):
                events.append(("deserialize", Path(args[0]).resolve()))
                return original_load(*args, **kwargs)

            with patch("nar_vae.checkpoint.torch.load", side_effect=deserialize):
                load_pretrained_checkpoint(
                    model,
                    checkpoint_path,
                    preload_validator=validate,
                )

            self.assertEqual(
                events,
                [
                    ("validate", checkpoint_path.resolve()),
                    ("deserialize", checkpoint_path.resolve()),
                ],
            )

    def test_training_loader_rejects_vp_schedule_mismatch_before_model_mutation(self):
        source = _tiny_flow_model(
            generative_objective=VP_DIFFUSION_OBJECTIVE,
            diffusion_schedule_shift=0.2,
        )
        target_core = _tiny_flow_model(
            generative_objective=VP_DIFFUSION_OBJECTIVE,
            diffusion_schedule_shift=1.0,
        )
        target = ModuleWrapper(OrigModuleWrapper(target_core))
        with torch.no_grad():
            target_core.dit.in_proj.weight.zero_()
        weight_before = target_core.dit.in_proj.weight.detach().clone()
        metadata_before = target_core.diffusion_schedule_shift_metadata.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pytorch_model.bin"
            torch.save(source.state_dict(), checkpoint_path)
            with self.assertRaisesRegex(
                GenerativeObjectiveCheckpointError,
                "objective/schedule does not match",
            ):
                load_pretrained_checkpoint(target, checkpoint_path)

        torch.testing.assert_close(target_core.dit.in_proj.weight, weight_before)
        torch.testing.assert_close(
            target_core.diffusion_schedule_shift_metadata,
            metadata_before,
        )
        self.assertEqual(target_core.diffusion_schedule_shift, 1.0)

    def test_flow_checkpoint_rejects_vp_schedule_mismatch_before_base_or_ema_load(self):
        source = _tiny_flow_model(
            generative_objective=VP_DIFFUSION_OBJECTIVE,
            diffusion_schedule_shift=0.2,
        )
        base_state = dict(source.state_dict())
        target_factories = (
            lambda: FlowCheckpoint(path=Path("base.bin"), state_dict=base_state),
            lambda: FlowCheckpoint(
                path=Path("pytorch_model_ema.bin"),
                state_dict={
                    "dit.in_proj.weight": torch.ones_like(base_state["dit.in_proj.weight"])
                },
                base_state_dict=base_state,
                is_ema=True,
            ),
        )
        for checkpoint_factory in target_factories:
            with self.subTest(ema=checkpoint_factory is target_factories[1]):
                target_core = _tiny_flow_model(
                    generative_objective=VP_DIFFUSION_OBJECTIVE,
                    diffusion_schedule_shift=1.0,
                )
                target = ModuleWrapper(OrigModuleWrapper(target_core))
                with torch.no_grad():
                    target_core.dit.in_proj.weight.zero_()
                weight_before = target_core.dit.in_proj.weight.detach().clone()
                metadata_before = target_core.diffusion_schedule_shift_metadata.detach().clone()

                with self.assertRaisesRegex(
                    GenerativeObjectiveCheckpointError,
                    "objective/schedule does not match",
                ):
                    checkpoint_factory().load_into(target)

                torch.testing.assert_close(target_core.dit.in_proj.weight, weight_before)
                torch.testing.assert_close(
                    target_core.diffusion_schedule_shift_metadata,
                    metadata_before,
                )

    def test_checkpoint_load_preserves_legacy_rectified_flow_contract(self):
        source = _tiny_flow_model()
        target = _tiny_flow_model()
        with torch.no_grad():
            source.dit.in_proj.weight.fill_(0.25)
            target.dit.in_proj.weight.zero_()

        FlowCheckpoint(path=Path("legacy.bin"), state_dict=source.state_dict()).load_into(target)

        torch.testing.assert_close(target.dit.in_proj.weight, source.dit.in_proj.weight)
        self.assertEqual(target.generative_objective, "rectified_flow")
        self.assertNotIn("generative_objective_metadata", target.state_dict())

    def test_infers_architecture_facts_from_local_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pytorch_model.bin"
            torch.save(
                {
                    "dit.text_encoder.text_embedding.weight": torch.empty(100287, 2),
                    "dit.speaker_encoder.in_proj.weight": torch.empty(2, 2),
                },
                checkpoint_path,
            )

            checkpoint = FlowCheckpoint.load(checkpoint_path)

            self.assertEqual(checkpoint.infer_text_vocab_size(1), 100287)
            self.assertFalse(checkpoint.infer_speaker_conditioning(True))

    def test_snapshot_directory_resolves_main_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            checkpoint_path = snapshot / "pytorch_model.bin"
            checkpoint_path.touch()

            self.assertEqual(resolve_flow_checkpoint(snapshot), checkpoint_path)

    def test_snapshot_directory_prefers_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            checkpoint_dir = snapshot
            base_path = checkpoint_dir / "pytorch_model.bin"
            ema_path = checkpoint_dir / "pytorch_model_ema.bin"
            base_path.touch()
            ema_path.touch()

            self.assertEqual(resolve_flow_checkpoint(snapshot), ema_path)
            self.assertEqual(resolve_flow_checkpoint(snapshot, prefer_ema=False), base_path)

    def test_partial_ema_requires_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            ema_path = Path(directory) / "pytorch_model_ema.bin"
            torch.save({"weight": torch.ones(1)}, ema_path)

            with self.assertRaisesRegex(FileNotFoundError, "requires its full base"):
                FlowCheckpoint.load(ema_path)

    def test_speaker_capability_requires_versioned_patch_layout(self):
        valid = FlowCheckpoint(
            path=Path("valid.bin"),
            state_dict={
                "null_speaker_embed": torch.zeros(1, 2, 4),
                "speaker_conditioning_version": torch.tensor(SPEAKER_CONDITIONING_VERSION),
                "speaker_patch_layout_version": torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                "speaker_patch_size_metadata": torch.tensor(4),
                "dit.speaker_encoder.in_proj.weight": torch.empty(2, 8),
            },
        )
        legacy = FlowCheckpoint(
            path=Path("legacy.bin"),
            state_dict={"null_speaker_embed": torch.zeros(1, 2, 4)},
        )

        self.assertTrue(valid.infer_speaker_conditioning(False))
        self.assertEqual(valid.infer_speaker_patch_size(1), 4)
        with self.assertRaisesRegex(LegacySpeakerCheckpointError, "ambiguous"):
            legacy.infer_speaker_conditioning(False)

    def test_speaker_metadata_can_be_ignored_by_a_disabled_model(self):
        model = torch.nn.Linear(1, 1, bias=False)
        checkpoint = FlowCheckpoint(
            path=Path("speaker.bin"),
            state_dict={
                "weight": torch.ones(1, 1),
                "null_speaker_embed": torch.zeros(1, 1, 1),
                "speaker_conditioning_version": torch.tensor(SPEAKER_CONDITIONING_VERSION),
                "speaker_patch_layout_version": torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                "speaker_patch_size_metadata": torch.tensor(1),
            },
        )

        checkpoint.load_into(model)

        self.assertEqual(model.weight.item(), 1.0)

    def test_wrapped_ema_overlays_base(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            base_path = checkpoint_dir / "pytorch_model.bin"
            ema_path = checkpoint_dir / "ema_model.bin"
            torch.save({"weight": torch.zeros(1, 1)}, base_path)
            torch.save({"shadow": {"weight": torch.ones(1, 1)}, "decay": 0.9}, ema_path)

            checkpoint = FlowCheckpoint.load(ema_path)
            model = torch.nn.Linear(1, 1, bias=False)
            checkpoint.load_into(model)

            self.assertEqual(model.weight.item(), 1.0)

    def test_partial_ema_cannot_override_authenticated_topology_metadata(self):
        base_model = _tiny_fully_conditioned_model()
        base_state = dict(base_model.state_dict())
        protected_keys = (
            "generative_objective_metadata",
            "diffusion_schedule_shift_metadata",
            "text_conditioning_feature_size_metadata",
            "echodit_architecture_version",
            "target_patch_size_metadata",
            "speaker_num_summary_tokens_metadata",
            "supported_language_pair_ids_metadata",
            "duration_predictor_num_layers_metadata",
            "duration_alignment_hidden_size_metadata",
        )

        for key in protected_keys:
            with self.subTest(key=key):
                target = _tiny_fully_conditioned_model()
                with torch.no_grad():
                    target.dit.in_proj.weight.zero_()
                weight_before = target.dit.in_proj.weight.detach().clone()
                tampered = base_state[key].clone() + 1
                checkpoint = FlowCheckpoint(
                    path=Path("pytorch_model_ema.bin"),
                    state_dict={key: tampered},
                    base_state_dict=base_state,
                    is_ema=True,
                )

                with self.assertRaisesRegex(RuntimeError, "immutable metadata"):
                    checkpoint.load_into(target)

                # The guard runs before even the authenticated base is applied.
                torch.testing.assert_close(target.dit.in_proj.weight, weight_before)

    def test_partial_ema_may_repeat_equal_metadata_and_overlay_parameters(self):
        base_model = _tiny_fully_conditioned_model()
        base_state = dict(base_model.state_dict())
        weight_key = "dit.in_proj.weight"
        replacement = torch.ones_like(base_state[weight_key])
        checkpoint = FlowCheckpoint(
            path=Path("pytorch_model_ema.bin"),
            state_dict={
                "speaker_num_summary_tokens_metadata": base_state[
                    "speaker_num_summary_tokens_metadata"
                ].clone(),
                weight_key: replacement,
            },
            base_state_dict=base_state,
            is_ema=True,
        )
        target = _tiny_fully_conditioned_model()

        checkpoint.load_into(target)

        torch.testing.assert_close(target.dit.in_proj.weight, replacement)
        self.assertEqual(int(target.speaker_num_summary_tokens_metadata.item()), 3)

    def test_training_loader_requires_explicit_speaker_initialization(self):
        class TinySpeakerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.dit = torch.nn.Module()
                self.dit.speaker_encoder = torch.nn.Module()
                self.dit.speaker_encoder.in_proj = torch.nn.Linear(4, 1, bias=False)
                self.register_buffer("null_speaker_embed", torch.zeros(1, 1, 4))
                self.null_speaker_state = torch.nn.Parameter(torch.zeros(1, 1, 1))
                self.register_buffer(
                    "speaker_conditioning_version",
                    torch.tensor(SPEAKER_CONDITIONING_VERSION),
                )
                self.register_buffer(
                    "speaker_patch_layout_version",
                    torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                )
                self.register_buffer("speaker_patch_size_metadata", torch.tensor(4))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "text-only.bin"
            torch.save(
                {
                    "module.weight": torch.ones(1),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(RuntimeError, "initialize_speaker_conditioning"):
                load_pretrained_checkpoint(TinySpeakerModel(), checkpoint_path)

            model = TinySpeakerModel()
            result = load_pretrained_checkpoint(
                model,
                checkpoint_path,
                initialize_speaker_conditioning=True,
            )

            self.assertEqual(
                set(result.missing_keys),
                {
                    "dit.speaker_encoder.in_proj.weight",
                    "null_speaker_embed",
                    "null_speaker_state",
                    "speaker_conditioning_version",
                    "speaker_patch_layout_version",
                    "speaker_patch_size_metadata",
                },
            )
            self.assertEqual(model.weight.item(), 1.0)

    def test_training_loader_rejects_unversioned_speaker_checkpoint(self):
        class TinySpeakerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.dit = torch.nn.Module()
                self.dit.speaker_encoder = torch.nn.Module()
                self.dit.speaker_encoder.in_proj = torch.nn.Linear(4, 1, bias=False)
                self.register_buffer("null_speaker_embed", torch.zeros(1, 1, 4))
                self.null_speaker_state = torch.nn.Parameter(torch.zeros(1, 1, 1))
                self.register_buffer(
                    "speaker_conditioning_version",
                    torch.tensor(SPEAKER_CONDITIONING_VERSION),
                )
                self.register_buffer(
                    "speaker_patch_layout_version",
                    torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                )
                self.register_buffer("speaker_patch_size_metadata", torch.tensor(4))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.bin"
            torch.save(
                {
                    "weight": torch.ones(1),
                    "null_speaker_embed": torch.zeros(1, 1, 4),
                },
                checkpoint_path,
            )

            with self.assertRaises(LegacySpeakerCheckpointError):
                load_pretrained_checkpoint(TinySpeakerModel(), checkpoint_path)

    def test_multilingual_checkpoint_requires_both_language_embedding_paths(self):
        state = dict(_tiny_flow_model(use_language_conditioning=True).state_dict())

        info = inspect_language_conditioning(state)

        self.assertTrue(info.enabled)
        self.assertEqual(info.supported_languages, ("en", "tr"))
        self.assertIn(LANGUAGE_EMBEDDING_KEY, state)
        self.assertIn(GLOBAL_LANGUAGE_EMBEDDING_KEY, state)
        for missing_key in (LANGUAGE_EMBEDDING_KEY, GLOBAL_LANGUAGE_EMBEDDING_KEY):
            incomplete = dict(state)
            incomplete.pop(missing_key)
            with self.subTest(missing_key=missing_key):
                with self.assertRaises(LegacyLanguageCheckpointError):
                    inspect_language_conditioning(incomplete)

    def test_multilingual_checkpoint_validates_global_language_embedding_width(self):
        state = dict(_tiny_flow_model(use_language_conditioning=True).state_dict())
        global_embedding = state[GLOBAL_LANGUAGE_EMBEDDING_KEY]
        state[GLOBAL_LANGUAGE_EMBEDDING_KEY] = torch.empty(
            global_embedding.shape[0],
            global_embedding.shape[1] + 1,
        )

        with self.assertRaisesRegex(LegacyLanguageCheckpointError, "Global.*width"):
            inspect_language_conditioning(state)

    def test_real_text_only_checkpoint_initializes_exact_speaker_branch(self):
        text_only = _tiny_flow_model()
        speaker_model = _tiny_flow_model(use_speaker_conditioning=True)
        text_state = text_only.state_dict()
        speaker_state = speaker_model.state_dict()
        expected_missing = set(speaker_state) - set(text_state)

        self.assertIn("null_speaker_state", expected_missing)
        self.assertIn("dit.speaker_encoder.global_timbre_token", expected_missing)
        self.assertIn("dit.speaker_norm.weight", expected_missing)
        for block_index in range(2):
            self.assertIn(
                f"dit.blocks.{block_index}.attention.wk_speaker.weight",
                expected_missing,
            )
            self.assertIn(
                f"dit.blocks.{block_index}.attention.wv_speaker.weight",
                expected_missing,
            )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "text-only.bin"
            torch.save(text_state, checkpoint_path)

            result = load_pretrained_checkpoint(
                speaker_model,
                checkpoint_path,
                initialize_speaker_conditioning=True,
            )

        self.assertEqual(set(result.missing_keys), expected_missing)
        self.assertFalse(result.unexpected_keys)

    def test_text_only_checkpoint_can_initialize_fixed_reference_resampler(self):
        text_only = _tiny_flow_model()
        speaker_model = _tiny_flow_model(
            use_speaker_conditioning=True,
            speaker_num_summary_tokens=3,
        )
        expected_missing = set(speaker_model.state_dict()) - set(text_only.state_dict())

        self.assertIn("speaker_resampler_version", expected_missing)
        self.assertIn("speaker_num_summary_tokens_metadata", expected_missing)
        self.assertIn("dit.speaker_resampler.query_tokens", expected_missing)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "text-only.bin"
            torch.save(text_only.state_dict(), checkpoint_path)

            result = load_pretrained_checkpoint(
                speaker_model,
                checkpoint_path,
                initialize_speaker_conditioning=True,
            )

        self.assertEqual(set(result.missing_keys), expected_missing)
        self.assertFalse(result.unexpected_keys)

    def test_speaker_initialization_rejects_missing_non_speaker_state(self):
        text_only = _tiny_flow_model()
        text_state = dict(text_only.state_dict())
        text_state.pop("dit.in_proj.bias")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "incomplete-text-only.bin"
            torch.save(text_state, checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "incompatible with conditioning"):
                load_pretrained_checkpoint(
                    _tiny_flow_model(use_speaker_conditioning=True),
                    checkpoint_path,
                    initialize_speaker_conditioning=True,
                )

    def test_speaker_resampler_topology_is_authenticated_or_legacy_zero(self):
        legacy_state = _tiny_flow_model(use_speaker_conditioning=True).state_dict()
        fixed_state = _tiny_flow_model(
            use_speaker_conditioning=True,
            speaker_num_summary_tokens=3,
        ).state_dict()

        self.assertEqual(inspect_speaker_num_summary_tokens(legacy_state), 0)
        self.assertEqual(inspect_speaker_num_summary_tokens(fixed_state), 3)
        checkpoint = FlowCheckpoint(path=Path("fixed.bin"), state_dict=fixed_state)
        self.assertEqual(checkpoint.infer_speaker_num_summary_tokens(), 3)

    def test_speaker_resampler_rejects_partial_or_tampered_topologies(self):
        state = dict(
            _tiny_flow_model(
                use_speaker_conditioning=True,
                speaker_num_summary_tokens=3,
            ).state_dict()
        )
        invalid_states = []

        missing_metadata = dict(state)
        missing_metadata.pop("speaker_resampler_version")
        invalid_states.append(missing_metadata)

        wrong_version = dict(state)
        wrong_version["speaker_resampler_version"] = torch.tensor(2)
        invalid_states.append(wrong_version)

        zero_tokens = dict(state)
        zero_tokens["speaker_num_summary_tokens_metadata"] = torch.tensor(0)
        invalid_states.append(zero_tokens)

        wrong_queries = dict(state)
        wrong_queries["dit.speaker_resampler.query_tokens"] = torch.empty(2, 8)
        invalid_states.append(wrong_queries)

        incomplete_parameters = dict(state)
        incomplete_parameters.pop("dit.speaker_resampler.q_proj.weight")
        invalid_states.append(incomplete_parameters)

        for invalid_state in invalid_states:
            with self.subTest(keys=len(invalid_state)):
                with self.assertRaises(LegacySpeakerCheckpointError):
                    inspect_speaker_num_summary_tokens(invalid_state)

    def test_training_loader_rejects_different_speaker_summary_topology(self):
        checkpoint_model = _tiny_flow_model(
            use_speaker_conditioning=True,
            speaker_num_summary_tokens=3,
        )
        target_model = _tiny_flow_model(
            use_speaker_conditioning=True,
            speaker_num_summary_tokens=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "fixed-speaker.bin"
            torch.save(checkpoint_model.state_dict(), checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "summary-token topologies"):
                load_pretrained_checkpoint(target_model, checkpoint_path)

    def test_target_patch_size_metadata_is_required_and_positive(self):
        state = dict(_tiny_flow_model().state_dict())

        self.assertEqual(inspect_target_patch_size(state), 2)

        missing = dict(state)
        missing.pop("target_patch_size_metadata")
        with self.assertRaisesRegex(
            LegacyArchitectureCheckpointError,
            "do not declare the target-latent patch size",
        ):
            inspect_target_patch_size(missing)

        for invalid_size in (0, -1):
            invalid = dict(state)
            invalid["target_patch_size_metadata"] = torch.tensor(invalid_size)
            with self.subTest(invalid_size=invalid_size):
                with self.assertRaisesRegex(
                    LegacyArchitectureCheckpointError,
                    "patch size must be positive",
                ):
                    inspect_target_patch_size(invalid)


if __name__ == "__main__":
    unittest.main()
