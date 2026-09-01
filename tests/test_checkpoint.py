"""Tests for EchoDiT checkpoint compatibility inference."""

import hashlib
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
    inspect_architecture_version,
    inspect_duration_capability,
    inspect_language_conditioning,
    inspect_speaker_num_summary_tokens,
    inspect_target_patch_size,
    load_pretrained_checkpoint,
    resolve_flow_checkpoint,
)
from nar_vae.languages import language_id
from nar_vae.models.duration import (
    ECHODIT_ARCHITECTURE_VERSION,
    LEGACY_ECHODIT_ARCHITECTURE_VERSION,
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
    architecture_version: int = ECHODIT_ARCHITECTURE_VERSION,
    use_duration_predictor: bool = False,
    use_mas_duration: bool = False,
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
        target_patch_size=(1 if architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION else 2),
        timestep_embed_size=8,
        adaln_rank=2,
        use_speaker_conditioning=use_speaker_conditioning,
        use_language_conditioning=use_language_conditioning,
        speaker_num_summary_tokens=speaker_num_summary_tokens,
        supported_languages=("en", "tr") if use_language_conditioning else None,
        supported_language_pairs=(
            (("en", "en"), ("tr", "tr"))
            if use_speaker_conditioning and use_language_conditioning
            else None
        ),
        use_duration_predictor=use_duration_predictor,
        duration_predictor_hidden_size=8,
        duration_predictor_num_layers=1,
        use_mas_duration=use_mas_duration,
        duration_alignment_hidden_size=2,
        generative_objective=generative_objective,
        diffusion_schedule_shift=diffusion_schedule_shift,
        architecture_version=architecture_version,
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
    def test_authenticated_v3_topology_loads_exactly_with_or_without_duration_metadata(self):
        legacy_kwargs = {
            "architecture_version": LEGACY_ECHODIT_ARCHITECTURE_VERSION,
            "use_speaker_conditioning": True,
            "use_language_conditioning": True,
        }
        source = _tiny_flow_model(
            **legacy_kwargs,
            use_duration_predictor=True,
            use_mas_duration=True,
        )
        state = source.state_dict()
        self.assertEqual(
            inspect_architecture_version(state, expected_version=3),
            LEGACY_ECHODIT_ARCHITECTURE_VERSION,
        )
        self.assertTrue(
            inspect_duration_capability(
                state,
                expected_architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
            ).enabled
        )
        for v4_only_key in (
            "target_patch_size_metadata",
            "null_speaker_state",
            "dit.speaker_encoder.global_timbre_token",
            GLOBAL_LANGUAGE_EMBEDDING_KEY,
        ):
            self.assertNotIn(v4_only_key, state)

        target = _tiny_flow_model(
            **legacy_kwargs,
            use_duration_predictor=True,
            use_mas_duration=True,
        )
        FlowCheckpoint(path=Path("legacy-v3.bin"), state_dict=state).load_into(target)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

        duration_disabled = _tiny_flow_model(**legacy_kwargs).state_dict()
        self.assertNotIn("echodit_architecture_version", duration_disabled)
        checkpoint = FlowCheckpoint(path=Path("legacy-v3.bin"), state_dict=duration_disabled)
        self.assertEqual(
            checkpoint.validate_architecture(
                expected_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
                allow_missing_legacy_metadata=True,
            ),
            LEGACY_ECHODIT_ARCHITECTURE_VERSION,
        )
        self.assertEqual(
            checkpoint.infer_target_patch_size(
                expected_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
                allow_missing_legacy_metadata=True,
            ),
            1,
        )
        with self.assertRaises(LegacyArchitectureCheckpointError):
            checkpoint.validate_architecture()

        impossible_metadata = dict(duration_disabled)
        impossible_metadata["echodit_architecture_version"] = torch.tensor(3)
        with self.assertRaisesRegex(
            LegacyArchitectureCheckpointError,
            "present exactly when the duration predictor",
        ):
            inspect_architecture_version(impossible_metadata, expected_version=3)

    def test_v3_language_default_and_duration_mas_keep_full_legacy_token_axis(self):
        model = _tiny_flow_model(
            architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
            use_language_conditioning=True,
            use_duration_predictor=True,
            use_mas_duration=True,
        ).eval()
        english = language_id("en")
        default_ids = model._prepare_language_ids(
            None,
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(default_ids, torch.tensor([english, english])))

        conditioning_ids = torch.tensor([[4, 5, 6, 0]])
        text_mask = torch.tensor([[True, True, True, False]])
        language_ids = torch.tensor([english])
        token_language_ids = torch.tensor([[english, english, english, 0]])
        alignment_mask = text_mask.clone()
        expected_text_state = model.dit.encode_text(
            conditioning_ids,
            text_mask,
            language_ids,
        )
        actual_text_state, actual_mask, full_alignment, speaker_state, speaker_mask = (
            model._encode_duration_conditioning(
                conditioning_ids,
                text_mask,
                None,
                None,
                language_ids,
                token_language_ids,
                alignment_mask,
            )
        )
        self.assertTrue(torch.equal(actual_text_state, expected_text_state))
        self.assertTrue(torch.equal(actual_mask, text_mask))
        self.assertTrue(torch.equal(full_alignment, text_mask))
        self.assertIsNone(speaker_state)
        self.assertIsNone(speaker_mask)
        expected_cache = model.dit.get_kv_cache_text(
            conditioning_ids,
            text_mask,
            language_ids,
        )
        compatibility_cache = model.encode_text(
            conditioning_ids,
            text_mask,
            language_ids,
            token_language_ids,
        )
        for expected_layer, actual_layer in zip(expected_cache, compatibility_cache):
            for expected_tensor, actual_tensor in zip(expected_layer, actual_layer):
                self.assertTrue(torch.equal(expected_tensor, actual_tensor))

        mismatched_languages = token_language_ids.clone()
        mismatched_languages[0, 1] = language_id("tr")
        with self.assertRaisesRegex(ValueError, "must broadcast"):
            model.encode_text(
                conditioning_ids,
                text_mask,
                language_ids,
                mismatched_languages,
            )
        partial_alignment = alignment_mask.clone()
        partial_alignment[0, 1] = False
        with self.assertRaisesRegex(ValueError, "select every valid text token"):
            model.encode_inference_conditioning(
                conditioning_ids,
                text_mask,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=partial_alignment,
            )

        expected_log, expected_weights = model.duration_predictor(
            expected_text_state,
            text_mask,
            None,
            None,
            return_token_durations=True,
        )
        actual_weights = model.predict_expected_token_durations(
            conditioning_ids,
            text_mask,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
        )
        self.assertTrue(torch.equal(actual_weights, expected_weights))

        encoded = model.encode_inference_conditioning(
            conditioning_ids,
            text_mask,
            cfg_mode="joint",
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
        )
        frames, cfg_weights = model.predict_duration_frames_and_token_weights(encoded)
        self.assertTrue(torch.equal(frames, model._duration_frames_from_log(expected_log)))
        self.assertTrue(torch.equal(cfg_weights, expected_weights))
        allocation = model.allocate_token_duration_frames(
            cfg_weights,
            text_mask,
            total_frames=12,
            alignment_mask=alignment_mask,
        )
        conditional, prepared_cfg = model.finalize_inference_conditioning(
            encoded,
            token_durations=allocation,
        )
        self.assertEqual(conditional.frame_text_state.shape[1], 12)
        self.assertIsNotNone(prepared_cfg)

    def test_v3_null_speaker_buffer_is_cast_at_the_acoustic_boundary(self):
        model = _tiny_flow_model(
            architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
            use_speaker_conditioning=True,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if torch.is_floating_point(parameter):
                    parameter.data = parameter.data.to(torch.bfloat16)
        latent, mask = model._prepare_speaker_inputs(
            None,
            None,
            batch_size=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(latent.dtype, torch.bfloat16)
        self.assertIsNone(mask)

    def test_v3_sparse_ema_rejects_each_v4_only_parameter_before_mutation(self):
        model = _tiny_flow_model(
            architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
            use_speaker_conditioning=True,
            use_language_conditioning=True,
        )
        base = model.state_dict()
        forbidden = {
            "null_speaker_state": torch.zeros(1, 1, 8),
            "dit.speaker_encoder.global_timbre_token": torch.zeros(1, 1, 8),
            GLOBAL_LANGUAGE_EMBEDDING_KEY: torch.zeros(19, 8),
        }
        for key, value in forbidden.items():
            target = _tiny_flow_model(
                architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
                use_speaker_conditioning=True,
                use_language_conditioning=True,
            )
            before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
            checkpoint = FlowCheckpoint(
                path=Path("legacy-v3-ema.bin"),
                state_dict={key: value},
                base_state_dict=base,
                is_ema=True,
            )
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(
                    LegacyArchitectureCheckpointError,
                    "v4-only state",
                ),
            ):
                checkpoint.load_into(target)
            for name, tensor in before.items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[name]), name)

    def test_v3_origin_frozen_numerical_oracle_covers_all_conditioning_paths(self):
        """Compare against values frozen independently from origin/main@122615c."""
        model = FlowMatchingEchoDiT(
            latent_size=2,
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
            adaln_rank=2,
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=("en", "tr"),
            supported_language_pairs=(("en", "en"), ("tr", "tr")),
            use_duration_predictor=True,
            duration_predictor_hidden_size=8,
            duration_predictor_num_layers=1,
            duration_predictor_use_speaker=True,
            use_mas_duration=True,
            duration_alignment_hidden_size=1,
            architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
        ).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
                generator = torch.Generator().manual_seed(seed)
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * 0.08
                )
        state_rows = [
            f"{key}:{tuple(value.shape)}:{value.dtype}"
            for key, value in sorted(model.state_dict().items())
        ]
        self.assertEqual(len(state_rows), 107)
        self.assertEqual(
            hashlib.sha256("\n".join(state_rows).encode()).hexdigest(),
            "97fa5d124cadbf9725aecc3cf834c1542b0d41bb1acc5d43784ca39263f63bf3",
        )

        conditioning_ids = torch.tensor([[2, 3, 4]])
        text_mask = torch.ones_like(conditioning_ids, dtype=torch.bool)
        language_ids = torch.tensor([language_id("en")])
        token_languages = language_ids[:, None].expand_as(conditioning_ids)
        speaker = torch.tensor([[[0.1, -0.2, 0.3, -0.4], [0.5, -0.6, 0.7, -0.8]]])
        latents = torch.tensor([[[0.2, -0.1, 0.4, -0.3], [0.6, -0.5, 0.8, -0.7]]])
        timesteps = torch.tensor([0.35])
        token_durations = torch.tensor([[1, 2, 1]])

        def assert_golden(actual: torch.Tensor, expected: list[float]) -> None:
            torch.testing.assert_close(
                actual.flatten(),
                torch.tensor(expected),
                rtol=1e-6,
                atol=1e-7,
            )

        assert_golden(
            model.dit.encode_text(conditioning_ids, text_mask, language_ids),
            [
                0.0077911378,
                -0.1141348556,
                0.0054870057,
                0.0191652589,
                -0.0558012538,
                -0.0252671875,
                0.0479272157,
                0.0747701749,
                0.0024035675,
                -0.0161669925,
                0.0084437970,
                -0.0091775842,
                -0.0803086907,
                -0.1056584120,
                -0.0522376224,
                0.0943915769,
                0.0042046485,
                -0.1390307397,
                0.0011998713,
                -0.0088239815,
                -0.0656334460,
                0.0538458489,
                -0.0131217521,
                0.0491950735,
            ],
        )
        assert_golden(
            model.dit.encode_speaker(speaker),
            [
                -0.2510858774,
                -0.1228565797,
                0.0082521383,
                0.0205281749,
                0.0015281101,
                -0.0250276029,
                -0.0762323812,
                0.0470382236,
                -0.2375718504,
                -0.1568381786,
                0.0087401737,
                0.0183225926,
                0.0154344235,
                -0.0147740506,
                -0.0718115121,
                0.0256661102,
            ],
        )
        common = {
            "attention_mask": text_mask,
            "speaker_latent": speaker,
            "language_ids": language_ids,
            "token_language_ids": token_languages,
            "alignment_mask": text_mask,
        }
        assert_golden(
            model.predict_log_duration(conditioning_ids, **common),
            [1.1287559271],
        )
        assert_golden(
            model.predict_expected_token_durations(conditioning_ids, **common),
            [0.6976966262, 0.6962333322, 0.6978779435],
        )
        assert_golden(
            model(
                latents,
                conditioning_ids,
                timesteps,
                use_cfg_dropout=False,
                token_durations=token_durations,
                **common,
            ),
            [
                -0.0743091702,
                -0.0724570081,
                -0.0822443143,
                -0.0685446933,
                -0.0635081083,
                -0.0749017969,
                -0.0544648245,
                -0.0800204501,
            ],
        )
        cfg_goldens = {
            "joint": [
                -0.0749091059,
                -0.0729265139,
                -0.0827476755,
                -0.0687103197,
                -0.0633427650,
                -0.0746498257,
                -0.0539036393,
                -0.0801517591,
            ],
            "independent": [
                -0.0761104077,
                -0.0738677308,
                -0.0837558061,
                -0.0690436959,
                -0.0630115569,
                -0.0741449445,
                -0.0527806841,
                -0.0804133639,
            ],
            "alternating": [
                -0.0746098012,
                -0.0726927593,
                -0.0824966356,
                -0.0686283708,
                -0.0634251833,
                -0.0747753456,
                -0.0541839488,
                -0.0800856873,
            ],
        }
        for mode, expected in cfg_goldens.items():
            assert_golden(
                model.forward_with_cfg(
                    latents,
                    conditioning_ids,
                    timesteps,
                    cfg_scale=1.4,
                    cfg_mode=mode,
                    cfg_scale_text=1.2,
                    cfg_scale_speaker=0.7,
                    step_idx=0,
                    token_durations=token_durations,
                    **common,
                ),
                expected,
            )

    def test_v3_training_forward_rejects_before_rng_or_encoder_work(self):
        model = _tiny_flow_model(
            architecture_version=LEGACY_ECHODIT_ARCHITECTURE_VERSION,
        ).train()
        with (
            patch("nar_vae.models.flow_matching.torch.rand") as random_draw,
            patch.object(model.dit, "encode_text") as encode_text,
            self.assertRaisesRegex(RuntimeError, "inference only"),
        ):
            model(
                torch.zeros(1, 2, 2),
                torch.ones(1, 2, dtype=torch.long),
                torch.zeros(1),
            )
        random_draw.assert_not_called()
        encode_text.assert_not_called()

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
