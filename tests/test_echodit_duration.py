"""Deterministic EchoDiT v2 duration architecture and compatibility tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vyvotts.checkpoint import (
    DurationCheckpointInfo,
    FlowCheckpoint,
    LanguageCheckpointInfo,
    LegacyDurationCheckpointError,
    ReferenceLanguageCheckpointInfo,
    inspect_duration_capability,
    load_pretrained_checkpoint,
)
from vyvotts.configuration import DurationConfig
from vyvotts.inference import (
    FlowMatchingTTSInference,
    LearnedDurationUnsupportedError,
)
from vyvotts.languages import language_id
from vyvotts.models.flow_matching import create_flow_matching_echodit


def tiny_model(
    *,
    duration: bool,
    language: bool = False,
    speaker: bool = False,
    duration_speaker: bool = False,
):
    return create_flow_matching_echodit(
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
        use_speaker_conditioning=speaker,
        use_language_conditioning=language,
        supported_languages=("en", "tr") if language else None,
        use_duration_predictor=duration,
        duration_predictor_hidden_size=6,
        duration_predictor_num_layers=2,
        duration_predictor_use_speaker=duration_speaker,
    )


class EchoDiTDurationTest(unittest.TestCase):
    @staticmethod
    def checkpoint_mock(duration: DurationCheckpointInfo) -> Mock:
        checkpoint = Mock()
        checkpoint.infer_text_vocab_size.return_value = 20
        checkpoint.infer_speaker_conditioning.return_value = False
        checkpoint.language_capability.return_value = LanguageCheckpointInfo(False)
        checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(
            False
        )
        checkpoint.duration_capability.return_value = duration
        return checkpoint

    def test_legacy_model_has_no_duration_capability(self):
        model = tiny_model(duration=False)

        self.assertFalse(inspect_duration_capability(model.state_dict()).enabled)
        with self.assertRaisesRegex(RuntimeError, "versioned EchoDiT v2"):
            model.predict_duration_frames(torch.tensor([[1, 2]]))

    def test_versioned_head_predicts_positive_frames_and_ignores_padding(self):
        torch.manual_seed(5)
        model = tiny_model(duration=True).eval()
        mask = torch.tensor([[True, True, False]])

        first = model.predict_duration_frames(torch.tensor([[1, 2, 3]]), mask)
        changed_padding = model.predict_duration_frames(torch.tensor([[1, 2, 9]]), mask)

        self.assertEqual(first.dtype, torch.long)
        self.assertGreaterEqual(first.item(), 1)
        torch.testing.assert_close(first, changed_padding)

    def test_duration_frame_conversion_rejects_nonfinite_and_overflow_predictions(self):
        model = tiny_model(duration=True).eval()
        ids = torch.tensor([[1, 2]])

        for value, message in (
            (float("nan"), "non-finite log frame count"),
            (float("inf"), "non-finite log frame count"),
            (100.0, "representable frame-count range"),
        ):
            with (
                self.subTest(value=value),
                patch.object(
                    model,
                    "predict_log_duration",
                    return_value=torch.tensor([value]),
                ),
                self.assertRaisesRegex(RuntimeError, message),
            ):
                model.predict_duration_frames(ids)

    def test_target_language_reaches_duration_head_independently(self):
        torch.manual_seed(7)
        model = tiny_model(duration=True, language=True).eval()
        ids = torch.tensor([[1, 2]])
        english = torch.tensor([language_id("en")])
        turkish = torch.tensor([language_id("tr")])

        english_prediction = model.predict_log_duration(ids, language_ids=english)
        turkish_prediction = model.predict_log_duration(ids, language_ids=turkish)

        self.assertFalse(torch.equal(english_prediction, turkish_prediction))

    def test_optional_speaker_duration_path_is_versioned_separately(self):
        torch.manual_seed(13)
        model = tiny_model(
            duration=True,
            speaker=True,
            duration_speaker=True,
        ).eval()
        capability = inspect_duration_capability(model.state_dict())
        ids = torch.tensor([[1, 2]])

        first = model.predict_log_duration(ids, speaker_latent=torch.zeros(1, 4, 2))
        second = model.predict_log_duration(ids, speaker_latent=torch.ones(1, 4, 2))

        self.assertTrue(capability.uses_speaker)
        self.assertFalse(torch.equal(first, second))

    def test_checkpoint_rejects_unversioned_or_incomplete_duration_state(self):
        state = tiny_model(duration=True).state_dict()
        checkpoint = FlowCheckpoint(path=Path("v2.bin"), state_dict=state)
        capability = checkpoint.duration_capability()

        self.assertEqual(capability.hidden_size, 6)
        self.assertEqual(capability.num_layers, 2)
        self.assertFalse(capability.uses_speaker)

        del state["duration_predictor_version"]
        with self.assertRaisesRegex(LegacyDurationCheckpointError, "complete"):
            inspect_duration_capability(state)

    def test_pretrained_loader_requires_explicit_v2_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.bin"
            torch.save(tiny_model(duration=False).state_dict(), path)

            with self.assertRaisesRegex(RuntimeError, "initialize_duration_predictor"):
                load_pretrained_checkpoint(tiny_model(duration=True), path)

            model = tiny_model(duration=True)
            result = load_pretrained_checkpoint(
                model,
                path,
                initialize_duration_predictor=True,
            )

        self.assertTrue(result.missing_keys)
        self.assertTrue(
            all(
                key.startswith("duration_predictor.")
                or key.startswith("duration_predictor_")
                or key == "echodit_architecture_version"
                for key in result.missing_keys
            )
        )

    def test_inference_uses_learned_frames_but_explicit_duration_wins(self):
        runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
        runtime.uses_learned_duration = True
        runtime.frame_rate = 25.0
        runtime.settings = SimpleNamespace(duration=DurationConfig(0.1, 1.0, 5.0))
        runtime.flow_model = SimpleNamespace(
            predict_duration_frames=lambda *args, **kwargs: torch.tensor([80])
        )
        ids = torch.tensor([[1, 2]])

        learned_seconds, learned_frames = runtime._resolve_duration_shape(
            "hello",
            None,
            ids,
            None,
            None,
        )
        explicit_seconds, explicit_frames = runtime._resolve_duration_shape(
            "hello",
            2.0,
            ids,
            None,
            None,
        )

        self.assertEqual((learned_seconds, learned_frames), (3.2, 80))
        self.assertEqual((explicit_seconds, explicit_frames), (2.0, 50))

    def test_inference_rejects_learned_duration_above_the_configured_cap(self):
        runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
        runtime.uses_learned_duration = True
        runtime.frame_rate = 25.0
        runtime.settings = SimpleNamespace(duration=DurationConfig(0.1, 1.0, 5.0))
        runtime.flow_model = SimpleNamespace(
            predict_duration_frames=lambda *args, **kwargs: torch.tensor([126])
        )

        with self.assertRaisesRegex(ValueError, "Learned duration prediction.*exceeds"):
            runtime._resolve_duration_shape(
                "hello",
                None,
                torch.tensor([[1, 2]]),
                None,
                None,
            )

    def test_inference_cannot_claim_or_disable_learned_duration(self):
        legacy = self.checkpoint_mock(DurationCheckpointInfo(False))
        with (
            patch("vyvotts.inference.FlowCheckpoint.load", return_value=legacy),
            self.assertRaises(LearnedDurationUnsupportedError),
        ):
            FlowMatchingTTSInference(
                "checkpoint.bin",
                use_duration_predictor=True,
                device="cpu",
            )

        versioned = self.checkpoint_mock(DurationCheckpointInfo(True, 6, 2, False))
        with (
            patch("vyvotts.inference.FlowCheckpoint.load", return_value=versioned),
            self.assertRaises(LearnedDurationUnsupportedError),
        ):
            FlowMatchingTTSInference(
                "checkpoint.bin",
                use_duration_predictor=False,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
