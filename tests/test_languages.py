"""Stable language registry and cross-lingual request tests."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from nar_vae.benchmark import _run_once
from nar_vae.checkpoint import LanguageCheckpointInfo, ReferenceLanguageCheckpointInfo
from nar_vae.dataset.data_collator import FlowMatchingDataCollator
from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.languages import (
    CrossLingualUnsupportedError,
    LanguagePair,
    MultilingualUnsupportedError,
    UnsupportedLanguageError,
    language_from_id,
    language_id,
    normalize_language,
)


class LanguageRegistryTest(unittest.TestCase):
    @staticmethod
    def _checkpoint(language_info: LanguageCheckpointInfo) -> Mock:
        checkpoint = Mock()
        checkpoint.infer_text_vocab_size.return_value = 530
        checkpoint.infer_speaker_conditioning.return_value = False
        checkpoint.language_capability.return_value = language_info
        checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(
            False
        )
        return checkpoint

    def test_aliases_normalize_to_stable_bcp47_style_codes(self):
        self.assertEqual(normalize_language("EN_us"), "en")
        self.assertEqual(normalize_language("TR_tr"), "tr")
        self.assertEqual(normalize_language("zh-tw"), "zh-Hant")
        self.assertEqual(normalize_language("zh-Hant-HK"), "zh-Hant")
        self.assertEqual(language_from_id(language_id("Japanese")).code, "ja")

    def test_unknown_language_is_rejected_without_guessing(self):
        with self.assertRaisesRegex(UnsupportedLanguageError, "Unsupported language"):
            normalize_language("xx-fictional")

    def test_language_pair_keeps_source_and_target_independent(self):
        pair = LanguagePair.resolve("es", "en", has_reference=True)

        self.assertEqual(pair.target, "es")
        self.assertEqual(pair.reference, "en")
        self.assertTrue(pair.is_cross_lingual)

        with self.assertRaisesRegex(ValueError, "reference_language requires"):
            LanguagePair.resolve("es", "en", has_reference=False)

    def test_collator_preserves_target_and_reference_languages_separately(self):
        batch = FlowMatchingDataCollator(speaker_patch_size=2)._collate_tts(
            [
                {
                    "latents": torch.zeros(2, 2),
                    "conditioning_ids": [1],
                    "language": "es",
                    "speaker_latents": torch.ones(2, 2),
                    "speaker_language": "en",
                },
                {
                    "latents": torch.zeros(2, 2),
                    "conditioning_ids": [1],
                    "language": "ja",
                    "speaker_latents": torch.ones(2, 2),
                    "speaker_language": "ko",
                },
            ]
        )

        torch.testing.assert_close(
            batch["language_ids"],
            torch.tensor([language_id("es"), language_id("ja")]),
        )
        torch.testing.assert_close(
            batch["speaker_language_ids"],
            torch.tensor([language_id("en"), language_id("ko")]),
        )

    def test_id_only_rows_default_reference_language_to_the_target(self):
        batch = FlowMatchingDataCollator(speaker_patch_size=1)._collate_tts(
            [
                {
                    "latents": torch.zeros(2, 1),
                    "conditioning_ids": [1],
                    "language_id": language_id("ja"),
                    "speaker_latents": torch.ones(2, 1),
                }
            ]
        )

        torch.testing.assert_close(batch["language_ids"], batch["speaker_language_ids"])

    def test_inference_capability_gate_does_not_claim_unsupported_weights(self):
        runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
        runtime.checkpoint_path = Path("multilingual.bin")
        runtime.device = torch.device("cpu")
        runtime.supports_voice_cloning = True
        runtime.uses_language_conditioning = True
        runtime.supported_languages = ("en", "es")
        runtime.supported_reference_languages = ("en", "ja")
        runtime.supported_language_pairs = (
            LanguagePair("es", "en"),
            LanguagePair("es", "ja"),
        )
        runtime.supports_cross_lingual = True

        english_reference = runtime._resolve_language_pair(
            "es",
            "en",
            has_reference=True,
        )
        japanese_reference = runtime._resolve_language_pair(
            "es",
            "ja",
            has_reference=True,
        )

        self.assertEqual(english_reference.target, japanese_reference.target)
        torch.testing.assert_close(
            runtime._language_ids(english_reference),
            runtime._language_ids(japanese_reference),
        )
        with self.assertRaises(MultilingualUnsupportedError):
            runtime._resolve_language_pair("fr", None, has_reference=False)

    def test_cross_lingual_pair_requires_explicit_reference_coverage(self):
        runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
        runtime.checkpoint_path = Path("speaker-only.bin")
        runtime.supports_voice_cloning = True
        runtime.uses_language_conditioning = False
        runtime.supported_languages = ("en",)
        runtime.supported_reference_languages = ()
        runtime.supported_language_pairs = ()
        runtime.supports_cross_lingual = False

        with self.assertRaisesRegex(
            CrossLingualUnsupportedError,
            "does not declare trained target/reference language pair",
        ):
            runtime._resolve_language_pair("en", "es", has_reference=True)

    def test_speaker_only_checkpoint_accepts_same_language_reference(self):
        runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
        runtime.checkpoint_path = Path("speaker-only.bin")
        runtime.supports_voice_cloning = True
        runtime.uses_language_conditioning = False
        runtime.supported_languages = ("en",)
        runtime.supported_reference_languages = ()
        runtime.supported_language_pairs = (LanguagePair("en", "en"),)

        pair = runtime._resolve_language_pair("en", None, has_reference=True)

        self.assertEqual(pair, LanguagePair("en", "en"))

    def test_inference_rejects_untrained_pair_from_supported_language_projections(self):
        runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
        runtime.checkpoint_path = Path("exact-pairs.bin")
        runtime.supports_voice_cloning = True
        runtime.uses_language_conditioning = True
        runtime.supported_languages = ("en", "es")
        runtime.supported_reference_languages = ("en", "ja")
        runtime.supported_language_pairs = (
            LanguagePair("es", "en"),
            LanguagePair("en", "ja"),
        )

        runtime._resolve_language_pair("es", "en", has_reference=True)
        with self.assertRaisesRegex(CrossLingualUnsupportedError, r"\('es', 'ja'\)"):
            runtime._resolve_language_pair("es", "ja", has_reference=True)
        with self.assertRaisesRegex(CrossLingualUnsupportedError, r"\('en', 'en'\)"):
            runtime._resolve_language_pair("en", "en", has_reference=True)

    def test_inference_cannot_disable_checkpoint_language_conditioning(self):
        checkpoint = self._checkpoint(LanguageCheckpointInfo(True, ("en", "es")))
        with (
            patch("nar_vae.inference.FlowCheckpoint.load", return_value=checkpoint),
            self.assertRaisesRegex(MultilingualUnsupportedError, "cannot be disabled"),
        ):
            FlowMatchingTTSInference(
                "checkpoint.bin",
                use_language_conditioning=False,
                device="cpu",
            )

    def test_legacy_checkpoint_rejects_claimed_non_english_support(self):
        checkpoint = self._checkpoint(LanguageCheckpointInfo(False))
        with (
            patch("nar_vae.inference.FlowCheckpoint.load", return_value=checkpoint),
            self.assertRaisesRegex(MultilingualUnsupportedError, "only English"),
        ):
            FlowMatchingTTSInference(
                "checkpoint.bin",
                supported_languages=["en", "es"],
                device="cpu",
            )

    def test_benchmark_propagates_the_complete_language_pair(self):
        class RecordingRuntime:
            def synthesize_fast(self, text, **kwargs):
                self.text = text
                self.kwargs = kwargs
                return torch.ones(4), {"ttft": 0.1}

        runtime = RecordingRuntime()
        _run_once(
            runtime,
            text="hola",
            duration=1.0,
            config=object(),
            seed=7,
            language_pair=LanguagePair.resolve("es", "en", has_reference=True),
            reference_audio="reference.wav",
            reference_sample_rate=None,
            phonemes=("o", "l", "a"),
            language_spans=None,
        )

        self.assertEqual(runtime.kwargs["language"], "es")
        self.assertEqual(runtime.kwargs["reference_language"], "en")
        self.assertEqual(runtime.kwargs["reference_audio"], "reference.wav")
        self.assertEqual(runtime.kwargs["phonemes"], ("o", "l", "a"))
        self.assertIsNone(runtime.kwargs["language_spans"])


if __name__ == "__main__":
    unittest.main()
