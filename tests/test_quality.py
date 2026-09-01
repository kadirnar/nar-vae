"""Tests for dependency-light audio quality calculations."""

import unittest
from unittest.mock import patch

import torch

from nar_vae.quality import (
    audio_metrics,
    cross_lingual_quality_report,
    evaluate_audio_file,
    normalize_transcript,
    speaker_similarity,
    transcript_error_rate,
    word_error_rate,
)


class QualityTest(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(normalize_transcript("Hello, WORLD!"), ["HELLO", "WORLD"])

    def test_word_error_rate_counts_substitution_insertion_and_deletion(self):
        errors, rate = word_error_rate(
            "the quick brown fox",
            "the slow brown fox again",
        )

        self.assertEqual(errors, 2)
        self.assertEqual(rate, 0.5)

    def test_audio_metrics_report_duration_and_silence(self):
        waveform = torch.zeros(1, 48000)
        metrics = audio_metrics(waveform, 48000)

        self.assertEqual(metrics["duration_s"], 1.0)
        self.assertEqual(metrics["near_silence_ratio"], 1.0)
        self.assertEqual(metrics["clipping_ratio"], 0.0)

    def test_east_asian_scripts_use_character_error_rate(self):
        result = transcript_error_rate("こんにちは世界", "こんにちは世間", language="ja")

        self.assertEqual(result["metric"], "cer")
        self.assertEqual(result["errors"], 1)

    def test_multilingual_latin_text_uses_word_error_rate(self):
        result = transcript_error_rate(
            "Bonjour, le monde !",
            "bonjour le vaste monde",
            language="fr",
        )

        self.assertEqual(result["metric"], "wer")
        self.assertEqual(result["errors"], 1)

    def test_speaker_similarity_rejects_incomparable_embeddings(self):
        self.assertAlmostEqual(
            speaker_similarity(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
            1.0,
        )
        with self.assertRaisesRegex(ValueError, "same flattened shape"):
            speaker_similarity(torch.ones(2), torch.ones(3))

    def test_cross_lingual_report_keeps_both_language_roles(self):
        report = cross_lingual_quality_report(
            "hola mundo",
            "hola mundo",
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.0, 0.9]),
            target_language="es",
            reference_language="en",
            asr_model="fixed-asr-v1",
            speaker_verification_model="fixed-sv-v1",
        )

        self.assertEqual(
            report["language_pair"],
            {"target": "es", "reference": "en", "cross_lingual": True},
        )
        self.assertEqual(report["intelligibility"]["error_rate"], 0.0)
        self.assertGreater(report["speaker_similarity"], 0.99)
        self.assertEqual(
            report["models"],
            {"asr": "fixed-asr-v1", "speaker_verification": "fixed-sv-v1"},
        )
        self.assertIsNone(report["passed"])

    def test_cross_lingual_thresholds_are_explicit_and_optional(self):
        report = cross_lingual_quality_report(
            "hola mundo",
            "hola mundo",
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.0]),
            target_language="es",
            reference_language="en",
            maximum_error_rate=0.1,
            minimum_speaker_similarity=0.9,
        )

        self.assertEqual(
            report["checks"],
            {"intelligibility": True, "speaker_similarity": True},
        )
        self.assertTrue(report["passed"])

        with self.assertRaisesRegex(ValueError, "maximum_error_rate"):
            cross_lingual_quality_report(
                "hola",
                "hola",
                torch.ones(2),
                torch.ones(2),
                target_language="es",
                reference_language="en",
                maximum_error_rate=1.1,
            )

    def test_audio_evaluation_requires_an_explicit_wer_gate_to_pass(self):
        waveform = torch.full((1, 16000), 0.1)
        evaluator = {
            "id": "test-asr",
            "revision": "a" * 64,
            "revision_kind": "checkpoint_sha256",
        }
        with (
            patch("nar_vae.quality.torchaudio.load", return_value=(waveform, 16000)),
            patch(
                "nar_vae.quality.file_metadata",
                return_value={"path": "generated.wav", "size_bytes": 1, "sha256": "b" * 64},
            ),
            patch(
                "nar_vae.quality._greedy_wav2vec_transcript",
                return_value=("hello world", "test-asr", evaluator),
            ),
        ):
            ungated = evaluate_audio_file("generated.wav", "hello world")
            gated = evaluate_audio_file(
                "generated.wav",
                "hello world",
                maximum_wer=0.1,
            )

        self.assertEqual(ungated["word_error_rate"], 0.0)
        self.assertIsNone(ungated["checks"]["intelligible"])
        self.assertIsNone(ungated["passed"])
        self.assertIsNone(ungated["maximum_word_error_rate"])
        self.assertEqual(ungated["asr_evaluator"], evaluator)
        self.assertTrue(gated["checks"]["intelligible"])
        self.assertTrue(gated["passed"])

    def test_audio_evaluation_rejects_invalid_wer_gate_before_loading_audio(self):
        with (
            patch("nar_vae.quality.torchaudio.load") as load,
            self.assertRaisesRegex(ValueError, "maximum_wer"),
        ):
            evaluate_audio_file("generated.wav", "hello", maximum_wer=1.1)

        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
