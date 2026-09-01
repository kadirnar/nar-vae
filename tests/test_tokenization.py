"""Dependency-free multilingual frontend contracts."""

import importlib.util
import unittest
from pathlib import Path


def load_tokenization_module():
    module_path = Path(__file__).parents[1] / "nar_vae" / "tokenization.py"
    spec = importlib.util.spec_from_file_location("nar_vae_tokenization_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tokenization = load_tokenization_module()


class TokenizationTest(unittest.TestCase):
    def test_plain_text_uses_compact_v2_envelope_and_parallel_masks(self):
        encoded = tokenization.encode_tts_conditioning("Hi", language="en")

        self.assertEqual(encoded.conditioning_ids[0], tokenization.START_OF_TEXT)
        self.assertEqual(encoded.conditioning_ids[-1], tokenization.END_OF_TEXT)
        self.assertEqual(len(encoded.conditioning_ids), len(encoded.token_language_ids))
        self.assertEqual(len(encoded.conditioning_ids), len(encoded.alignment_mask))
        self.assertFalse(encoded.alignment_mask[0])
        self.assertFalse(encoded.alignment_mask[-1])
        self.assertTrue(all(encoded.alignment_mask[1:-1]))
        self.assertLess(tokenization.TOTAL_VOCAB_SIZE, 1024)
        self.assertEqual(tokenization.PAD_TOKEN, 0)

    def test_utf8_fallback_is_complete_and_lossless(self):
        encoded = tokenization.encode_tts_conditioning("世界 🌍", language="ja")

        self.assertIn(tokenization.BYTE_START, encoded.conditioning_ids)
        self.assertIn(tokenization.BYTE_END, encoded.conditioning_ids)
        self.assertEqual(tokenization.decode_utf8_fallback(encoded.conditioning_ids), "世界🌍")
        for token_id, aligned in zip(encoded.conditioning_ids, encoded.alignment_mask):
            if token_id in (tokenization.BYTE_START, tokenization.BYTE_END):
                self.assertFalse(aligned)
        # Each Unicode character owns one duration anchor, independent of its
        # UTF-8 byte width. Whitespace remains a non-acoustic boundary.
        self.assertEqual(sum(encoded.alignment_mask), 3)

    def test_utf8_combining_mark_stays_on_its_base_duration_anchor(self):
        encoded = tokenization.encode_tts_conditioning(
            "क\N{DEVANAGARI VOWEL SIGN AA}", language="hi"
        )

        self.assertEqual(tokenization.decode_utf8_fallback(encoded.conditioning_ids), "का")
        self.assertEqual(sum(encoded.alignment_mask), 1)

    def test_reviewed_phones_stress_tone_and_pause_have_distinct_alignment(self):
        encoded = tokenization.encode_tts_conditioning(
            "",
            phonemes="m e ˈ r | h a ˥ <pause>",
            language="tr",
        )

        stress = tokenization.SUPRASEGMENTAL_TOKENS["ˈ"]
        tone = tokenization.SUPRASEGMENTAL_TOKENS["˥"]
        self.assertFalse(encoded.alignment_mask[encoded.conditioning_ids.index(stress)])
        self.assertFalse(encoded.alignment_mask[encoded.conditioning_ids.index(tone)])
        self.assertTrue(
            encoded.alignment_mask[encoded.conditioning_ids.index(tokenization.EXPLICIT_PAUSE)]
        )
        self.assertFalse(
            encoded.alignment_mask[encoded.conditioning_ids.index(tokenization.WORD_BOUNDARY)]
        )

    def test_emotion_style_token_never_consumes_a_mas_frame(self):
        encoded = tokenization.encode_tts_conditioning("Hello <laugh> world", language="en")
        emotion_index = encoded.conditioning_ids.index(tokenization.EMOTION_TAGS["<laugh>"])

        self.assertFalse(encoded.alignment_mask[emotion_index])
        self.assertEqual(encoded.token_language_ids[emotion_index], 0)

    def test_explicit_spans_assign_content_languages_for_code_switching(self):
        encoded = tokenization.encode_tts_conditioning(
            None,
            language="en",
            language_spans=[
                {"text": "hello", "language": "English"},
                {"text": "dünya", "language": "tr"},
            ],
        )
        aligned_languages = {
            language
            for language, aligned in zip(encoded.token_language_ids, encoded.alignment_mask)
            if aligned
        }

        from nar_vae.languages import language_id

        self.assertEqual(aligned_languages, {language_id("en"), language_id("tr")})

    def test_encode_tts_text_preserves_list_return_api(self):
        ids = tokenization.encode_tts_text("Merhaba", object(), language="Turkish")

        self.assertIsInstance(ids, list)
        self.assertEqual(ids, tokenization.encode_tts_conditioning("Merhaba", language="tr")[0])

    def test_checkpoint_vocab_and_unknown_language_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "exceed the checkpoint vocabulary"):
            tokenization.encode_tts_text("Hi", vocab_size=2)
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            tokenization.encode_tts_text("encodable text", language="xx")


if __name__ == "__main__":
    unittest.main()
