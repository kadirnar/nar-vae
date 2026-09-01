"""Tests for the dependency-free TTS tokenization helpers."""

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


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class TokenizationTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_plain_text_has_tts_control_tokens(self):
        encoded = tokenization.encode_tts_text("Hi", self.tokenizer)

        self.assertEqual(encoded[0], tokenization.START_OF_HUMAN)
        self.assertEqual(encoded[1:3], [ord("H"), ord("i")])
        self.assertEqual(
            encoded[-3:],
            [
                tokenization.END_OF_HUMAN,
                tokenization.START_OF_AI,
                tokenization.START_OF_SPEECH,
            ],
        )

    def test_emotion_tag_uses_shared_special_token(self):
        encoded = tokenization.encode_tts_text(
            "Hello <laugh> world",
            self.tokenizer,
        )

        self.assertIn(tokenization.EMOTION_TAGS["<laugh>"], encoded)
        self.assertNotIn(ord("<"), encoded)

    def test_unsupported_emotion_tag_is_encoded_as_literal_text(self):
        encoded = tokenization.encode_tts_text(
            "Hello <laugh>",
            self.tokenizer,
            vocab_size=100287,
        )

        self.assertNotIn(tokenization.EMOTION_TAGS["<laugh>"], encoded)
        self.assertIn(ord("<"), encoded)

    def test_tokens_are_validated_against_checkpoint_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "exceed the checkpoint vocabulary"):
            tokenization.encode_tts_text("Hi", self.tokenizer, vocab_size=100)

    def test_vocab_size_includes_every_emotion_tag(self):
        self.assertEqual(
            tokenization.TOTAL_VOCAB_SIZE,
            max(tokenization.EMOTION_TAGS.values()) + 1,
        )

    def test_language_is_validated_without_changing_unicode_tokens(self):
        canonical = tokenization.encode_tts_text(
            "Merhaba 世界",
            self.tokenizer,
            language="tr",
        )
        alias = tokenization.encode_tts_text(
            "Merhaba 世界",
            self.tokenizer,
            language="Turkish",
        )

        self.assertEqual(canonical, alias)
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            tokenization.encode_tts_text("encodable text", self.tokenizer, language="xx")


if __name__ == "__main__":
    unittest.main()
