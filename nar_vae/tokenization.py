"""Shared tokenizer constants and text formatting for NAR-VAE."""

from __future__ import annotations

import re
from typing import Protocol

from nar_vae.languages import Language, normalize_language

TOKENIZER_NAME = "cl100k_base"
TOKENIZER_LENGTH = 100277

START_OF_TEXT = TOKENIZER_LENGTH + 1
END_OF_TEXT = TOKENIZER_LENGTH + 2
START_OF_SPEECH = TOKENIZER_LENGTH + 3
END_OF_SPEECH = TOKENIZER_LENGTH + 4
START_OF_HUMAN = TOKENIZER_LENGTH + 5
END_OF_HUMAN = TOKENIZER_LENGTH + 6
START_OF_AI = TOKENIZER_LENGTH + 7
END_OF_AI = TOKENIZER_LENGTH + 8
PAD_TOKEN = TOKENIZER_LENGTH + 9

_EMOTION_TAG_NAMES = (
    "laugh",
    "chuckle",
    "sigh",
    "gasp",
    "cough",
    "clear_throat",
    "sniffle",
    "groan",
    "yawn",
    "cry",
    "sob",
    "scream",
    "whisper",
    "shout",
    "mumble",
    "hum",
    "sing",
    "breath",
    "exhale",
    "inhale",
    "pause",
    "silence",
    "applause",
    "music",
    "noise",
)
EMOTION_TAGS = {
    f"<{name}>": TOKENIZER_LENGTH + 10 + index for index, name in enumerate(_EMOTION_TAG_NAMES)
}
TOTAL_VOCAB_SIZE = max(EMOTION_TAGS.values()) + 1

_EMOTION_PATTERN = re.compile(f"({'|'.join(re.escape(tag) for tag in EMOTION_TAGS)})")


class TokenEncoder(Protocol):
    """Small subset of the tiktoken encoder API used by NAR-VAE."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""


def encode_tts_text(
    text: str,
    tokenizer: TokenEncoder,
    *,
    parse_emotion_tags: bool = True,
    vocab_size: int | None = None,
    language: str | Language | None = None,
) -> list[int]:
    """Encode text and wrap it in the token sequence expected by EchoDiT.

    Emotion tags are encoded as ordinary text when the loaded checkpoint does
    not contain their optional embedding rows.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, received {type(text).__name__}")
    # Token IDs are language-neutral, but validation belongs at this boundary so
    # an encodable Unicode string cannot silently bypass the language registry.
    normalize_language(language)

    if parse_emotion_tags:
        text_tokens: list[int] = []
        for part in _EMOTION_PATTERN.split(text):
            if not part:
                continue
            emotion_token = EMOTION_TAGS.get(part)
            if emotion_token is not None and (vocab_size is None or emotion_token < vocab_size):
                text_tokens.append(emotion_token)
            else:
                text_tokens.extend(tokenizer.encode(part))
    else:
        text_tokens = tokenizer.encode(text)

    encoded = [
        START_OF_HUMAN,
        *text_tokens,
        END_OF_HUMAN,
        START_OF_AI,
        START_OF_SPEECH,
    ]
    if vocab_size is not None:
        invalid_ids = [token_id for token_id in encoded if not 0 <= token_id < vocab_size]
        if invalid_ids:
            raise ValueError(
                f"Token IDs {sorted(set(invalid_ids))} exceed the checkpoint vocabulary "
                f"of {vocab_size} entries."
            )
    return encoded


__all__ = [
    "EMOTION_TAGS",
    "END_OF_AI",
    "END_OF_HUMAN",
    "END_OF_SPEECH",
    "END_OF_TEXT",
    "PAD_TOKEN",
    "START_OF_AI",
    "START_OF_HUMAN",
    "START_OF_SPEECH",
    "START_OF_TEXT",
    "TOKENIZER_LENGTH",
    "TOKENIZER_NAME",
    "TOTAL_VOCAB_SIZE",
    "TokenEncoder",
    "encode_tts_text",
]
