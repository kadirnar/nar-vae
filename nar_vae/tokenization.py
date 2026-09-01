"""Compact, versioned multilingual text conditioning for NAR-VAE.

The frontend deliberately has no learned or pretrained dependency. Reviewed
phonemes use a small shared IPA inventory; normalized orthography uses compact
grapheme tokens where possible and a lossless UTF-8 byte escape everywhere
else. Dataset preparation stores the returned token-language and alignment
masks so text attention and monotonic acoustic alignment do not conflate
control tokens.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple, Protocol

from nar_vae.languages import NULL_LANGUAGE_ID, Language, language_id, normalize_language

TEXT_FRONTEND_NAME = "nar_vae.hybrid_ipa_utf8"
TEXT_FRONTEND_VERSION = 2
TOKENIZER_NAME = f"{TEXT_FRONTEND_NAME}/v{TEXT_FRONTEND_VERSION}"


class TokenEncoder(Protocol):
    """Legacy tokenizer protocol retained for call-signature compatibility."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""


class TextConditioning(NamedTuple):
    """Parallel arrays consumed by dataset preparation and collation."""

    conditioning_ids: list[int]
    token_language_ids: list[int]
    alignment_mask: list[bool]


class TextSpan(NamedTuple):
    """One explicitly language-tagged span, optionally with reviewed phones."""

    text: str
    language: str | Language
    normalized_text: str | None = None
    phonemes: str | Sequence[str] | None = None


_CONTROL_NAMES = (
    "pad",
    "bos",
    "eos",
    "start_speech",
    "end_speech",
    "start_human",
    "end_human",
    "start_ai",
    "end_ai",
    "word_boundary",
    "byte_start",
    "byte_end",
    "explicit_pause",
)

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

# Suprasegmentals affect pronunciation without receiving a mandatory MAS frame.
_STRESS_AND_TONE = (
    "ˈ",
    "ˌ",
    "ː",
    "ˑ",
    "˥",
    "˦",
    "˧",
    "˨",
    "˩",
    "↑",
    "↓",
    "↗",
    "↘",
    "¹",
    "²",
    "³",
    "⁴",
    "⁵",
)

_PUNCTUATION = tuple(dict.fromkeys(".,!?;:…—–-()[]{}\"'“”‘’«»、。，！？：；؟،؛/\\"))

# Unknown or uncommon phones always take the lossless byte path, so this table
# can be extended without creating a data-loss dependency.
_IPA_SYMBOLS = tuple(
    dict.fromkeys(
        (
            "p",
            "b",
            "t",
            "d",
            "ʈ",
            "ɖ",
            "c",
            "ɟ",
            "k",
            "g",
            "q",
            "ɢ",
            "ʔ",
            "m",
            "ɱ",
            "n",
            "ɳ",
            "ɲ",
            "ŋ",
            "ɴ",
            "ʙ",
            "r",
            "ʀ",
            "ɾ",
            "ɽ",
            "ɸ",
            "β",
            "f",
            "v",
            "θ",
            "ð",
            "s",
            "z",
            "ʃ",
            "ʒ",
            "ʂ",
            "ʐ",
            "ç",
            "ʝ",
            "x",
            "ɣ",
            "χ",
            "ʁ",
            "ħ",
            "ʕ",
            "h",
            "ɦ",
            "ɬ",
            "ɮ",
            "ʋ",
            "ɹ",
            "ɻ",
            "j",
            "w",
            "l",
            "ɭ",
            "ʎ",
            "i",
            "y",
            "ɨ",
            "ʉ",
            "ɯ",
            "u",
            "ɪ",
            "ʏ",
            "ʊ",
            "e",
            "ø",
            "ɘ",
            "ɵ",
            "ɤ",
            "o",
            "ə",
            "ɛ",
            "œ",
            "ɜ",
            "ɞ",
            "ʌ",
            "ɔ",
            "æ",
            "ɐ",
            "a",
            "ɶ",
            "ɑ",
            "ɒ",
            "ɚ",
            "ɝ",
            "ɐ̯",
            "t͡s",
            "d͡z",
            "t͡ʃ",
            "d͡ʒ",
            "t͡ɕ",
            "d͡ʑ",
            "ʰ",
            "ʲ",
            "ʷ",
            "ˠ",
            "ˤ",
            "̃",
            "̩",
            "̯",
            "̥",
            "̬",
            "̞",
            "̝",
            "̹",
            "̜",
        )
    )
)

_GRAPHEMES = tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789çğıİöşüÇĞÖŞÜ")


def _build_vocabulary() -> tuple[dict[str, int], dict[int, str]]:
    names: list[str] = []
    names.extend(f"control:{name}" for name in _CONTROL_NAMES)
    names.extend(f"style:<{name}>" for name in _EMOTION_TAG_NAMES)
    names.extend(f"suprasegmental:{token}" for token in _STRESS_AND_TONE)
    names.extend(f"punctuation:{token}" for token in _PUNCTUATION)
    names.extend(f"ipa:{token}" for token in _IPA_SYMBOLS)
    names.extend(f"grapheme:{token}" for token in _GRAPHEMES)
    names.extend(f"byte:{value}" for value in range(256))
    if len(names) != len(set(names)):
        raise RuntimeError("The NAR-VAE text vocabulary contains duplicate entries.")
    token_to_id = {name: index for index, name in enumerate(names)}
    return token_to_id, {index: name for name, index in token_to_id.items()}


_TOKEN_TO_ID, _ID_TO_TOKEN = _build_vocabulary()

PAD_TOKEN = _TOKEN_TO_ID["control:pad"]
START_OF_TEXT = _TOKEN_TO_ID["control:bos"]
END_OF_TEXT = _TOKEN_TO_ID["control:eos"]
START_OF_SPEECH = _TOKEN_TO_ID["control:start_speech"]
END_OF_SPEECH = _TOKEN_TO_ID["control:end_speech"]
START_OF_HUMAN = _TOKEN_TO_ID["control:start_human"]
END_OF_HUMAN = _TOKEN_TO_ID["control:end_human"]
START_OF_AI = _TOKEN_TO_ID["control:start_ai"]
END_OF_AI = _TOKEN_TO_ID["control:end_ai"]
WORD_BOUNDARY = _TOKEN_TO_ID["control:word_boundary"]
BYTE_START = _TOKEN_TO_ID["control:byte_start"]
BYTE_END = _TOKEN_TO_ID["control:byte_end"]
EXPLICIT_PAUSE = _TOKEN_TO_ID["control:explicit_pause"]

EMOTION_TAGS = {f"<{name}>": _TOKEN_TO_ID[f"style:<{name}>"] for name in _EMOTION_TAG_NAMES}
SUPRASEGMENTAL_TOKENS = {
    token: _TOKEN_TO_ID[f"suprasegmental:{token}"] for token in _STRESS_AND_TONE
}
PUNCTUATION_TOKENS = {token: _TOKEN_TO_ID[f"punctuation:{token}"] for token in _PUNCTUATION}
IPA_TOKENS = {token: _TOKEN_TO_ID[f"ipa:{token}"] for token in _IPA_SYMBOLS}
GRAPHEME_TOKENS = {token: _TOKEN_TO_ID[f"grapheme:{token}"] for token in _GRAPHEMES}
BYTE_TOKEN_OFFSET = _TOKEN_TO_ID["byte:0"]
BYTE_TOKENS = tuple(_TOKEN_TO_ID[f"byte:{value}"] for value in range(256))

TOTAL_VOCAB_SIZE = len(_TOKEN_TO_ID)
TOKENIZER_LENGTH = TOTAL_VOCAB_SIZE

_ALIGNMENT_TOKEN_IDS = frozenset(
    (*IPA_TOKENS.values(), *GRAPHEME_TOKENS.values(), *BYTE_TOKENS, EXPLICIT_PAUSE)
)
_EMOTION_PATTERN = re.compile(f"({'|'.join(re.escape(tag) for tag in EMOTION_TAGS)})")
_EXPLICIT_PAUSES = frozenset(("<pause>", "<sil>", "<silence>", "|pause|"))
_PHONE_MATCH_ORDER = tuple(sorted((*IPA_TOKENS, *SUPRASEGMENTAL_TOKENS), key=len, reverse=True))


def token_receives_alignment(token_id: int) -> bool:
    """Return default alignment eligibility when no sequence mask is available.

    The full frontend mask is authoritative: it keeps UTF-8 continuation bytes
    unaligned even though a standalone byte token is conservatively eligible.
    """
    return int(token_id) in _ALIGNMENT_TOKEN_IDS


def _append(
    output_ids: list[int],
    output_languages: list[int],
    output_alignment: list[bool],
    token_id: int,
    token_language: int = NULL_LANGUAGE_ID,
    *,
    aligned: bool | None = None,
) -> None:
    output_ids.append(token_id)
    output_languages.append(token_language)
    output_alignment.append(
        token_receives_alignment(token_id) if aligned is None else bool(aligned)
    )


def _append_utf8(
    value: str,
    language: int,
    output_ids: list[int],
    output_languages: list[int],
    output_alignment: list[bool],
) -> None:
    if not value:
        return
    _append(output_ids, output_languages, output_alignment, BYTE_START)
    # UTF-8 is a lossless text representation, not an acoustic segmentation.
    # Give each approximate Unicode grapheme cluster one MAS anchor while the
    # remaining bytes stay visible to the bidirectional text encoder. This
    # avoids forcing three or four minimum-duration frames onto every CJK or
    # other non-Latin character merely because of its byte width.
    clusters: list[str] = []
    for character in value:
        continues_cluster = bool(clusters) and (
            unicodedata.category(character).startswith("M")
            or character == "\u200d"
            or clusters[-1].endswith("\u200d")
        )
        if continues_cluster:
            clusters[-1] += character
        else:
            clusters.append(character)
    for cluster in clusters:
        for byte_index, byte in enumerate(cluster.encode("utf-8", errors="strict")):
            _append(
                output_ids,
                output_languages,
                output_alignment,
                BYTE_TOKEN_OFFSET + byte,
                language,
                aligned=byte_index == 0,
            )
    _append(output_ids, output_languages, output_alignment, BYTE_END)


def decode_utf8_fallback(token_ids: Sequence[int]) -> str:
    """Decode all complete byte escapes in ``token_ids``."""
    decoded: list[str] = []
    pending: list[int] | None = None
    for raw_id in token_ids:
        token_id = int(raw_id)
        if token_id == BYTE_START:
            if pending is not None:
                raise ValueError("Nested UTF-8 byte escapes are invalid.")
            pending = []
        elif token_id == BYTE_END:
            if pending is None:
                raise ValueError("UTF-8 byte escape ended without a start token.")
            decoded.append(bytes(pending).decode("utf-8", errors="strict"))
            pending = None
        elif pending is not None:
            byte = token_id - BYTE_TOKEN_OFFSET
            if not 0 <= byte <= 255:
                raise ValueError("A UTF-8 byte escape contains a non-byte token.")
            pending.append(byte)
    if pending is not None:
        raise ValueError("Unterminated UTF-8 byte escape.")
    return "".join(decoded)


def _append_word_boundary(
    output_ids: list[int],
    output_languages: list[int],
    output_alignment: list[bool],
) -> None:
    if output_ids and output_ids[-1] not in (START_OF_TEXT, WORD_BOUNDARY):
        _append(output_ids, output_languages, output_alignment, WORD_BOUNDARY)


def _append_text_content(
    text: str,
    language: int,
    output_ids: list[int],
    output_languages: list[int],
    output_alignment: list[bool],
    *,
    parse_emotion_tags: bool,
) -> None:
    normalized = unicodedata.normalize("NFKC", text)
    parts = _EMOTION_PATTERN.split(normalized) if parse_emotion_tags else (normalized,)
    for part in parts:
        if not part:
            continue
        emotion_id = EMOTION_TAGS.get(part) if parse_emotion_tags else None
        if emotion_id is not None:
            _append(output_ids, output_languages, output_alignment, emotion_id)
            continue

        pending_bytes: list[str] = []

        def flush_bytes() -> None:
            if pending_bytes:
                _append_utf8(
                    "".join(pending_bytes),
                    language,
                    output_ids,
                    output_languages,
                    output_alignment,
                )
                pending_bytes.clear()

        for character in part:
            if character.isspace():
                flush_bytes()
                _append_word_boundary(output_ids, output_languages, output_alignment)
                continue
            punctuation_id = PUNCTUATION_TOKENS.get(character)
            if punctuation_id is not None:
                flush_bytes()
                _append(output_ids, output_languages, output_alignment, punctuation_id)
                continue
            grapheme_id = GRAPHEME_TOKENS.get(character)
            if grapheme_id is not None:
                flush_bytes()
                _append(
                    output_ids,
                    output_languages,
                    output_alignment,
                    grapheme_id,
                    language,
                    aligned=True,
                )
                continue
            pending_bytes.append(character)
        flush_bytes()


def _phone_units(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        chunks = re.findall(r"<[^>]+>|\||[^\s|]+", unicodedata.normalize("NFD", value))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        chunks = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError("phonemes must contain non-empty strings.")
            chunks.append(unicodedata.normalize("NFD", item))
    else:
        raise TypeError("phonemes must be a string or a sequence of strings.")

    units: list[str] = []
    for chunk in chunks:
        if chunk == "|" or chunk in _EXPLICIT_PAUSES:
            units.append(chunk)
            continue
        offset = 0
        while offset < len(chunk):
            match = next(
                (
                    candidate
                    for candidate in _PHONE_MATCH_ORDER
                    if chunk.startswith(candidate, offset)
                ),
                None,
            )
            if match is not None:
                units.append(match)
                offset += len(match)
                continue
            end = offset + 1
            while end < len(chunk) and unicodedata.combining(chunk[end]):
                end += 1
            units.append(unicodedata.normalize("NFC", chunk[offset:end]))
            offset = end
    return units


def _append_phone_content(
    phonemes: str | Sequence[str],
    language: int,
    output_ids: list[int],
    output_languages: list[int],
    output_alignment: list[bool],
) -> None:
    for unit in _phone_units(phonemes):
        if unit == "|":
            _append_word_boundary(output_ids, output_languages, output_alignment)
            continue
        if unit in _EXPLICIT_PAUSES:
            _append(
                output_ids,
                output_languages,
                output_alignment,
                EXPLICIT_PAUSE,
                aligned=True,
            )
            continue
        phone_id = IPA_TOKENS.get(unit)
        if phone_id is not None:
            _append(
                output_ids,
                output_languages,
                output_alignment,
                phone_id,
                language,
                aligned=True,
            )
            continue
        suprasegmental_id = SUPRASEGMENTAL_TOKENS.get(unit)
        if suprasegmental_id is not None:
            _append(output_ids, output_languages, output_alignment, suprasegmental_id, language)
            continue
        _append_utf8(
            unit,
            language,
            output_ids,
            output_languages,
            output_alignment,
        )


def _normalize_span(value: TextSpan | Mapping[str, Any]) -> TextSpan:
    if isinstance(value, TextSpan):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("language_spans must contain TextSpan values or mappings.")
    text = value.get("text", "")
    if not isinstance(text, str):
        raise TypeError("Every language span text value must be a string.")
    if "language" not in value:
        raise ValueError("Every language span must declare its language.")
    normalized_text = value.get("normalized_text")
    if normalized_text is not None and not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string or null.")
    return TextSpan(
        text=text,
        language=value["language"],
        normalized_text=normalized_text,
        phonemes=value.get("phonemes"),
    )


def encode_tts_conditioning(
    text: str | None,
    *,
    normalized_text: str | None = None,
    phonemes: str | Sequence[str] | None = None,
    language: str | Language | None = None,
    language_spans: Sequence[TextSpan | Mapping[str, Any]] | None = None,
    parse_emotion_tags: bool = True,
    vocab_size: int | None = None,
) -> TextConditioning:
    """Encode an utterance into compact tokens and parallel training masks."""
    target_language = normalize_language(language)
    if normalized_text is not None and not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string or null.")
    if text is not None and not isinstance(text, str):
        raise TypeError(f"text must be a string or null, received {type(text).__name__}")

    if language_spans is not None:
        if isinstance(language_spans, (str, bytes)) or not isinstance(language_spans, Sequence):
            raise TypeError("language_spans must be a sequence.")
        spans = [_normalize_span(span) for span in language_spans]
        if not spans:
            raise ValueError("language_spans must not be empty.")
        if normalized_text is not None or phonemes is not None:
            raise ValueError(
                "Top-level normalized_text/phonemes cannot be combined with language_spans."
            )
    else:
        source_text = text or ""
        if (
            not source_text.strip()
            and not (isinstance(normalized_text, str) and normalized_text.strip())
            and phonemes is None
        ):
            raise ValueError("text, normalized_text, or phonemes must contain content.")
        spans = [
            TextSpan(
                text=source_text,
                language=target_language,
                normalized_text=normalized_text,
                phonemes=phonemes,
            )
        ]

    token_ids: list[int] = []
    token_languages: list[int] = []
    alignment_mask: list[bool] = []
    _append(token_ids, token_languages, alignment_mask, START_OF_TEXT)

    for index, span in enumerate(spans):
        span_language = language_id(span.language)
        if index:
            _append_word_boundary(token_ids, token_languages, alignment_mask)
        if span.phonemes is not None:
            _append_phone_content(
                span.phonemes,
                span_language,
                token_ids,
                token_languages,
                alignment_mask,
            )
        else:
            source = span.normalized_text if span.normalized_text is not None else span.text
            _append_text_content(
                source,
                span_language,
                token_ids,
                token_languages,
                alignment_mask,
                parse_emotion_tags=parse_emotion_tags,
            )

    while token_ids and token_ids[-1] == WORD_BOUNDARY:
        token_ids.pop()
        token_languages.pop()
        alignment_mask.pop()
    _append(token_ids, token_languages, alignment_mask, END_OF_TEXT)

    if not any(alignment_mask):
        raise ValueError("The text frontend produced no alignable speech-content tokens.")
    if not (len(token_ids) == len(token_languages) == len(alignment_mask)):
        raise RuntimeError("Text conditioning arrays must have identical lengths.")
    if vocab_size is not None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer.")
        invalid_ids = sorted({token_id for token_id in token_ids if not 0 <= token_id < vocab_size})
        if invalid_ids:
            raise ValueError(
                f"Token IDs {invalid_ids} exceed the checkpoint vocabulary of {vocab_size} entries."
            )
    return TextConditioning(token_ids, token_languages, alignment_mask)


def encode_tts_text(
    text: str,
    tokenizer: TokenEncoder | None = None,
    *,
    parse_emotion_tags: bool = True,
    vocab_size: int | None = None,
    language: str | Language | None = None,
    normalized_text: str | None = None,
    phonemes: str | Sequence[str] | None = None,
    language_spans: Sequence[TextSpan | Mapping[str, Any]] | None = None,
) -> list[int]:
    """Return only token IDs while preserving the historical public API."""
    del tokenizer
    return encode_tts_conditioning(
        text,
        normalized_text=normalized_text,
        phonemes=phonemes,
        language=language,
        language_spans=language_spans,
        parse_emotion_tags=parse_emotion_tags,
        vocab_size=vocab_size,
    ).conditioning_ids


__all__ = [
    "BYTE_END",
    "BYTE_START",
    "BYTE_TOKEN_OFFSET",
    "BYTE_TOKENS",
    "EMOTION_TAGS",
    "END_OF_AI",
    "END_OF_HUMAN",
    "END_OF_SPEECH",
    "END_OF_TEXT",
    "EXPLICIT_PAUSE",
    "GRAPHEME_TOKENS",
    "IPA_TOKENS",
    "PAD_TOKEN",
    "PUNCTUATION_TOKENS",
    "START_OF_AI",
    "START_OF_HUMAN",
    "START_OF_SPEECH",
    "START_OF_TEXT",
    "SUPRASEGMENTAL_TOKENS",
    "TEXT_FRONTEND_NAME",
    "TEXT_FRONTEND_VERSION",
    "TOKENIZER_LENGTH",
    "TOKENIZER_NAME",
    "TOTAL_VOCAB_SIZE",
    "TextConditioning",
    "TextSpan",
    "TokenEncoder",
    "WORD_BOUNDARY",
    "decode_utf8_fallback",
    "encode_tts_conditioning",
    "encode_tts_text",
    "token_receives_alignment",
]
