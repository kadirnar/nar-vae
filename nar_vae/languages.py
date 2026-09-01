"""Canonical language metadata shared by inference, training, and evaluation.

Language IDs are stable checkpoint data. Append new entries to ``LANGUAGES``;
never reorder or remove existing entries without incrementing the registry
version and providing a checkpoint migration.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

LANGUAGE_CONDITIONING_VERSION = 1
LANGUAGE_REGISTRY_VERSION = 1
DEFAULT_LANGUAGE = "en"
NULL_LANGUAGE_ID = 0


class UnsupportedLanguageError(ValueError):
    """Raised when a language is not represented by the stable registry."""


class MultilingualUnsupportedError(RuntimeError):
    """Raised when checkpoint weights do not support a requested target language."""


class CrossLingualUnsupportedError(MultilingualUnsupportedError):
    """Raised when checkpoint metadata does not support a source/target pair."""


@dataclass(frozen=True, slots=True)
class Language:
    """One stable language-conditioning entry."""

    code: str
    name: str
    script: str
    aliases: tuple[str, ...] = ()


# IDs are one-based by position; zero is reserved for classifier-free guidance.
LANGUAGES = (
    Language("en", "English", "Latn", ("eng", "en-US", "en-GB")),
    Language("ar", "Arabic", "Arab", ("ara",)),
    Language("cs", "Czech", "Latn", ("ces", "cze")),
    Language("de", "German", "Latn", ("deu", "ger")),
    Language("es", "Spanish", "Latn", ("spa", "es-ES", "es-MX")),
    Language("fr", "French", "Latn", ("fra", "fre", "fr-FR")),
    Language("hi", "Hindi", "Deva", ("hin",)),
    Language("hu", "Hungarian", "Latn", ("hun",)),
    Language("it", "Italian", "Latn", ("ita",)),
    Language("ja", "Japanese", "Jpan", ("jpn",)),
    Language("ko", "Korean", "Kore", ("kor",)),
    Language("nl", "Dutch", "Latn", ("nld", "dut")),
    Language("pl", "Polish", "Latn", ("pol",)),
    Language("pt", "Portuguese", "Latn", ("por", "pt-BR", "pt-PT")),
    Language("ru", "Russian", "Cyrl", ("rus",)),
    Language("tr", "Turkish", "Latn", ("tur",)),
    Language("zh-Hans", "Chinese (Simplified)", "Hans", ("zh", "cmn", "zh-CN", "zh-SG")),
    Language("zh-Hant", "Chinese (Traditional)", "Hant", ("zh-TW", "zh-HK")),
)

LANGUAGE_COUNT = len(LANGUAGES)
LANGUAGE_BY_CODE = {language.code: language for language in LANGUAGES}
LANGUAGE_ID_BY_CODE = {language.code: index for index, language in enumerate(LANGUAGES, start=1)}
LANGUAGE_BY_ID = {index: language for index, language in enumerate(LANGUAGES, start=1)}


def _lookup_key(value: str) -> str:
    return value.strip().replace("_", "-").casefold()


_LANGUAGE_ALIASES = {
    _lookup_key(alias): language.code
    for language in LANGUAGES
    for alias in (language.code, language.name, *language.aliases)
}


def normalize_language(value: str | Language | None) -> str:
    """Return one canonical BCP 47-style code from a code or known alias."""
    if value is None:
        return DEFAULT_LANGUAGE
    if isinstance(value, Language):
        return value.code
    if not isinstance(value, str):
        raise TypeError(f"language must be a string, received {type(value).__name__}")
    key = _lookup_key(value)
    if not key:
        raise UnsupportedLanguageError("language must not be empty")
    try:
        return _LANGUAGE_ALIASES[key]
    except KeyError as exc:
        subtags = key.split("-")
        for end in range(len(subtags) - 1, 1, -1):
            registered_prefix = _LANGUAGE_ALIASES.get("-".join(subtags[:end]))
            if registered_prefix is not None:
                return registered_prefix
        primary_language = subtags[0]
        if primary_language in LANGUAGE_BY_CODE:
            return primary_language
        choices = ", ".join(LANGUAGE_BY_CODE)
        raise UnsupportedLanguageError(
            f"Unsupported language {value!r}. Registered language codes: {choices}."
        ) from exc


def language_id(value: str | Language | None) -> int:
    """Return the stable one-based conditioning ID for a language."""
    return LANGUAGE_ID_BY_CODE[normalize_language(value)]


def language_from_id(value: int) -> Language:
    """Resolve a stable checkpoint ID to its language metadata."""
    try:
        return LANGUAGE_BY_ID[int(value)]
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedLanguageError(
            f"Unsupported language ID {value!r}; expected 1 through {LANGUAGE_COUNT}."
        ) from exc


def normalize_languages(
    values: Iterable[str | Language] | str | Language | None,
) -> tuple[str, ...]:
    """Normalize and de-duplicate an ordered language collection."""
    if values is None:
        return (DEFAULT_LANGUAGE,)
    if isinstance(values, (str, Language)):
        values = (values,)
    normalized = tuple(dict.fromkeys(normalize_language(value) for value in values))
    if not normalized:
        raise ValueError("supported_languages must contain at least one language")
    return normalized


@dataclass(frozen=True, slots=True)
class LanguagePair:
    """Target-text and source-reference languages for one synthesis request."""

    target: str
    reference: str | None = None

    @classmethod
    def resolve(
        cls,
        target: str | Language | None,
        reference: str | Language | None,
        *,
        has_reference: bool,
    ) -> "LanguagePair":
        target_code = normalize_language(target)
        if not has_reference:
            if reference is not None:
                raise ValueError("reference_language requires reference_audio or speaker_latent")
            return cls(target=target_code)
        reference_code = normalize_language(reference if reference is not None else target_code)
        return cls(target=target_code, reference=reference_code)

    @property
    def is_cross_lingual(self) -> bool:
        return self.reference is not None and self.reference != self.target

    def as_tuple(self) -> tuple[str, str]:
        """Return the complete pair used by training/checkpoint capability metadata."""
        if self.reference is None:
            raise ValueError("A supported language pair must include a reference language.")
        return self.target, self.reference


def normalize_language_pairs(
    values: Iterable[LanguagePair | Sequence[str | Language]] | LanguagePair | None,
) -> tuple[LanguagePair, ...]:
    """Normalize and de-duplicate an ordered collection of complete language pairs."""
    if values is None:
        return ()
    if isinstance(values, LanguagePair):
        values = (values,)

    normalized: list[LanguagePair] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(values):
        if isinstance(value, LanguagePair):
            if value.reference is None:
                raise ValueError(
                    f"supported_language_pairs[{index}] must include a reference language"
                )
            pair = LanguagePair(
                target=normalize_language(value.target),
                reference=normalize_language(value.reference),
            )
        else:
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(
                    f"supported_language_pairs[{index}] must be a two-language sequence"
                )
            if len(value) != 2:
                raise ValueError(
                    f"supported_language_pairs[{index}] must contain target and reference"
                )
            pair = LanguagePair(
                target=normalize_language(value[0]),
                reference=normalize_language(value[1]),
            )
        key = pair.as_tuple()
        if key not in seen:
            normalized.append(pair)
            seen.add(key)
    if not normalized:
        raise ValueError("supported_language_pairs must contain at least one language pair")
    return tuple(normalized)


def resolve_language_pair_support(
    supported_languages: Iterable[str | Language] | str | Language | None,
    *,
    supported_reference_languages: Iterable[str | Language] | str | Language | None = None,
    supported_language_pairs: (
        Iterable[LanguagePair | Sequence[str | Language]] | LanguagePair | None
    ) = None,
) -> tuple[tuple[str, ...], tuple[LanguagePair, ...]]:
    """Resolve exact pair coverage, with reference languages as a legacy shorthand.

    When ``supported_language_pairs`` is supplied it is authoritative. Otherwise,
    the legacy reference-language collection expands to its Cartesian product with
    every supported target language.
    """
    targets = normalize_languages(supported_languages)
    target_set = set(targets)
    if supported_language_pairs is not None:
        pairs = normalize_language_pairs(supported_language_pairs)
        undeclared_targets = tuple(
            dict.fromkeys(pair.target for pair in pairs if pair.target not in target_set)
        )
        if undeclared_targets:
            raise ValueError(
                "supported_language_pairs contains target languages outside "
                f"supported_languages: {undeclared_targets!r}"
            )
        references = tuple(dict.fromkeys(pair.reference for pair in pairs))
        return references, pairs

    if supported_reference_languages is None:
        return (), ()
    references = normalize_languages(supported_reference_languages)
    pairs = tuple(
        LanguagePair(target=target, reference=reference)
        for target in targets
        for reference in references
    )
    return references, pairs


__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LANGUAGE_BY_CODE",
    "LANGUAGE_BY_ID",
    "LANGUAGE_CONDITIONING_VERSION",
    "LANGUAGE_COUNT",
    "LANGUAGE_ID_BY_CODE",
    "LANGUAGE_REGISTRY_VERSION",
    "CrossLingualUnsupportedError",
    "Language",
    "LanguagePair",
    "MultilingualUnsupportedError",
    "NULL_LANGUAGE_ID",
    "UnsupportedLanguageError",
    "language_from_id",
    "language_id",
    "normalize_language",
    "normalize_language_pairs",
    "normalize_languages",
    "resolve_language_pair_support",
]
