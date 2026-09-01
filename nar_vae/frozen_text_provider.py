"""Pinned Hugging Face text features outside the acoustic model.

The provider owns the heavyweight tokenizer/backbone only while preparing a
dataset or an inference request.  ``FlowMatchingEchoDiT`` receives the returned
CPU cache tensors and therefore never registers, saves, trains, or distributes
the Hugging Face model.

``hf_non_special_tokens_v1`` uses the provider tokenizer's exact token axis.
Hugging Face BOS/EOS tokens remain visible to contextual attention but cannot
own MAS frames. In the XPhone-compatible phoneme frontend, only caller-supplied
phones, punctuation, and the native ``▁`` word boundary enter the backbone.
There are deliberately no invented control phones: unsupported style/control
tags and unknown provider tokens fail closed instead of collapsing to ``<unk>``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn

from nar_vae.languages import NULL_LANGUAGE_ID, Language, language_id, normalize_language
from nar_vae.tokenization import TextSpan

FROZEN_TEXT_ALIGNMENT_POLICY = "hf_non_special_tokens_v1"
FROZEN_TEXT_CONFIG_FILENAME = "config.json"
FROZEN_TEXT_REPRESENTATION_NAME = "nar_vae.frozen_text_provider"
FROZEN_TEXT_REPRESENTATION_VERSION = 1

_HUB_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FRONTENDS = frozenset({"raw_text", "phonemes"})
_CACHE_DTYPES = MappingProxyType(
    {
        "float16": torch.float16,
        "float32": torch.float32,
    }
)
_CACHE_DTYPE_NAMES = MappingProxyType({value: key for key, value in _CACHE_DTYPES.items()})
_CONTROL_TAG = re.compile(r"<[^>]+>")
_PHONEME_WORD_BOUNDARIES = frozenset({"|", "_", "▁"})
_PHONEME_NON_ALIGNING_UNITS = frozenset(
    {
        "▁",
        ".",
        ",",
        ";",
        ":",
        "?",
        "!",
        "…",
        "—",
        "–",
        "-",
        "ˈ",
        "ˌ",
        "ː",
        "ˑ",
        "˥",
        "˦",
        "˧",
        "˨",
        "˩",
    }
)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _hub_id(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or value.count("/") != 1
        or not all(value.split("/"))
    ):
        raise ValueError(f"{name} must use the non-empty 'owner/name' Hub format.")
    return value


def _commit(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _HUB_COMMIT.fullmatch(value):
        raise ValueError(f"{name} must be a full 40-character Hub commit.")
    return value.lower()


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256.")
    return value


def _relative_filename(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty repository-relative filename.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."}:
        raise ValueError(f"{name} must be a non-empty repository-relative filename.")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class FrozenTextProviderSpec:
    """Immutable provider/cache identity using the training configuration names."""

    text_conditioning_mode: str
    text_vocab_size: int
    pad_token: int
    conditioning_feature_size: int
    conditioning_feature_dtype: str
    frozen_text_alignment: str
    frozen_text_cache_version: int
    frozen_text_config_sha256: str
    frozen_text_encoder_id: str
    frozen_text_encoder_revision: str
    frozen_text_frontend: str
    frozen_text_hidden_layer: int
    frozen_text_model_filename: str
    frozen_text_model_sha256: str
    frozen_text_tokenizer_filename: str
    frozen_text_tokenizer_id: str
    frozen_text_tokenizer_revision: str
    frozen_text_tokenizer_sha256: str

    def __post_init__(self) -> None:
        if self.text_conditioning_mode != "frozen_features":
            raise ValueError(
                "FrozenTextProviderSpec requires text_conditioning_mode='frozen_features'."
            )
        _positive_integer(
            self.conditioning_feature_size,
            name="conditioning_feature_size",
        )
        _positive_integer(self.text_vocab_size, name="text_vocab_size")
        if (
            isinstance(self.pad_token, bool)
            or not isinstance(self.pad_token, int)
            or not 0 <= self.pad_token < self.text_vocab_size
        ):
            raise ValueError("pad_token must be an integer within text_vocab_size.")
        if self.conditioning_feature_dtype not in _CACHE_DTYPES:
            raise ValueError(
                "conditioning_feature_dtype must be float16 or float32; Arrow does not "
                "losslessly preserve bfloat16 cache rows."
            )
        if self.frozen_text_alignment != FROZEN_TEXT_ALIGNMENT_POLICY:
            raise ValueError(
                "frozen_text_alignment must use the versioned "
                f"{FROZEN_TEXT_ALIGNMENT_POLICY!r} policy."
            )
        _positive_integer(self.frozen_text_cache_version, name="frozen_text_cache_version")
        if self.frozen_text_frontend not in _FRONTENDS:
            raise ValueError("frozen_text_frontend must be 'phonemes' or 'raw_text'.")
        if isinstance(self.frozen_text_hidden_layer, bool) or not isinstance(
            self.frozen_text_hidden_layer,
            int,
        ):
            raise ValueError("frozen_text_hidden_layer must be an integer layer index.")

        normalized = {
            "frozen_text_encoder_id": _hub_id(
                self.frozen_text_encoder_id,
                name="frozen_text_encoder_id",
            ),
            "frozen_text_tokenizer_id": _hub_id(
                self.frozen_text_tokenizer_id,
                name="frozen_text_tokenizer_id",
            ),
            "frozen_text_encoder_revision": _commit(
                self.frozen_text_encoder_revision,
                name="frozen_text_encoder_revision",
            ),
            "frozen_text_tokenizer_revision": _commit(
                self.frozen_text_tokenizer_revision,
                name="frozen_text_tokenizer_revision",
            ),
            "frozen_text_config_sha256": _sha256(
                self.frozen_text_config_sha256,
                name="frozen_text_config_sha256",
            ),
            "frozen_text_model_sha256": _sha256(
                self.frozen_text_model_sha256,
                name="frozen_text_model_sha256",
            ),
            "frozen_text_tokenizer_sha256": _sha256(
                self.frozen_text_tokenizer_sha256,
                name="frozen_text_tokenizer_sha256",
            ),
            "frozen_text_model_filename": _relative_filename(
                self.frozen_text_model_filename,
                name="frozen_text_model_filename",
            ),
            "frozen_text_tokenizer_filename": _relative_filename(
                self.frozen_text_tokenizer_filename,
                name="frozen_text_tokenizer_filename",
            ),
        }
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FrozenTextProviderSpec":
        """Build the exact provider identity from a validated training configuration."""
        if not isinstance(config, Mapping):
            raise TypeError("Frozen text provider configuration must be a mapping.")
        fields = tuple(cls.__dataclass_fields__)
        missing = sorted(name for name in fields if name not in config)
        if missing:
            raise ValueError(f"Frozen text provider configuration is missing: {missing}.")
        return cls(**{name: config[name] for name in fields})

    @classmethod
    def from_manifest(cls, manifest: Any) -> "FrozenTextProviderSpec":
        """Build from a model manifest or its ``text_conditioning`` section.

        The manifest uses concise public field names while training YAML uses
        ``frozen_text_*`` names. This method is the sole deterministic mapping
        between those representations.
        """
        if isinstance(manifest, Mapping):
            section = manifest.get("text_conditioning", manifest)
        else:
            section = getattr(manifest, "text_conditioning", None)
        if not isinstance(section, Mapping):
            raise TypeError(
                "Frozen text manifest input must be a mapping, a manifest mapping, "
                "or an object with a text_conditioning mapping."
            )
        manifest_fields = {
            "text_conditioning_mode": "mode",
            "text_vocab_size": "provider_vocab_size",
            "pad_token": "provider_pad_token",
            "conditioning_feature_size": "feature_size",
            "conditioning_feature_dtype": "feature_dtype",
            "frozen_text_alignment": "alignment",
            "frozen_text_cache_version": "cache_version",
            "frozen_text_config_sha256": "config_sha256",
            "frozen_text_encoder_id": "encoder_id",
            "frozen_text_encoder_revision": "encoder_revision",
            "frozen_text_frontend": "frontend",
            "frozen_text_hidden_layer": "hidden_layer",
            "frozen_text_model_filename": "model_filename",
            "frozen_text_model_sha256": "model_sha256",
            "frozen_text_tokenizer_filename": "tokenizer_filename",
            "frozen_text_tokenizer_id": "tokenizer_id",
            "frozen_text_tokenizer_revision": "tokenizer_revision",
            "frozen_text_tokenizer_sha256": "tokenizer_sha256",
        }
        missing = sorted(value for value in manifest_fields.values() if value not in section)
        if missing:
            raise ValueError(f"Frozen text manifest section is missing: {missing}.")
        return cls(
            **{
                config_name: section[manifest_name]
                for config_name, manifest_name in manifest_fields.items()
            }
        )

    @property
    def cache_dtype(self) -> torch.dtype:
        return _CACHE_DTYPES[self.conditioning_feature_dtype]

    def to_config(self) -> dict[str, Any]:
        """Return an independent mapping suitable for a run or model manifest."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def contract_sha256(self) -> str:
        """Canonical identity stored with every prepared frozen-feature row."""
        payload = json.dumps(
            self.to_config(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedFrozenTextArtifacts:
    """Hash-verified local paths used to construct one provider."""

    config_path: Path
    model_path: Path
    tokenizer_path: Path


@dataclass(frozen=True, slots=True)
class FrozenTextConditioning:
    """One unbatched, CPU-resident cache row on the provider token axis."""

    conditioning_ids: torch.Tensor
    conditioning_features: torch.Tensor
    conditioning_mask: torch.Tensor
    token_language_ids: torch.Tensor
    alignment_mask: torch.Tensor
    rendered_text: str
    target_language_id: int
    cache_version: int
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.conditioning_ids.ndim != 1 or self.conditioning_ids.dtype != torch.long:
            raise ValueError("conditioning_ids must be a one-dimensional torch.long tensor.")
        length = self.conditioning_ids.numel()
        if length <= 0:
            raise ValueError("Frozen text conditioning cannot be empty.")
        if self.conditioning_features.ndim != 2 or self.conditioning_features.shape[0] != length:
            raise ValueError("conditioning_features must have shape [provider_token, feature].")
        if not torch.is_floating_point(self.conditioning_features):
            raise TypeError("conditioning_features must use a floating-point dtype.")
        if self.conditioning_features.dtype not in _CACHE_DTYPE_NAMES:
            raise TypeError(
                "conditioning_features cache dtype must be torch.float16 or torch.float32."
            )
        for name in ("conditioning_mask", "token_language_ids", "alignment_mask"):
            value = getattr(self, name)
            if value.ndim != 1 or value.numel() != length:
                raise ValueError(f"{name} must share the provider token axis.")
        if self.conditioning_mask.dtype != torch.bool or self.alignment_mask.dtype != torch.bool:
            raise TypeError("conditioning_mask and alignment_mask must be boolean tensors.")
        if self.token_language_ids.dtype != torch.long:
            raise TypeError("token_language_ids must use torch.long.")
        if bool((self.alignment_mask & ~self.conditioning_mask).any()):
            raise ValueError("alignment_mask cannot select a masked provider token.")
        if not bool(self.alignment_mask.any()):
            raise ValueError("At least one provider token must be eligible for MAS alignment.")
        if not bool(torch.isfinite(self.conditioning_features).all()):
            raise ValueError("conditioning_features must contain only finite values.")
        _positive_integer(self.target_language_id, name="target_language_id")
        _positive_integer(self.cache_version, name="cache_version")
        _sha256(self.contract_sha256, name="contract_sha256")
        if not isinstance(self.rendered_text, str) or not self.rendered_text:
            raise ValueError("rendered_text must record the non-empty provider input.")
        for tensor in (
            self.conditioning_ids,
            self.conditioning_features,
            self.conditioning_mask,
            self.token_language_ids,
            self.alignment_mask,
        ):
            if tensor.device.type != "cpu":
                raise ValueError("Frozen text cache rows must be CPU-resident.")

    @property
    def attention_mask(self) -> torch.Tensor:
        """Alias used by direct model inference APIs."""
        return self.conditioning_mask

    def to_cache_row(self) -> dict[str, Any]:
        """Serialize fields accepted by dataset collation without losing cache dtype."""
        return {
            "conditioning_ids": self.conditioning_ids.tolist(),
            "conditioning_features": self.conditioning_features.clone(),
            "conditioning_mask": self.conditioning_mask.tolist(),
            "token_language_ids": self.token_language_ids.tolist(),
            "alignment_mask": self.alignment_mask.tolist(),
            "language_id": self.target_language_id,
            "frozen_text_cache_version": self.cache_version,
            "frozen_text_contract_sha256": self.contract_sha256,
            "conditioning_feature_dtype": _CACHE_DTYPE_NAMES[self.conditioning_features.dtype],
        }

    def as_model_inputs(
        self,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a batch-of-one mapping for ``FlowMatchingEchoDiT`` inference."""
        destination = torch.device("cpu") if device is None else torch.device(device)
        return {
            "conditioning_ids": self.conditioning_ids.to(destination)[None, :],
            "conditioning_features": self.conditioning_features.to(destination)[None, :, :],
            "attention_mask": self.conditioning_mask.to(destination)[None, :],
            "token_language_ids": self.token_language_ids.to(destination)[None, :],
            "alignment_mask": self.alignment_mask.to(destination)[None, :],
            "language_ids": torch.tensor(
                [self.target_language_id],
                dtype=torch.long,
                device=destination,
            ),
        }


@dataclass(frozen=True, slots=True)
class _RenderedSegment:
    start: int
    stop: int
    language_id: int
    alignable: bool
    kind: str


@dataclass(frozen=True, slots=True)
class _RenderedInput:
    text: str
    segments: tuple[_RenderedSegment, ...]
    target_language_id: int


class _RenderedInputBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._segments: list[_RenderedSegment] = []
        self._length = 0

    def add(
        self,
        value: str,
        *,
        language: int = NULL_LANGUAGE_ID,
        alignable: bool = False,
        kind: str,
    ) -> None:
        if not value:
            return
        if self._parts:
            self._append(" ", language=NULL_LANGUAGE_ID, alignable=False, kind="separator")
        self._append(value, language=language, alignable=alignable, kind=kind)

    def _append(self, value: str, *, language: int, alignable: bool, kind: str) -> None:
        start = self._length
        self._parts.append(value)
        self._length += len(value)
        self._segments.append(
            _RenderedSegment(
                start=start,
                stop=self._length,
                language_id=language,
                alignable=alignable,
                kind=kind,
            )
        )

    def add_word_boundary(self) -> None:
        """Append one canonical XPhone boundary, collapsing adjacent aliases."""
        if self._segments and self._segments[-1].kind == "word_boundary":
            return
        self.add("▁", kind="word_boundary")

    def finish(self, *, target_language_id: int) -> _RenderedInput:
        if not any(segment.alignable for segment in self._segments):
            raise ValueError("Frozen text provider input contains no alignable speech content.")
        return _RenderedInput(
            text="".join(self._parts),
            segments=tuple(self._segments),
            target_language_id=target_language_id,
        )


def _normalize_input_span(value: TextSpan | Mapping[str, Any]) -> TextSpan:
    if isinstance(value, TextSpan):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("language_spans must contain TextSpan values or mappings.")
    if "language" not in value:
        raise ValueError("Every language span must declare its language.")
    text = value.get("text", "")
    normalized_text = value.get("normalized_text")
    if not isinstance(text, str):
        raise TypeError("Every language span text value must be a string.")
    if normalized_text is not None and not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string or null.")
    return TextSpan(
        text=text,
        language=value["language"],
        normalized_text=normalized_text,
        phonemes=value.get("phonemes"),
    )


def _raw_text_segments(builder: _RenderedInputBuilder, value: str, span_language: int) -> None:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if _CONTROL_TAG.search(normalized):
        raise ValueError(
            "The raw_text frozen frontend does not invent provider control/style tokens; "
            "remove angle-bracket tags or use a provider natively trained for them."
        )
    if not normalized:
        raise ValueError("Every raw_text span must contain text or normalized_text.")
    builder.add(
        normalized,
        language=span_language,
        alignable=True,
        kind="content",
    )


def _phoneme_units(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        units = re.findall(r"<[^>]+>|[^\s]+", unicodedata.normalize("NFD", value))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        units = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("phonemes must contain non-empty strings.")
            normalized = unicodedata.normalize("NFD", item.strip())
            if any(character.isspace() for character in normalized):
                raise ValueError(
                    "Each phoneme sequence item must be one whitespace-free provider unit."
                )
            units.append(normalized)
    else:
        raise TypeError("phonemes must be a string or a sequence of strings.")
    if not units:
        raise ValueError("phonemes must contain at least one supplied phone.")
    return tuple(units)


def _phoneme_unit_alignable(unit: str) -> bool:
    if unit in _PHONEME_NON_ALIGNING_UNITS:
        return False
    categories = {unicodedata.category(character) for character in unit}
    return not categories or not all(
        category.startswith("P") or category in {"Lm", "Mn", "Sk"} for category in categories
    )


def _phoneme_segments(
    builder: _RenderedInputBuilder,
    value: str | Sequence[str],
    span_language: int,
) -> None:
    for unit in _phoneme_units(value):
        if _CONTROL_TAG.fullmatch(unit):
            raise ValueError(
                "The phonemes frozen frontend accepts only provider-native phones, "
                "punctuation, and word boundaries; style/control tags are unsupported."
            )
        if unit in _PHONEME_WORD_BOUNDARIES:
            builder.add_word_boundary()
            continue
        alignable = _phoneme_unit_alignable(unit)
        builder.add(
            unit,
            language=span_language if alignable else NULL_LANGUAGE_ID,
            alignable=alignable,
            kind="content" if alignable else "boundary",
        )


def _render_provider_input(
    spec: FrozenTextProviderSpec,
    text: str | None,
    *,
    normalized_text: str | None,
    phonemes: str | Sequence[str] | None,
    language: str | Language | None,
    language_spans: Sequence[TextSpan | Mapping[str, Any]] | None,
) -> _RenderedInput:
    if text is not None and not isinstance(text, str):
        raise TypeError("text must be a string or null.")
    if normalized_text is not None and not isinstance(normalized_text, str):
        raise TypeError("normalized_text must be a string or null.")
    if language_spans is not None:
        if isinstance(language_spans, (str, bytes)) or not isinstance(language_spans, Sequence):
            raise TypeError("language_spans must be a sequence.")
        spans = tuple(_normalize_input_span(span) for span in language_spans)
        if not spans:
            raise ValueError("language_spans must not be empty.")
        if normalized_text is not None or phonemes is not None:
            raise ValueError(
                "Top-level normalized_text/phonemes cannot be combined with language_spans."
            )
        target_language = normalize_language(
            language if language is not None else spans[0].language
        )
    else:
        target_language = normalize_language(language)
        spans = (
            TextSpan(
                text=text or "",
                language=target_language,
                normalized_text=normalized_text,
                phonemes=phonemes,
            ),
        )

    builder = _RenderedInputBuilder()
    for index, span in enumerate(spans):
        span_code = normalize_language(span.language)
        span_language = language_id(span_code)
        if spec.frozen_text_frontend == "raw_text":
            if span.phonemes is not None:
                raise ValueError("raw_text frozen frontend does not accept phoneme inputs.")
            source = span.normalized_text if span.normalized_text is not None else span.text
            _raw_text_segments(builder, source, span_language)
        else:
            if span.normalized_text is not None:
                raise ValueError("phonemes frozen frontend does not accept normalized_text.")
            if span.phonemes is None:
                raise ValueError(
                    "phonemes frozen frontend requires caller-supplied phonemes for every span."
                )
            if index:
                builder.add_word_boundary()
            _phoneme_segments(builder, span.phonemes, span_language)
    return builder.finish(target_language_id=language_id(target_language))


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_from_hub_cache_path(path: Path) -> str | None:
    for candidate in (path, path.resolve()):
        parts = candidate.parts
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots" and _HUB_COMMIT.fullmatch(parts[index + 1]):
                return parts[index + 1].lower()
    return None


def _resolve_one(
    resolver: Callable[..., str | os.PathLike[str]],
    *,
    repo_id: str,
    filename: str,
    revision: str,
    expected_sha256: str,
) -> Path:
    path = Path(
        resolver(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
        )
    )
    if not path.is_file():
        raise FileNotFoundError(f"Resolved frozen text artifact is missing: {path}.")
    resolved_commit = _commit_from_hub_cache_path(path)
    if resolved_commit is not None and resolved_commit != revision.lower():
        raise RuntimeError(
            "Hugging Face Hub resolved a different frozen text commit than requested: "
            f"{resolved_commit!r} != {revision.lower()!r}."
        )
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen text artifact SHA-256 mismatch for {repo_id}/{filename}: "
            f"{actual_sha256} != {expected_sha256}."
        )
    return path


def _default_artifact_resolver(**kwargs) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _snapshot_root(path: Path, revision: str) -> Path:
    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and parts[index + 1].lower() == revision.lower():
            return Path(*parts[: index + 2])
    return path.parent


def _default_model_loader(
    spec: FrozenTextProviderSpec,
    artifacts: ResolvedFrozenTextArtifacts,
) -> nn.Module:
    from transformers import AutoModel

    config_root = _snapshot_root(
        artifacts.config_path,
        spec.frozen_text_encoder_revision,
    )
    model_root = _snapshot_root(
        artifacts.model_path,
        spec.frozen_text_encoder_revision,
    )
    if config_root != model_root:
        raise RuntimeError("Frozen text config and model did not resolve to one Hub snapshot.")
    suffix = artifacts.model_path.name.lower()
    use_safetensors = (
        True if suffix.endswith(".safetensors") else False if suffix.endswith(".bin") else None
    )
    return AutoModel.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
        output_hidden_states=True,
        add_pooling_layer=False,
        use_safetensors=use_safetensors,
    )


def _default_tokenizer_loader(
    spec: FrozenTextProviderSpec,
    artifacts: ResolvedFrozenTextArtifacts,
) -> Any:
    from transformers import AutoTokenizer

    # The declared tokenizer payload is supplied explicitly. Auxiliary immutable
    # tokenizer metadata may still be resolved from the same full Hub commit.
    return AutoTokenizer.from_pretrained(
        spec.frozen_text_tokenizer_id,
        revision=spec.frozen_text_tokenizer_revision,
        tokenizer_file=str(artifacts.tokenizer_path),
        use_fast=True,
        trust_remote_code=False,
    )


def _extract_hidden_states(output: Any) -> Sequence[torch.Tensor]:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None and isinstance(output, Mapping):
        hidden_states = output.get("hidden_states")
    if (
        hidden_states is None
        or isinstance(hidden_states, torch.Tensor)
        or not isinstance(hidden_states, Sequence)
        or not hidden_states
    ):
        raise RuntimeError("Frozen text AutoModel must return a non-empty hidden_states sequence.")
    if not all(isinstance(state, torch.Tensor) for state in hidden_states):
        raise RuntimeError("Frozen text hidden_states must contain tensors.")
    return hidden_states


def _provider_token_labels(
    rendered: _RenderedInput,
    input_ids: torch.Tensor,
    offsets: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    unk_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_languages = torch.zeros(offsets.shape[0], dtype=torch.long)
    alignment_mask = torch.zeros(offsets.shape[0], dtype=torch.bool)
    for index, ((raw_start, raw_stop), special, attended) in enumerate(
        zip(offsets.tolist(), special_tokens_mask.tolist(), attention_mask.tolist())
    ):
        if not attended or special:
            continue
        if unk_token_id is not None and int(input_ids[index]) == unk_token_id:
            raise ValueError(
                "Frozen text input produced the provider unknown token. Supply only "
                "provider-native text/phones; v1 never silently collapses content or controls."
            )
        start, stop = int(raw_start), int(raw_stop)
        if start < 0 or stop < start or stop > len(rendered.text):
            raise ValueError("Provider tokenizer returned an invalid character offset.")
        if start == stop:
            raise ValueError(
                "A non-special provider token has an empty character offset; "
                "hf_non_special_tokens_v1 requires a fast tokenizer with exact offsets."
            )
        overlaps = tuple(
            segment
            for segment in rendered.segments
            if max(start, segment.start) < min(stop, segment.stop)
        )
        semantic = tuple(segment for segment in overlaps if segment.kind != "separator")
        if not semantic:
            continue
        languages = {segment.language_id for segment in semantic if segment.language_id != 0}
        if len(languages) > 1:
            raise ValueError(
                "One provider token crosses language spans; use a tokenizer/alignment policy "
                "that preserves code-switch boundaries."
            )
        if languages:
            token_languages[index] = next(iter(languages))
        alignment_mask[index] = bool(semantic) and all(segment.alignable for segment in semantic)
    if not bool(alignment_mask.any()):
        raise ValueError("Provider tokenization produced no MAS-eligible speech token.")
    return token_languages, alignment_mask


class FrozenTextProvider:
    """Load one pinned frozen encoder and create token-aligned cache rows.

    Dependency injection keeps tests and offline pipelines independent of network
    access. Production defaults use ``hf_hub_download``, ``AutoTokenizer``, and
    ``AutoModel`` with full commit revisions and remote code disabled.
    """

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
        artifact_resolver: Callable[..., str | os.PathLike[str]] | None = None,
        tokenizer_loader: Callable[[FrozenTextProviderSpec, ResolvedFrozenTextArtifacts], Any]
        | None = None,
        model_loader: Callable[[FrozenTextProviderSpec, ResolvedFrozenTextArtifacts], nn.Module]
        | None = None,
    ) -> "FrozenTextProvider":
        """Construct from training YAML and verify its acoustic vocab/PAD fields."""
        provider = cls(
            FrozenTextProviderSpec.from_config(config),
            device=device,
            artifact_resolver=artifact_resolver,
            tokenizer_loader=tokenizer_loader,
            model_loader=model_loader,
        )
        provider.validate_acoustic_contract(config)
        return provider

    @classmethod
    def from_manifest(
        cls,
        manifest: Any,
        *,
        device: torch.device | str = "cpu",
        artifact_resolver: Callable[..., str | os.PathLike[str]] | None = None,
        tokenizer_loader: Callable[[FrozenTextProviderSpec, ResolvedFrozenTextArtifacts], Any]
        | None = None,
        model_loader: Callable[[FrozenTextProviderSpec, ResolvedFrozenTextArtifacts], nn.Module]
        | None = None,
    ) -> "FrozenTextProvider":
        """Construct from a validated model manifest or text-conditioning section."""
        provider = cls(
            FrozenTextProviderSpec.from_manifest(manifest),
            device=device,
            artifact_resolver=artifact_resolver,
            tokenizer_loader=tokenizer_loader,
            model_loader=model_loader,
        )
        provider.validate_acoustic_contract(manifest)
        return provider

    def __init__(
        self,
        spec: FrozenTextProviderSpec,
        *,
        device: torch.device | str = "cpu",
        artifact_resolver: Callable[..., str | os.PathLike[str]] | None = None,
        tokenizer_loader: Callable[[FrozenTextProviderSpec, ResolvedFrozenTextArtifacts], Any]
        | None = None,
        model_loader: Callable[
            [FrozenTextProviderSpec, ResolvedFrozenTextArtifacts],
            nn.Module,
        ]
        | None = None,
    ) -> None:
        if not isinstance(spec, FrozenTextProviderSpec):
            raise TypeError("spec must be a FrozenTextProviderSpec.")
        self.spec = spec
        self.device = torch.device(device)
        resolver = _default_artifact_resolver if artifact_resolver is None else artifact_resolver
        artifacts = ResolvedFrozenTextArtifacts(
            config_path=_resolve_one(
                resolver,
                repo_id=spec.frozen_text_encoder_id,
                filename=FROZEN_TEXT_CONFIG_FILENAME,
                revision=spec.frozen_text_encoder_revision,
                expected_sha256=spec.frozen_text_config_sha256,
            ),
            model_path=_resolve_one(
                resolver,
                repo_id=spec.frozen_text_encoder_id,
                filename=spec.frozen_text_model_filename,
                revision=spec.frozen_text_encoder_revision,
                expected_sha256=spec.frozen_text_model_sha256,
            ),
            tokenizer_path=_resolve_one(
                resolver,
                repo_id=spec.frozen_text_tokenizer_id,
                filename=spec.frozen_text_tokenizer_filename,
                revision=spec.frozen_text_tokenizer_revision,
                expected_sha256=spec.frozen_text_tokenizer_sha256,
            ),
        )
        try:
            config_payload = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Frozen text config artifact is not valid UTF-8 JSON.") from exc
        if not isinstance(config_payload, dict):
            raise RuntimeError("Frozen text config artifact must contain a JSON object.")

        load_tokenizer = _default_tokenizer_loader if tokenizer_loader is None else tokenizer_loader
        load_model = _default_model_loader if model_loader is None else model_loader
        self.tokenizer = load_tokenizer(spec, artifacts)
        if not bool(getattr(self.tokenizer, "is_fast", False)):
            raise TypeError(
                "hf_non_special_tokens_v1 requires a fast tokenizer with offset mappings."
            )
        model = load_model(spec, artifacts)
        if not isinstance(model, nn.Module):
            raise TypeError("AutoModel loader must return a torch.nn.Module.")
        model.eval()
        model.requires_grad_(False)
        self.model = model.to(self.device)
        self.artifacts = artifacts
        self.provider_config = MappingProxyType(dict(config_payload))

        model_config = getattr(self.model, "config", None)
        declared_width = getattr(model_config, "hidden_size", None)
        if isinstance(declared_width, int) and declared_width != spec.conditioning_feature_size:
            raise ValueError(
                "conditioning_feature_size does not match AutoModel config.hidden_size: "
                f"{spec.conditioning_feature_size} != {declared_width}."
            )
        self._validate_hf_contract()

    def _validate_hf_contract(self) -> None:
        """Bind verified JSON, loaded model, and tokenizer sequence metadata."""

        def required_config_integer(name: str) -> int:
            value = self.provider_config.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Frozen text config.json must declare a non-negative integer {name}."
                )
            return value

        config_values = {
            name: required_config_integer(name)
            for name in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id")
        }
        if config_values["vocab_size"] <= 0:
            raise ValueError("Frozen text config.json vocab_size must be positive.")
        if len(set(config_values.values())) != len(config_values):
            # vocab_size is a size rather than an ID and may numerically collide
            # only by accident; exclude it from the distinct-token check.
            token_ids = {
                config_values["pad_token_id"],
                config_values["bos_token_id"],
                config_values["eos_token_id"],
            }
            if len(token_ids) != 3:
                raise ValueError("Frozen text PAD/BOS/EOS token IDs must be distinct.")

        model_config = getattr(self.model, "config", None)
        for name, expected in config_values.items():
            actual = getattr(model_config, name, None)
            if actual != expected:
                raise ValueError(
                    f"Loaded AutoModel {name} does not match verified config.json: "
                    f"{actual!r} != {expected!r}."
                )

        tokenizer_values = {
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
        }
        for name, actual in tokenizer_values.items():
            expected = config_values[name]
            if actual != expected:
                raise ValueError(
                    f"Loaded AutoTokenizer {name} does not match verified config.json: "
                    f"{actual!r} != {expected!r}."
                )
        try:
            tokenizer_size = len(self.tokenizer)
        except TypeError as exc:
            raise TypeError(
                "Frozen text tokenizer must expose its complete vocabulary size."
            ) from exc
        if tokenizer_size != config_values["vocab_size"]:
            raise ValueError(
                "Loaded AutoTokenizer vocabulary does not match verified config.json: "
                f"{tokenizer_size} != {config_values['vocab_size']}."
            )

        self.vocab_size = config_values["vocab_size"]
        self.pad_token_id = config_values["pad_token_id"]
        self.bos_token_id = config_values["bos_token_id"]
        self.eos_token_id = config_values["eos_token_id"]
        if self.spec.text_vocab_size != self.vocab_size or self.spec.pad_token != self.pad_token_id:
            raise ValueError(
                "Frozen provider spec vocab/PAD identity does not match its verified artifacts."
            )

    def validate_acoustic_contract(self, value: Any) -> None:
        """Compare provider vocab/PAD with a training config or model manifest."""
        text_conditioning: Mapping[str, Any] = {}
        if isinstance(value, Mapping):
            architecture = value.get("architecture")
            source = architecture if isinstance(architecture, Mapping) else value
            candidate = value.get("text_conditioning")
            if isinstance(candidate, Mapping):
                text_conditioning = candidate
        else:
            architecture = getattr(value, "architecture", None)
            source = architecture if isinstance(architecture, Mapping) else {}
            candidate = getattr(value, "text_conditioning", None)
            if isinstance(candidate, Mapping):
                text_conditioning = candidate
        expected_vocab = source.get("text_vocab_size")
        if expected_vocab is not None and expected_vocab != self.vocab_size:
            raise ValueError(
                "Acoustic text_vocab_size does not match the frozen provider: "
                f"{expected_vocab!r} != {self.vocab_size}."
            )
        expected_pad = source.get("pad_token", text_conditioning.get("provider_pad_token"))
        if expected_pad is not None and expected_pad != self.pad_token_id:
            raise ValueError(
                "Acoustic pad_token does not match the frozen provider: "
                f"{expected_pad!r} != {self.pad_token_id}."
            )
        if self.spec.text_vocab_size != self.vocab_size or self.spec.pad_token != self.pad_token_id:
            raise ValueError(
                "Frozen provider spec vocab/PAD identity does not match its verified artifacts."
            )

    def encode(
        self,
        text: str | None,
        *,
        normalized_text: str | None = None,
        phonemes: str | Sequence[str] | None = None,
        language: str | Language | None = None,
        language_spans: Sequence[TextSpan | Mapping[str, Any]] | None = None,
    ) -> FrozenTextConditioning:
        """Return one serialized cache row with no gradients or acoustic modules."""
        rendered = _render_provider_input(
            self.spec,
            text,
            normalized_text=normalized_text,
            phonemes=phonemes,
            language=language,
            language_spans=language_spans,
        )
        try:
            encoded = self.tokenizer(
                rendered.text,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=True,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        except (NotImplementedError, TypeError) as exc:
            raise TypeError(
                "hf_non_special_tokens_v1 requires a fast tokenizer that returns offsets and "
                "special_tokens_mask."
            ) from exc
        if not isinstance(encoded, Mapping):
            raise TypeError("Frozen text tokenizer must return a mapping of tensors.")
        required = {"input_ids", "attention_mask", "special_tokens_mask", "offset_mapping"}
        missing = required - set(encoded)
        if missing:
            raise ValueError(f"Frozen text tokenizer output is missing: {sorted(missing)}.")
        tensors = {name: encoded[name] for name in required}
        if not all(isinstance(value, torch.Tensor) for value in tensors.values()):
            raise TypeError("Frozen text tokenizer outputs must be torch tensors.")
        input_ids = tensors["input_ids"]
        attention_mask = tensors["attention_mask"]
        special_tokens_mask = tensors["special_tokens_mask"]
        offsets = tensors["offset_mapping"]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
            raise ValueError("Frozen text tokenizer input_ids must have shape [1, provider_token].")
        token_shape = tuple(input_ids.shape)
        if (
            tuple(attention_mask.shape) != token_shape
            or tuple(special_tokens_mask.shape) != token_shape
        ):
            raise ValueError("Frozen text tokenizer masks must have the input_ids shape.")
        if tuple(offsets.shape) != (*token_shape, 2):
            raise ValueError(
                "Frozen text tokenizer offsets must have shape [1, provider_token, 2]."
            )
        if bool((input_ids < 0).any()) or bool((input_ids >= self.vocab_size).any()):
            raise ValueError("Frozen text tokenizer emitted an ID outside config.json vocab_size.")
        if not bool(attention_mask.to(dtype=torch.bool).all()):
            raise ValueError("Unbatched frozen text tokenization must not contain padding.")
        if (
            int(input_ids[0, 0]) != self.bos_token_id
            or int(input_ids[0, -1]) != self.eos_token_id
            or not bool(special_tokens_mask[0, 0])
            or not bool(special_tokens_mask[0, -1])
        ):
            raise ValueError(
                "Frozen text tokenizer must add the verified BOS/EOS tokens around every input."
            )
        max_positions = getattr(
            getattr(self.model, "config", None), "max_position_embeddings", None
        )
        if isinstance(max_positions, int) and input_ids.shape[1] > max_positions:
            raise ValueError(
                "Frozen text input exceeds AutoModel max_position_embeddings without truncation: "
                f"{input_ids.shape[1]} > {max_positions}."
            )

        cache_ids = input_ids[0].detach().to(device="cpu", dtype=torch.long).contiguous()
        cache_attention = (
            attention_mask[0]
            .detach()
            .to(
                device="cpu",
                dtype=torch.bool,
            )
            .contiguous()
        )
        token_languages, alignment_mask = _provider_token_labels(
            rendered,
            cache_ids,
            offsets[0].detach().to(device="cpu", dtype=torch.long),
            special_tokens_mask[0].detach().to(device="cpu", dtype=torch.bool),
            cache_attention,
            unk_token_id=getattr(self.tokenizer, "unk_token_id", None),
        )

        model_inputs = {
            name: value.to(self.device)
            for name, value in encoded.items()
            if name not in {"offset_mapping", "special_tokens_mask"}
            and isinstance(value, torch.Tensor)
        }
        with torch.no_grad():
            output = self.model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = _extract_hidden_states(output)
        layer = self.spec.frozen_text_hidden_layer
        if not -len(hidden_states) <= layer < len(hidden_states):
            raise ValueError(
                f"frozen_text_hidden_layer={layer} is invalid for {len(hidden_states)} states."
            )
        features = hidden_states[layer]
        expected_shape = (
            1,
            input_ids.shape[1],
            self.spec.conditioning_feature_size,
        )
        if tuple(features.shape) != expected_shape:
            raise ValueError(
                "Selected frozen text hidden state has shape "
                f"{tuple(features.shape)}, expected {expected_shape}."
            )
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("Frozen text AutoModel returned non-finite hidden states.")
        cache_features = (
            features[0]
            .detach()
            .to(
                device="cpu",
                dtype=self.spec.cache_dtype,
            )
            .contiguous()
        )
        return FrozenTextConditioning(
            conditioning_ids=cache_ids,
            conditioning_features=cache_features,
            conditioning_mask=cache_attention,
            token_language_ids=token_languages,
            alignment_mask=alignment_mask,
            rendered_text=rendered.text,
            target_language_id=rendered.target_language_id,
            cache_version=self.spec.frozen_text_cache_version,
            contract_sha256=self.spec.contract_sha256,
        )


def resolve_frozen_text_provider(
    *,
    provider: FrozenTextProvider | None = None,
    config: Mapping[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> FrozenTextProvider | None:
    """Resolve the canonical preparation provider without hidden configuration parsing.

    Dataset APIs accept either a preconstructed provider (useful for reusing one loaded
    backbone) or the validated training-config mapping.  A frozen-default config therefore
    constructs the pinned provider automatically, while scratch preparation returns ``None``.
    """
    if provider is not None and config is not None:
        raise ValueError("Pass frozen_text_provider or frozen_text_config, not both.")
    if provider is not None:
        if not isinstance(provider, FrozenTextProvider) and not (
            hasattr(provider, "spec") and callable(getattr(provider, "encode", None))
        ):
            raise TypeError(
                "frozen_text_provider must expose the authenticated spec and encode() API."
            )
        provider_device = torch.device(getattr(provider, "device", "cpu"))
        selected_device = torch.device(device)
        if provider_device.type == "cuda" and provider_device != selected_device:
            raise ValueError(
                "A preconstructed CUDA frozen_text_provider must live on this process's "
                f"selected local-rank device: {provider_device} != {selected_device}. Pass "
                "frozen_text_config to construct it after distributed initialization."
            )
        return provider
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError("frozen_text_config must be a training-config mapping.")
    mode = config.get("text_conditioning_mode", "scratch_tokens")
    if mode == "scratch_tokens":
        return None
    if mode != "frozen_features":
        raise ValueError(
            "frozen_text_config text_conditioning_mode must be scratch_tokens or frozen_features."
        )
    return FrozenTextProvider.from_config(config, device=device)


__all__ = [
    "FROZEN_TEXT_ALIGNMENT_POLICY",
    "FROZEN_TEXT_CONFIG_FILENAME",
    "FROZEN_TEXT_REPRESENTATION_NAME",
    "FROZEN_TEXT_REPRESENTATION_VERSION",
    "FrozenTextConditioning",
    "FrozenTextProvider",
    "FrozenTextProviderSpec",
    "ResolvedFrozenTextArtifacts",
    "resolve_frozen_text_provider",
]
