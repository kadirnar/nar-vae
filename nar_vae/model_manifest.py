"""Versioned acoustic-model, frontend, and codec binding manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nar_vae.dacvae.loader import (
    DACVAESourceDescription,
    HubDACVAESource,
    describe_dacvae_source,
    normalize_dacvae_source,
)
from nar_vae.dacvae_encoding import DACVAE_POSTERIOR_SAMPLING_POLICY
from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_VERSION,
    TEXT_FRONTEND_NAME,
    TEXT_FRONTEND_VERSION,
)
from nar_vae.frozen_text_provider import (
    FROZEN_TEXT_REPRESENTATION_NAME,
    FROZEN_TEXT_REPRESENTATION_VERSION,
)
from nar_vae.languages import (
    normalize_language_pairs,
    normalize_languages,
    resolve_language_pair_support,
)
from nar_vae.model_presets import ARCHITECTURE_FIELDS, resolve_model_architecture
from nar_vae.models.duration import ECHODIT_ARCHITECTURE_VERSION
from nar_vae.objectives import (
    DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    RECTIFIED_FLOW_OBJECTIVE,
    normalize_generative_objective,
    validate_diffusion_schedule_shift,
)

MODEL_MANIFEST_FILENAME = "nar_vae_manifest.json"
MODEL_MANIFEST_SCHEMA_VERSION = 5
LEGACY_MODEL_MANIFEST_SCHEMA_VERSION = 3
PREVIOUS_MODEL_MANIFEST_SCHEMA_VERSION = 4
MODEL_MANIFEST_LIBRARY = "nar-vae"
MODEL_MANIFEST_STAGES = ("pretrain", "sft", "grpo")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HUB_COMMIT = re.compile(r"[0-9a-fA-F]{40}")

_ARCHITECTURE_FIELDS = (
    "architecture_version",
    "latent_size",
    *ARCHITECTURE_FIELDS,
    "text_vocab_size",
    "speaker_patch_size",
    "speaker_num_summary_tokens",
    "target_patch_size",
    "use_speaker_conditioning",
    "use_mas_duration",
    "norm_eps",
)
_LEGACY_ARCHITECTURE_FIELDS = tuple(
    name for name in _ARCHITECTURE_FIELDS if name != "speaker_num_summary_tokens"
)
_CAPABILITY_FIELDS = (
    "speaker_conditioning",
    "language_conditioning",
    "supported_languages",
    "supported_reference_languages",
    "supported_language_pairs",
    "duration_predictor",
    "duration_predictor_hidden_size",
    "duration_predictor_num_layers",
    "duration_predictor_use_speaker",
    "monotonic_alignment",
    "duration_alignment_hidden_size",
)
_LEGACY_REPRESENTATION_FIELDS = (
    "contract_version",
    "text_frontend_name",
    "text_frontend_version",
    "codec_source",
    "codec_backend",
    "codec_revision",
    "codec_filename",
    "codec_sha256",
    "sample_rate",
    "hop_length",
    "latent_width",
)
_REPRESENTATION_FIELDS = (
    "contract_version",
    "text_frontend_name",
    "text_frontend_version",
    "codec_source",
    "codec_backend",
    "codec_revision",
    "codec_filename",
    "codec_sha256",
    "codec_encoding_policy",
    "sample_rate",
    "hop_length",
    "latent_width",
)
_GENERATION_FIELDS = ("objective", "diffusion_schedule_shift")
_TEXT_CONDITIONING_FIELDS = (
    "mode",
    "provider_vocab_size",
    "provider_pad_token",
    "feature_size",
    "adapter_version",
    "encoder_id",
    "encoder_revision",
    "config_sha256",
    "model_filename",
    "model_sha256",
    "tokenizer_id",
    "tokenizer_revision",
    "tokenizer_filename",
    "tokenizer_sha256",
    "hidden_layer",
    "feature_dtype",
    "frontend",
    "alignment",
    "cache_version",
)
_PARENT_FIELDS = (
    "manifest_sha256",
    "stage",
    "weights_sha256",
    "representation_sha256",
)
_GRPO_PARENT_FIELDS = (
    *_PARENT_FIELDS,
    "selected_weight_filename",
    "selected_weight_sha256",
    "base_weight_filename",
    "base_weight_sha256",
)


class ModelManifestError(ValueError):
    """Raised when model artifacts do not share one validated contract."""


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelManifestError(f"{name} must be a positive integer.")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelManifestError(f"{name} must be a non-negative integer.")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ModelManifestError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelManifestError(f"{name} must be a finite number.") from exc
    if not (-float("inf") < result < float("inf")):
        raise ModelManifestError(f"{name} must be a finite number.")
    return result


def _strict_boolean(config: Mapping[str, Any], name: str, *, default: bool = False) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ModelManifestError(f"{name} must be a boolean.")
    return value


def _relative_filename(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"{name} must be a non-empty relative filename.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ModelManifestError(f"{name} must be a repository-relative filename.")
    return path.as_posix()


def _mapping_with_exact_fields(
    value: Any,
    fields: Sequence[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelManifestError(f"Model manifest {name} must be an object.")
    result = dict(value)
    expected = set(fields)
    if set(result) != expected:
        missing = sorted(expected - set(result))
        unknown = sorted(set(result) - expected)
        raise ModelManifestError(
            f"Model manifest {name} fields are invalid: missing={missing}, unknown={unknown}."
        )
    return result


def architecture_from_config(config: Mapping[str, Any]) -> dict[str, int | float]:
    """Resolve the exact model-construction shape stored with exported weights."""
    preset = resolve_model_architecture(config)
    architecture: dict[str, int | float] = {
        "architecture_version": ECHODIT_ARCHITECTURE_VERSION,
        "latent_size": _positive_integer(config.get("dacvae_latent_dim"), name="latent_size"),
        **preset.model_kwargs(),
        "text_vocab_size": _positive_integer(config.get("text_vocab_size"), name="text_vocab_size"),
        "speaker_patch_size": _positive_integer(
            config.get("speaker_patch_size"), name="speaker_patch_size"
        ),
        "speaker_num_summary_tokens": _nonnegative_integer(
            config.get("speaker_num_summary_tokens", 0),
            name="speaker_num_summary_tokens",
        ),
        "target_patch_size": _positive_integer(
            config.get("target_patch_size", 1), name="target_patch_size"
        ),
        "use_speaker_conditioning": _strict_boolean(config, "use_speaker_conditioning"),
        "use_mas_duration": _strict_boolean(config, "use_mas_duration"),
        "norm_eps": _finite_number(config.get("norm_eps", 1e-6), name="norm_eps"),
    }
    if architecture["norm_eps"] <= 0:
        raise ModelManifestError("norm_eps must be positive.")
    if (
        architecture["speaker_num_summary_tokens"] > 0
        and not architecture["use_speaker_conditioning"]
    ):
        raise ModelManifestError("speaker_num_summary_tokens requires speaker conditioning.")
    return architecture


def generation_from_config(config: Mapping[str, Any]) -> dict[str, str | float]:
    """Build the exact continuous generative objective/schedule contract."""
    try:
        objective = normalize_generative_objective(
            config.get("generative_objective", RECTIFIED_FLOW_OBJECTIVE)
        )
        shift = validate_diffusion_schedule_shift(
            config.get("diffusion_schedule_shift", DEFAULT_DIFFUSION_SCHEDULE_SHIFT)
        )
    except ValueError as exc:
        raise ModelManifestError(str(exc)) from exc
    if objective == RECTIFIED_FLOW_OBJECTIVE and shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT:
        raise ModelManifestError(
            "Rectified-flow manifests cannot declare a shifted diffusion schedule."
        )
    return {"objective": objective, "diffusion_schedule_shift": shift}


def text_conditioning_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the frozen provider/cache identity without embedding it in acoustic weights."""
    mode = config.get("text_conditioning_mode", "scratch_tokens")
    if mode == "scratch_tokens":
        return {
            "mode": mode,
            "provider_vocab_size": 0,
            "provider_pad_token": 0,
            "feature_size": 0,
            "adapter_version": 0,
            "encoder_id": None,
            "encoder_revision": None,
            "config_sha256": None,
            "model_filename": None,
            "model_sha256": None,
            "tokenizer_id": None,
            "tokenizer_revision": None,
            "tokenizer_filename": None,
            "tokenizer_sha256": None,
            "hidden_layer": None,
            "feature_dtype": None,
            "frontend": None,
            "alignment": None,
            "cache_version": 0,
        }
    if mode != "frozen_features":
        raise ModelManifestError(f"Unsupported text_conditioning_mode: {mode!r}.")
    provider_vocab_size = _positive_integer(
        config.get("text_vocab_size"),
        name="text_vocab_size",
    )
    provider_pad_token = config.get("pad_token")
    if (
        isinstance(provider_pad_token, bool)
        or not isinstance(provider_pad_token, int)
        or not 0 <= provider_pad_token < provider_vocab_size
    ):
        raise ModelManifestError("pad_token must be within text_vocab_size.")
    return {
        "mode": mode,
        "provider_vocab_size": provider_vocab_size,
        "provider_pad_token": provider_pad_token,
        "feature_size": _positive_integer(
            config.get("conditioning_feature_size"),
            name="conditioning_feature_size",
        ),
        "adapter_version": 1,
        "encoder_id": config.get("frozen_text_encoder_id"),
        "encoder_revision": config.get("frozen_text_encoder_revision"),
        "config_sha256": config.get("frozen_text_config_sha256"),
        "model_filename": config.get("frozen_text_model_filename"),
        "model_sha256": config.get("frozen_text_model_sha256"),
        "tokenizer_id": config.get("frozen_text_tokenizer_id"),
        "tokenizer_revision": config.get("frozen_text_tokenizer_revision"),
        "tokenizer_filename": config.get("frozen_text_tokenizer_filename"),
        "tokenizer_sha256": config.get("frozen_text_tokenizer_sha256"),
        "hidden_layer": config.get("frozen_text_hidden_layer"),
        "feature_dtype": config.get("conditioning_feature_dtype"),
        "frontend": config.get("frozen_text_frontend"),
        "alignment": config.get("frozen_text_alignment"),
        "cache_version": _positive_integer(
            config.get("frozen_text_cache_version"),
            name="frozen_text_cache_version",
        ),
    }


def capabilities_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the explicit training capability declaration for an export."""
    speaker_conditioning = _strict_boolean(config, "use_speaker_conditioning")
    language_conditioning = _strict_boolean(config, "use_language_conditioning")
    supported_languages = normalize_languages(config.get("supported_languages", ("en",)))
    normalized_reference_languages, supported_language_pairs = resolve_language_pair_support(
        supported_languages,
        supported_reference_languages=config.get("supported_reference_languages"),
        supported_language_pairs=config.get("supported_language_pairs"),
    )
    if supported_language_pairs and not (speaker_conditioning and language_conditioning):
        raise ModelManifestError(
            "Reference-language pair support requires speaker and language conditioning."
        )
    if speaker_conditioning and language_conditioning and not supported_language_pairs:
        raise ModelManifestError(
            "Speaker-conditioned multilingual manifests require exact supported_language_pairs."
        )
    duration_predictor = _strict_boolean(config, "use_duration_predictor")
    monotonic_alignment = _strict_boolean(config, "use_mas_duration")
    duration_uses_speaker = _strict_boolean(config, "duration_predictor_use_speaker")
    return {
        "speaker_conditioning": speaker_conditioning,
        "language_conditioning": language_conditioning,
        "supported_languages": list(supported_languages),
        "supported_reference_languages": list(normalized_reference_languages),
        "supported_language_pairs": [list(pair.as_tuple()) for pair in supported_language_pairs],
        "duration_predictor": duration_predictor,
        "duration_predictor_hidden_size": (
            _positive_integer(
                config.get("duration_predictor_hidden_size", 256),
                name="duration_predictor_hidden_size",
            )
            if duration_predictor
            else 0
        ),
        "duration_predictor_num_layers": (
            _positive_integer(
                config.get("duration_predictor_num_layers", 2),
                name="duration_predictor_num_layers",
            )
            if duration_predictor
            else 0
        ),
        "duration_predictor_use_speaker": duration_predictor and duration_uses_speaker,
        "monotonic_alignment": monotonic_alignment,
        "duration_alignment_hidden_size": (
            _positive_integer(
                config.get("duration_alignment_hidden_size", 64),
                name="duration_alignment_hidden_size",
            )
            if monotonic_alignment
            else 0
        ),
    }


def _codec_description_from_config(config: Mapping[str, Any]) -> DACVAESourceDescription:
    source = config.get("dacvae_model")
    if not isinstance(source, (str, os.PathLike)) or not os.fspath(source).strip():
        raise ModelManifestError("dacvae_model must identify the codec used to prepare the data.")
    try:
        normalized = normalize_dacvae_source(
            source,
            dacvae_revision=config.get("dacvae_revision"),
            dacvae_filename=config.get("dacvae_filename"),
        )
    except ValueError as exc:
        raise ModelManifestError(str(exc)) from exc
    return describe_dacvae_source(normalized)


def representation_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the prepared-data representation contract persisted with a model."""
    source = _codec_description_from_config(config)
    backend = config.get("dacvae_backend")
    if backend not in {"bundled", "fast"}:
        raise ModelManifestError(
            "dacvae_backend must be the explicit resolved backend 'bundled' or 'fast'."
        )
    codec_sha256 = config.get("dacvae_sha256")
    if not isinstance(codec_sha256, str) or not _SHA256.fullmatch(codec_sha256):
        raise ModelManifestError(
            "dacvae_sha256 must bind the exact local or Hub codec artifact with a lowercase "
            "64-character SHA-256."
        )
    frozen_features = config.get("text_conditioning_mode", "scratch_tokens") == "frozen_features"
    return {
        "contract_version": REPRESENTATION_CONTRACT_VERSION,
        "text_frontend_name": (
            FROZEN_TEXT_REPRESENTATION_NAME if frozen_features else TEXT_FRONTEND_NAME
        ),
        "text_frontend_version": (
            FROZEN_TEXT_REPRESENTATION_VERSION if frozen_features else TEXT_FRONTEND_VERSION
        ),
        "codec_source": source.identifier,
        "codec_backend": backend,
        "codec_revision": source.revision,
        "codec_filename": source.filename,
        "codec_sha256": codec_sha256,
        "codec_encoding_policy": DACVAE_POSTERIOR_SAMPLING_POLICY,
        "sample_rate": _positive_integer(
            config.get("dacvae_sample_rate"), name="dacvae_sample_rate"
        ),
        "hop_length": _positive_integer(config.get("dacvae_hop_length"), name="dacvae_hop_length"),
        "latent_width": _positive_integer(
            config.get("dacvae_latent_dim"), name="dacvae_latent_dim"
        ),
    }


@dataclass(frozen=True)
class ModelManifest:
    """Validated immutable manifest loaded beside an acoustic checkpoint."""

    path: Path
    stage: str
    weights: Mapping[str, str]
    architecture: Mapping[str, int | float]
    capabilities: Mapping[str, Any]
    representation: Mapping[str, Any]
    generation: Mapping[str, Any]
    text_conditioning: Mapping[str, Any]
    parent: Mapping[str, Any] | None
    raw: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.raw)


def _validate_manifest(value: Any, *, path: Path) -> ModelManifest:
    if not isinstance(value, Mapping):
        raise ModelManifestError("NAR-VAE model manifest must be a JSON object.")
    raw = dict(value)
    schema_version = raw.get("schema_version")
    if schema_version not in {
        LEGACY_MODEL_MANIFEST_SCHEMA_VERSION,
        PREVIOUS_MODEL_MANIFEST_SCHEMA_VERSION,
        MODEL_MANIFEST_SCHEMA_VERSION,
    }:
        raise ModelManifestError("Unsupported NAR-VAE model-manifest schema.")
    expected_root = {
        "schema_version",
        "library",
        "stage",
        "weights",
        "architecture",
        "capabilities",
        "representation",
        "parent",
    }
    if schema_version != LEGACY_MODEL_MANIFEST_SCHEMA_VERSION:
        expected_root.update({"generation", "text_conditioning"})
    if set(raw) != expected_root:
        raise ModelManifestError("NAR-VAE model manifest has incomplete or unknown root fields.")
    if raw["library"] != MODEL_MANIFEST_LIBRARY:
        raise ModelManifestError("The acoustic checkpoint was not exported by NAR-VAE.")
    stage = raw["stage"]
    if stage not in MODEL_MANIFEST_STAGES:
        raise ModelManifestError(f"Unsupported NAR-VAE training stage {stage!r}.")

    weights_value = raw["weights"]
    if not isinstance(weights_value, Mapping) or not weights_value:
        raise ModelManifestError("Model manifest weights must be a non-empty object.")
    weights: dict[str, str] = {}
    for filename, checksum in weights_value.items():
        resolved_name = _relative_filename(filename, name="weight filename")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise ModelManifestError(f"Weight {resolved_name!r} has an invalid SHA-256.")
        weights[resolved_name] = checksum

    architecture_fields = (
        _LEGACY_ARCHITECTURE_FIELDS
        if schema_version == LEGACY_MODEL_MANIFEST_SCHEMA_VERSION
        else _ARCHITECTURE_FIELDS
    )
    architecture = _mapping_with_exact_fields(
        raw["architecture"], architecture_fields, name="architecture"
    )
    for name in architecture_fields:
        if name in {"use_speaker_conditioning", "use_mas_duration"}:
            if not isinstance(architecture[name], bool):
                raise ModelManifestError(f"Manifest architecture {name} must be a boolean.")
        elif name == "norm_eps":
            architecture[name] = _finite_number(architecture[name], name=name)
            if architecture[name] <= 0:
                raise ModelManifestError("Manifest architecture norm_eps must be positive.")
        elif name in {"text_num_layers", "speaker_num_summary_tokens"}:
            value = architecture[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ModelManifestError(f"Manifest architecture {name} must be non-negative.")
        else:
            architecture[name] = _positive_integer(architecture[name], name=name)
    if schema_version == LEGACY_MODEL_MANIFEST_SCHEMA_VERSION:
        architecture["speaker_num_summary_tokens"] = 0
    if architecture["architecture_version"] != ECHODIT_ARCHITECTURE_VERSION:
        raise ModelManifestError(
            "The model manifest uses an unsupported NAR-VAE architecture version."
        )

    capabilities = _mapping_with_exact_fields(
        raw["capabilities"], _CAPABILITY_FIELDS, name="capabilities"
    )
    for name in (
        "speaker_conditioning",
        "language_conditioning",
        "duration_predictor",
        "duration_predictor_use_speaker",
        "monotonic_alignment",
    ):
        if not isinstance(capabilities[name], bool):
            raise ModelManifestError(f"Manifest capability {name} must be a boolean.")
    for name in ("supported_languages", "supported_reference_languages"):
        value = capabilities[name]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ModelManifestError(f"Manifest capability {name} must be a language list.")
        normalized = list(normalize_languages(value)) if value else []
        if value != normalized:
            raise ModelManifestError(f"Manifest capability {name} is not normalized.")
    pair_values = capabilities["supported_language_pairs"]
    if not isinstance(pair_values, list) or not all(
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(language, str) for language in pair)
        for pair in pair_values
    ):
        raise ModelManifestError(
            "Manifest capability supported_language_pairs must be a list of "
            "[target, reference] language lists."
        )
    normalized_pairs = (
        [list(pair.as_tuple()) for pair in normalize_language_pairs(pair_values)]
        if pair_values
        else []
    )
    if pair_values != normalized_pairs:
        raise ModelManifestError("Manifest capability supported_language_pairs is not normalized.")
    pair_targets = {pair[0] for pair in pair_values}
    if not pair_targets.issubset(capabilities["supported_languages"]):
        raise ModelManifestError(
            "Manifest language-pair targets must be declared in supported_languages."
        )
    pair_references = list(dict.fromkeys(pair[1] for pair in pair_values))
    if pair_references != capabilities["supported_reference_languages"]:
        raise ModelManifestError(
            "Manifest supported_reference_languages must equal the ordered reference "
            "projection of supported_language_pairs."
        )
    for name in (
        "duration_predictor_hidden_size",
        "duration_predictor_num_layers",
        "duration_alignment_hidden_size",
    ):
        value = capabilities[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelManifestError(f"Manifest capability {name} must be non-negative.")
    duration_enabled = capabilities["duration_predictor"]
    duration_values = (
        capabilities["duration_predictor_hidden_size"],
        capabilities["duration_predictor_num_layers"],
    )
    if duration_enabled != all(value > 0 for value in duration_values):
        raise ModelManifestError(
            "Duration capability and duration-predictor architecture metadata disagree."
        )
    if capabilities["duration_predictor_use_speaker"] and not (
        duration_enabled and capabilities["speaker_conditioning"]
    ):
        raise ModelManifestError(
            "Speaker-conditioned duration requires both duration and speaker capabilities."
        )
    alignment_hidden_size = capabilities["duration_alignment_hidden_size"]
    if capabilities["monotonic_alignment"] != (alignment_hidden_size > 0):
        raise ModelManifestError(
            "Monotonic-alignment capability and hidden-size metadata disagree."
        )
    if capabilities["supported_language_pairs"] and not (
        capabilities["speaker_conditioning"] and capabilities["language_conditioning"]
    ):
        raise ModelManifestError(
            "Reference-language pair coverage requires speaker and language conditioning."
        )
    if (
        capabilities["speaker_conditioning"]
        and capabilities["language_conditioning"]
        and not capabilities["supported_language_pairs"]
    ):
        raise ModelManifestError(
            "Speaker-conditioned multilingual manifests require exact supported_language_pairs."
        )
    if architecture["use_speaker_conditioning"] != capabilities["speaker_conditioning"]:
        raise ModelManifestError("Speaker topology and speaker capability metadata disagree.")
    if architecture["speaker_num_summary_tokens"] > 0 and not capabilities["speaker_conditioning"]:
        raise ModelManifestError("Speaker summary tokens require speaker-conditioning capability.")
    if architecture["use_mas_duration"] != capabilities["monotonic_alignment"]:
        raise ModelManifestError(
            "MAS topology and monotonic-alignment capability metadata disagree."
        )

    if schema_version == LEGACY_MODEL_MANIFEST_SCHEMA_VERSION:
        generation = generation_from_config({})
        text_conditioning = text_conditioning_from_config({})
    else:
        generation = _mapping_with_exact_fields(
            raw["generation"], _GENERATION_FIELDS, name="generation"
        )
        try:
            objective = normalize_generative_objective(generation["objective"])
            shift = validate_diffusion_schedule_shift(generation["diffusion_schedule_shift"])
        except ValueError as exc:
            raise ModelManifestError(str(exc)) from exc
        if objective == RECTIFIED_FLOW_OBJECTIVE and shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT:
            raise ModelManifestError(
                "Rectified-flow manifests cannot declare a shifted diffusion schedule."
            )
        generation = {"objective": objective, "diffusion_schedule_shift": shift}

        text_conditioning = _mapping_with_exact_fields(
            raw["text_conditioning"],
            _TEXT_CONDITIONING_FIELDS,
            name="text_conditioning",
        )
        mode = text_conditioning["mode"]
        if mode == "scratch_tokens":
            expected_scratch = text_conditioning_from_config({})
            if text_conditioning != expected_scratch or architecture["text_num_layers"] <= 0:
                raise ModelManifestError(
                    "Scratch-token text conditioning metadata/topology is inconsistent."
                )
        elif mode == "frozen_features":
            if architecture["text_num_layers"] != 0:
                raise ModelManifestError("Frozen-feature manifests require text_num_layers=0.")
            for name in (
                "provider_vocab_size",
                "feature_size",
                "adapter_version",
                "cache_version",
            ):
                text_conditioning[name] = _positive_integer(
                    text_conditioning[name], name=f"text_conditioning.{name}"
                )
            provider_pad_token = text_conditioning["provider_pad_token"]
            if (
                isinstance(provider_pad_token, bool)
                or not isinstance(provider_pad_token, int)
                or not 0 <= provider_pad_token < text_conditioning["provider_vocab_size"]
            ):
                raise ModelManifestError(
                    "text_conditioning.provider_pad_token must be within provider_vocab_size."
                )
            if text_conditioning["provider_vocab_size"] != architecture["text_vocab_size"]:
                raise ModelManifestError(
                    "Frozen provider vocabulary must match architecture.text_vocab_size."
                )
            if text_conditioning["adapter_version"] != 1:
                raise ModelManifestError("Unsupported frozen text adapter version.")
            for name in ("encoder_id", "tokenizer_id"):
                value = text_conditioning[name]
                if not isinstance(value, str) or value.count("/") != 1 or not all(value.split("/")):
                    raise ModelManifestError(f"text_conditioning.{name} must use owner/name.")
            for name in ("encoder_revision", "tokenizer_revision"):
                value = text_conditioning[name]
                if not isinstance(value, str) or not _HUB_COMMIT.fullmatch(value):
                    raise ModelManifestError(f"text_conditioning.{name} must be a full Hub commit.")
            for name in ("config_sha256", "model_sha256", "tokenizer_sha256"):
                value = text_conditioning[name]
                if not isinstance(value, str) or not _SHA256.fullmatch(value):
                    raise ModelManifestError(
                        f"text_conditioning.{name} must be a lowercase SHA-256."
                    )
            for name in ("model_filename", "tokenizer_filename"):
                text_conditioning[name] = _relative_filename(
                    text_conditioning[name], name=f"text_conditioning.{name}"
                )
            hidden_layer = text_conditioning["hidden_layer"]
            if isinstance(hidden_layer, bool) or not isinstance(hidden_layer, int):
                raise ModelManifestError("text_conditioning.hidden_layer must be an integer.")
            if text_conditioning["feature_dtype"] not in {
                "float16",
                "float32",
            }:
                raise ModelManifestError("Unsupported frozen text feature dtype.")
            if text_conditioning["frontend"] not in {"phonemes", "raw_text"}:
                raise ModelManifestError("Unsupported frozen text frontend.")
            if text_conditioning["alignment"] != "hf_non_special_tokens_v1":
                raise ModelManifestError("Unsupported frozen text alignment policy.")
        else:
            raise ModelManifestError(f"Unsupported text-conditioning mode: {mode!r}.")

    representation_fields = (
        _REPRESENTATION_FIELDS
        if schema_version == MODEL_MANIFEST_SCHEMA_VERSION
        else _LEGACY_REPRESENTATION_FIELDS
    )
    representation = _mapping_with_exact_fields(
        raw["representation"], representation_fields, name="representation"
    )
    expected_representation_version = (
        REPRESENTATION_CONTRACT_VERSION if schema_version == MODEL_MANIFEST_SCHEMA_VERSION else 2
    )
    if representation["contract_version"] != expected_representation_version:
        raise ModelManifestError("Unsupported prepared-data representation contract version.")
    for name in ("text_frontend_name", "codec_source", "codec_backend"):
        if not isinstance(representation[name], str) or not representation[name].strip():
            raise ModelManifestError(f"Manifest representation {name} must be non-empty.")
    if representation["text_frontend_version"] < 1:
        raise ModelManifestError("Manifest text_frontend_version must be positive.")
    for name in ("sample_rate", "hop_length", "latent_width"):
        representation[name] = _positive_integer(representation[name], name=name)
    revision = representation["codec_revision"]
    filename = representation["codec_filename"]
    if (revision is None) != (filename is None):
        raise ModelManifestError("codec_revision and codec_filename must be set together.")
    if revision is not None:
        if not isinstance(revision, str) or not _HUB_COMMIT.fullmatch(revision):
            raise ModelManifestError("Manifest codec_revision must be a full Hub commit.")
        representation["codec_filename"] = _relative_filename(filename, name="codec_filename")
    artifact_sha256 = representation["codec_sha256"]
    if not isinstance(artifact_sha256, str) or not _SHA256.fullmatch(artifact_sha256):
        raise ModelManifestError("Manifest codec_sha256 must be a lowercase SHA-256.")
    if (
        schema_version == MODEL_MANIFEST_SCHEMA_VERSION
        and representation["codec_encoding_policy"] != DACVAE_POSTERIOR_SAMPLING_POLICY
    ):
        raise ModelManifestError("Manifest codec_encoding_policy is unsupported.")
    expected_frontend = (
        (FROZEN_TEXT_REPRESENTATION_NAME, FROZEN_TEXT_REPRESENTATION_VERSION)
        if text_conditioning["mode"] == "frozen_features"
        else (TEXT_FRONTEND_NAME, TEXT_FRONTEND_VERSION)
    )
    observed_frontend = (
        representation["text_frontend_name"],
        representation["text_frontend_version"],
    )
    if observed_frontend != expected_frontend:
        raise ModelManifestError(
            "Manifest representation frontend contradicts its text-conditioning mode."
        )

    parent = raw["parent"]
    if stage == "pretrain" and parent is not None:
        raise ModelManifestError("A pretraining model manifest cannot declare a parent.")
    if stage in {"sft", "grpo"} and parent is None:
        parent_name = "pretraining" if stage == "sft" else "SFT reference"
        raise ModelManifestError(
            f"A {stage.upper()} model manifest must declare its {parent_name} parent."
        )
    if parent is not None:
        parent_fields = _GRPO_PARENT_FIELDS if stage == "grpo" else _PARENT_FIELDS
        parent = _mapping_with_exact_fields(
            parent,
            parent_fields,
            name="parent",
        )
        if parent["stage"] not in MODEL_MANIFEST_STAGES:
            raise ModelManifestError("Model manifest parent has an unsupported stage.")
        if stage == "sft" and parent["stage"] != "pretrain":
            raise ModelManifestError("An SFT model manifest must retain its pretraining parent.")
        if stage == "grpo" and parent["stage"] != "sft":
            raise ModelManifestError(
                "A GRPO model manifest must bind the exact SFT checkpoint used as its "
                "frozen reference policy."
            )
        for name in ("manifest_sha256", "weights_sha256", "representation_sha256"):
            if not isinstance(parent[name], str) or not _SHA256.fullmatch(parent[name]):
                raise ModelManifestError(f"Model manifest parent {name} is invalid.")
        if stage == "grpo":
            selected_name = _relative_filename(
                parent["selected_weight_filename"],
                name="selected_weight_filename",
            )
            selected_sha256 = parent["selected_weight_sha256"]
            if not isinstance(selected_sha256, str) or not _SHA256.fullmatch(selected_sha256):
                raise ModelManifestError("Model manifest selected SFT weight hash is invalid.")
            base_name = parent["base_weight_filename"]
            base_sha256 = parent["base_weight_sha256"]
            if (base_name is None) != (base_sha256 is None):
                raise ModelManifestError(
                    "GRPO parent base weight filename and hash must be set together."
                )
            selected_is_ema = selected_name in {"ema_model.bin", "pytorch_model_ema.bin"} or (
                "_ema" in Path(selected_name).stem
            )
            if selected_is_ema and base_name is None:
                raise ModelManifestError(
                    "A sparse EMA GRPO parent must bind its full SFT base weight."
                )
            if not selected_is_ema and base_name is not None:
                raise ModelManifestError(
                    "A non-EMA GRPO parent cannot declare a separate base weight."
                )
            if base_name is not None:
                base_name = _relative_filename(base_name, name="base_weight_filename")
                if not isinstance(base_sha256, str) or not _SHA256.fullmatch(base_sha256):
                    raise ModelManifestError("Model manifest SFT base weight hash is invalid.")
                if base_name == selected_name:
                    raise ModelManifestError("GRPO selected and base weight filenames must differ.")
            parent["selected_weight_filename"] = selected_name
            parent["base_weight_filename"] = base_name

    raw_architecture = dict(architecture)
    if schema_version == LEGACY_MODEL_MANIFEST_SCHEMA_VERSION:
        raw_architecture.pop("speaker_num_summary_tokens")
    normalized_raw = {
        **raw,
        "weights": weights,
        "architecture": raw_architecture,
        "capabilities": capabilities,
        "representation": representation,
        "parent": parent,
    }
    if schema_version != LEGACY_MODEL_MANIFEST_SCHEMA_VERSION:
        normalized_raw["generation"] = generation
        normalized_raw["text_conditioning"] = text_conditioning
    return ModelManifest(
        path=path,
        stage=stage,
        weights=weights,
        architecture=architecture,
        capabilities=capabilities,
        representation=representation,
        generation=generation,
        text_conditioning=text_conditioning,
        parent=parent,
        raw=normalized_raw,
    )


def load_model_manifest(path: str | os.PathLike[str]) -> ModelManifest:
    """Load and schema-validate one model manifest without loading model tensors."""
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise ModelManifestError(
            "Inference and fresh SFT require a NAR-VAE model manifest beside the weights; "
            f"missing: {manifest_path}."
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelManifestError(
            f"Could not read NAR-VAE model manifest: {manifest_path}."
        ) from exc
    return _validate_manifest(value, path=manifest_path.resolve())


def validate_manifest_weight(
    manifest: ModelManifest,
    checkpoint_path: str | os.PathLike[str],
    *,
    selected_filename: str | None = None,
) -> None:
    """Require that the selected local artifact is exactly hash-bound by a manifest."""
    path = Path(checkpoint_path).expanduser()
    filename = path.name if selected_filename is None else selected_filename
    normalized_name = _relative_filename(filename, name="selected checkpoint filename")
    expected = manifest.weights.get(normalized_name)
    if expected is None:
        raise ModelManifestError(
            f"Selected checkpoint {normalized_name!r} is not bound by {manifest.path.name}."
        )
    if not path.is_file():
        raise ModelManifestError(f"Selected checkpoint is missing: {path}.")
    actual = _file_sha256(path)
    if actual != expected:
        raise ModelManifestError(
            f"Selected checkpoint {normalized_name!r} does not match its manifest SHA-256."
        )


def _parent_reference(parent: ModelManifest | Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(parent, ModelManifest):
        reference = _mapping_with_exact_fields(
            parent,
            _PARENT_FIELDS,
            name="parent",
        )
        if reference["stage"] not in MODEL_MANIFEST_STAGES:
            raise ModelManifestError("Model manifest parent has an unsupported stage.")
        for name in ("manifest_sha256", "weights_sha256", "representation_sha256"):
            if not isinstance(reference[name], str) or not _SHA256.fullmatch(reference[name]):
                raise ModelManifestError(f"Model manifest parent {name} is invalid.")
        return reference
    return {
        "manifest_sha256": parent.sha256,
        "stage": parent.stage,
        "weights_sha256": _canonical_sha256(dict(parent.weights)),
        "representation_sha256": _canonical_sha256(dict(parent.representation)),
    }


def _grpo_parent_reference(
    parent: ModelManifest,
    checkpoint_path: str | os.PathLike[str],
    base_checkpoint_path: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Bind the exact SFT policy artifact selected for a GRPO run."""
    checkpoint = Path(checkpoint_path).expanduser()
    validate_manifest_weight(parent, checkpoint)
    selected_name = checkpoint.name
    selected_is_ema = selected_name in {"ema_model.bin", "pytorch_model_ema.bin"} or (
        "_ema" in checkpoint.stem
    )
    if selected_is_ema and base_checkpoint_path is None:
        raise ModelManifestError("A sparse EMA GRPO parent requires its full SFT base checkpoint.")
    if not selected_is_ema and base_checkpoint_path is not None:
        raise ModelManifestError("A non-EMA GRPO parent cannot declare a separate base checkpoint.")
    base_name = None
    base_sha256 = None
    if base_checkpoint_path is not None:
        base_checkpoint = Path(base_checkpoint_path).expanduser()
        validate_manifest_weight(parent, base_checkpoint)
        if base_checkpoint.name == selected_name:
            raise ModelManifestError("GRPO selected and base checkpoint filenames must differ.")
        base_name = base_checkpoint.name
        base_sha256 = _file_sha256(base_checkpoint)
    return {
        **_parent_reference(parent),
        "selected_weight_filename": selected_name,
        "selected_weight_sha256": _file_sha256(checkpoint),
        "base_weight_filename": base_name,
        "base_weight_sha256": base_sha256,
    }


def write_model_manifest(
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    stage: str,
    checkpoint_files: Sequence[str | os.PathLike[str]],
    parent_manifest: ModelManifest | Mapping[str, Any] | None = None,
    parent_checkpoint_path: str | os.PathLike[str] | None = None,
    parent_base_checkpoint_path: str | os.PathLike[str] | None = None,
) -> ModelManifest:
    """Atomically write a hash-bound manifest beside a training export."""
    if stage not in MODEL_MANIFEST_STAGES:
        raise ModelManifestError(f"Unknown training stage {stage!r}.")
    if stage == "pretrain" and parent_manifest is not None:
        raise ModelManifestError("Pretraining exports cannot declare a parent manifest.")
    if stage == "sft":
        if not isinstance(parent_manifest, ModelManifest):
            raise ModelManifestError(
                "SFT exports require the fully validated pretraining parent model manifest."
            )
        _validate_sft_parent_contract(parent_manifest, config)
    if stage != "grpo" and (
        parent_checkpoint_path is not None or parent_base_checkpoint_path is not None
    ):
        raise ModelManifestError("Exact parent checkpoint selection is only valid for GRPO.")
    if stage == "grpo":
        if not isinstance(parent_manifest, ModelManifest):
            raise ModelManifestError(
                "GRPO exports require the fully validated SFT reference model manifest."
            )
        if (
            parent_manifest.stage != "sft"
            or parent_manifest.parent is None
            or parent_manifest.parent.get("stage") != "pretrain"
        ):
            raise ModelManifestError(
                "GRPO exports require an SFT reference with an explicit NAR-VAE pretraining parent."
            )
        if (
            parent_manifest.representation.get("codec_encoding_policy")
            != DACVAE_POSTERIOR_SAMPLING_POLICY
        ):
            raise ModelManifestError(
                "GRPO exports require an SFT reference trained with the current seeded DACVAE "
                "posterior policy; legacy representations cannot be relabeled during export."
            )
        if dict(parent_manifest.architecture) != architecture_from_config(config):
            raise ModelManifestError("GRPO exports must preserve the SFT reference architecture.")
        if dict(parent_manifest.capabilities) != capabilities_from_config(config):
            raise ModelManifestError("GRPO exports must preserve the SFT reference capabilities.")
        if dict(parent_manifest.representation) != representation_from_config(config):
            raise ModelManifestError(
                "GRPO exports must preserve the SFT reference codec/frontend representation."
            )
        if dict(parent_manifest.generation) != generation_from_config(config):
            raise ModelManifestError(
                "GRPO exports must preserve the SFT reference generative objective and schedule."
            )
        if dict(parent_manifest.text_conditioning) != text_conditioning_from_config(config):
            raise ModelManifestError(
                "GRPO exports must preserve the SFT reference frozen text-conditioning contract."
            )
        if parent_checkpoint_path is None:
            raise ModelManifestError("GRPO exports must bind the exact selected SFT checkpoint.")
    directory = Path(output_dir)
    weights: dict[str, str] = {}
    for checkpoint_file in checkpoint_files:
        filename = _relative_filename(os.fspath(checkpoint_file), name="checkpoint_file")
        path = directory / filename
        if not path.is_file():
            raise ModelManifestError(f"Cannot bind missing checkpoint artifact: {path}.")
        weights[filename] = _file_sha256(path)
    if not weights:
        raise ModelManifestError("At least one checkpoint artifact must be bound.")

    raw = {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "library": MODEL_MANIFEST_LIBRARY,
        "stage": stage,
        "weights": weights,
        "architecture": architecture_from_config(config),
        "capabilities": capabilities_from_config(config),
        "representation": representation_from_config(config),
        "generation": generation_from_config(config),
        "text_conditioning": text_conditioning_from_config(config),
        "parent": (
            _grpo_parent_reference(
                parent_manifest,
                parent_checkpoint_path,
                parent_base_checkpoint_path,
            )
            if stage == "grpo"
            else _parent_reference(parent_manifest)
            if parent_manifest is not None
            else None
        ),
    }
    manifest = _validate_manifest(raw, path=(directory / MODEL_MANIFEST_FILENAME).resolve())
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / MODEL_MANIFEST_FILENAME
    temporary = directory / f".{MODEL_MANIFEST_FILENAME}.tmp"
    temporary.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return ModelManifest(**{**manifest.__dict__, "path": destination.resolve()})


def _validate_sft_parent_contract(
    manifest: ModelManifest,
    config: Mapping[str, Any],
) -> ModelManifest:
    """Validate every immutable or explicitly migratable pretraining-to-SFT contract."""
    if manifest.stage != "pretrain":
        raise ModelManifestError("Fresh SFT requires a NAR-VAE pretraining model manifest.")
    requested_architecture = architecture_from_config(config)
    parent_architecture = dict(manifest.architecture)
    speaker_migration = _strict_boolean(config, "initialize_speaker_conditioning")
    architecture_matches = parent_architecture == requested_architecture
    if not architecture_matches and speaker_migration:
        parent_without_speaker = dict(parent_architecture)
        requested_without_speaker = dict(requested_architecture)
        parent_speaker = parent_without_speaker.pop("use_speaker_conditioning")
        requested_speaker = requested_without_speaker.pop("use_speaker_conditioning")
        parent_summary_tokens = parent_without_speaker.pop("speaker_num_summary_tokens")
        requested_without_speaker.pop("speaker_num_summary_tokens")
        architecture_matches = (
            parent_speaker is False
            and requested_speaker is True
            and parent_summary_tokens == 0
            and parent_without_speaker == requested_without_speaker
        )
    if not architecture_matches:
        raise ModelManifestError("SFT architecture does not match its pretraining parent manifest.")
    requested_capabilities = capabilities_from_config(config)
    migrated_capabilities = dict(manifest.capabilities)

    def migrate_capability(
        flag: str,
        enabled_key: str,
        copied_keys: tuple[str, ...],
    ) -> None:
        if not _strict_boolean(config, flag):
            return
        if migrated_capabilities[enabled_key] or not requested_capabilities[enabled_key]:
            raise ModelManifestError(
                f"{flag} requires an additive false-to-true SFT capability migration."
            )
        for key in copied_keys:
            migrated_capabilities[key] = requested_capabilities[key]

    migrate_capability(
        "initialize_speaker_conditioning",
        "speaker_conditioning",
        ("speaker_conditioning",),
    )
    migrate_capability(
        "initialize_language_conditioning",
        "language_conditioning",
        ("language_conditioning", "supported_languages"),
    )
    if _strict_boolean(config, "initialize_cross_lingual_capability"):
        parent_language_pairs = migrated_capabilities["supported_language_pairs"]
        requested_language_pairs = requested_capabilities["supported_language_pairs"]
        if parent_language_pairs or not requested_language_pairs:
            raise ModelManifestError(
                "initialize_cross_lingual_capability requires an additive SFT reference-"
                "language-pair migration from no declared coverage."
            )
        migrated_capabilities["supported_reference_languages"] = requested_capabilities[
            "supported_reference_languages"
        ]
        migrated_capabilities["supported_language_pairs"] = requested_language_pairs
    migrate_capability(
        "initialize_duration_predictor",
        "duration_predictor",
        (
            "duration_predictor",
            "duration_predictor_hidden_size",
            "duration_predictor_num_layers",
            "duration_predictor_use_speaker",
        ),
    )
    if migrated_capabilities != requested_capabilities:
        changed = sorted(
            key
            for key in requested_capabilities
            if migrated_capabilities[key] != requested_capabilities[key]
        )
        raise ModelManifestError(
            "SFT cannot remove or silently change pretrained capabilities; use only explicit "
            f"validated additive migrations. Mismatched fields: {changed}."
        )
    requested_representation = representation_from_config(config)
    if dict(manifest.representation) != requested_representation:
        raise ModelManifestError(
            "SFT must preserve the pretraining parent's codec/frontend representation."
        )
    if dict(manifest.generation) != generation_from_config(config):
        raise ModelManifestError(
            "SFT must preserve the pretraining parent's generative objective and schedule."
        )
    if dict(manifest.text_conditioning) != text_conditioning_from_config(config):
        raise ModelManifestError(
            "SFT must preserve the pretraining parent's frozen text-conditioning contract."
        )
    return manifest


def validate_sft_parent_manifest(
    checkpoint_path: str | os.PathLike[str],
    config: Mapping[str, Any],
) -> ModelManifest:
    """Validate a pretraining parent and preserve its shape/representation for SFT."""
    checkpoint = Path(checkpoint_path).expanduser()
    manifest = load_model_manifest(checkpoint.parent / MODEL_MANIFEST_FILENAME)
    validate_manifest_weight(manifest, checkpoint)
    return _validate_sft_parent_contract(manifest, config)


def validate_sft_resume_manifest(
    manifest: ModelManifest,
    config: Mapping[str, Any],
) -> None:
    """Require a resumed SFT export to retain its exact topology and representation."""
    if manifest.stage != "sft" or manifest.parent is None:
        raise ModelManifestError(
            "A same-run SFT resume requires an SFT model manifest with its original parent."
        )
    if dict(manifest.architecture) != architecture_from_config(config):
        raise ModelManifestError("Resumed SFT architecture does not match its model manifest.")
    if dict(manifest.capabilities) != capabilities_from_config(config):
        raise ModelManifestError("Resumed SFT capabilities do not match its model manifest.")
    if dict(manifest.representation) != representation_from_config(config):
        raise ModelManifestError(
            "Resumed SFT codec/frontend representation does not match its model manifest."
        )
    if dict(manifest.generation) != generation_from_config(config):
        raise ModelManifestError(
            "Resumed SFT objective/schedule does not match its model manifest."
        )
    if dict(manifest.text_conditioning) != text_conditioning_from_config(config):
        raise ModelManifestError(
            "Resumed SFT frozen text-conditioning contract does not match its manifest."
        )


def validate_grpo_parent_manifest(
    checkpoint_path: str | os.PathLike[str],
) -> ModelManifest:
    """Require a hash-bound NAR-VAE SFT checkpoint descended from scratch pretraining.

    The returned SFT manifest is also the immutable reference-policy identity for a
    fresh GRPO run.  GRPO is not an alternate model-import path: legacy, external,
    pretraining-only, and already post-trained checkpoints fail closed here.
    """
    checkpoint = Path(checkpoint_path).expanduser()
    manifest = load_model_manifest(checkpoint.parent / MODEL_MANIFEST_FILENAME)
    validate_manifest_weight(manifest, checkpoint)
    if manifest.stage != "sft":
        raise ModelManifestError(
            "Fresh GRPO requires a NAR-VAE SFT model manifest; third-party and "
            "pretraining-only checkpoints are not accepted."
        )
    if manifest.parent is None or manifest.parent.get("stage") != "pretrain":
        raise ModelManifestError(
            "The GRPO SFT reference must retain its NAR-VAE scratch-pretraining parent."
        )
    if manifest.representation.get("codec_encoding_policy") != DACVAE_POSTERIOR_SAMPLING_POLICY:
        raise ModelManifestError(
            "Fresh GRPO requires an SFT reference trained with the current seeded DACVAE "
            "posterior policy; legacy representations cannot be relabeled during export."
        )
    is_ema = checkpoint.name in {"ema_model.bin", "pytorch_model_ema.bin"} or (
        "_ema" in checkpoint.stem
    )
    if is_ema:
        base = (
            checkpoint.with_name("pytorch_model.bin")
            if checkpoint.name in {"ema_model.bin", "pytorch_model_ema.bin"}
            else Path(str(checkpoint).replace("_ema", ""))
        )
        if not base.is_file():
            raise ModelManifestError("A GRPO EMA reference requires its full SFT base checkpoint.")
        validate_manifest_weight(manifest, base)
    return manifest


def validate_inference_manifest(
    manifest: ModelManifest,
    *,
    checkpoint_path: str | os.PathLike[str],
    selected_filename: str,
    base_checkpoint_path: str | os.PathLike[str] | None = None,
    base_filename: str | None = None,
    architecture: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    generation: Mapping[str, Any] | None = None,
    text_conditioning: Mapping[str, Any] | None = None,
    codec_source: str | os.PathLike[str] | HubDACVAESource,
    codec_backend: str,
) -> None:
    """Validate all pre-codec inference inputs against one selected model export."""
    validate_manifest_weight(
        manifest,
        checkpoint_path,
        selected_filename=selected_filename,
    )
    if base_checkpoint_path is not None:
        resolved_base_filename = (
            Path(base_checkpoint_path).name if base_filename is None else base_filename
        )
        if (
            Path(base_checkpoint_path).resolve() != Path(checkpoint_path).resolve()
            or resolved_base_filename != selected_filename
        ):
            validate_manifest_weight(
                manifest,
                base_checkpoint_path,
                selected_filename=resolved_base_filename,
            )
    if dict(manifest.architecture) != dict(architecture):
        raise ModelManifestError(
            "Inference architecture arguments do not match the NAR-VAE model manifest."
        )
    if dict(manifest.capabilities) != dict(capabilities):
        raise ModelManifestError("Checkpoint capabilities do not match the NAR-VAE model manifest.")
    if generation is not None and dict(manifest.generation) != dict(generation):
        raise ModelManifestError(
            "Checkpoint objective/schedule does not match the NAR-VAE model manifest."
        )
    if text_conditioning is not None and dict(manifest.text_conditioning) != dict(
        text_conditioning
    ):
        raise ModelManifestError(
            "Checkpoint frozen text-conditioning topology does not match the manifest."
        )
    representation = manifest.representation
    expected_frontend = (
        (FROZEN_TEXT_REPRESENTATION_NAME, FROZEN_TEXT_REPRESENTATION_VERSION)
        if manifest.text_conditioning["mode"] == "frozen_features"
        else (TEXT_FRONTEND_NAME, TEXT_FRONTEND_VERSION)
    )
    if (
        representation["text_frontend_name"] != expected_frontend[0]
        or representation["text_frontend_version"] != expected_frontend[1]
    ):
        raise ModelManifestError(
            "The checkpoint text frontend is incompatible with this NAR-VAE runtime."
        )
    source = describe_dacvae_source(codec_source)
    expected_source = DACVAESourceDescription(
        representation["codec_source"],
        representation["codec_revision"],
        representation["codec_filename"],
    )
    if source != expected_source:
        raise ModelManifestError(
            "The selected DACVAE source does not match the checkpoint representation manifest."
        )
    if codec_backend != representation["codec_backend"]:
        raise ModelManifestError(
            "The selected DACVAE backend does not match the checkpoint representation manifest."
        )


def validate_loaded_codec(manifest: ModelManifest, codec: Any) -> None:
    """Validate resolved codec facts after load, including more than latent width."""
    representation = manifest.representation
    actual = {
        "codec_source": getattr(codec, "nar_vae_codec_identifier", None),
        "codec_backend": getattr(codec, "nar_vae_backend", None),
        "codec_revision": getattr(codec, "nar_vae_codec_revision", None),
        "codec_filename": getattr(codec, "nar_vae_codec_filename", None),
        "codec_sha256": getattr(codec, "nar_vae_codec_sha256", None),
        "sample_rate": getattr(codec, "sample_rate", None),
        "hop_length": getattr(codec, "hop_length", None),
    }
    try:
        actual["latent_width"] = int(codec.quantizer.out_proj.in_channels)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ModelManifestError("The loaded codec does not expose its latent width.") from exc
    expected = {name: representation[name] for name in actual}
    if actual != expected:
        raise ModelManifestError(
            "Loaded DACVAE facts do not match the checkpoint representation manifest: "
            f"actual={actual!r}, expected={expected!r}."
        )


__all__ = [
    "MODEL_MANIFEST_FILENAME",
    "MODEL_MANIFEST_LIBRARY",
    "MODEL_MANIFEST_SCHEMA_VERSION",
    "ModelManifest",
    "ModelManifestError",
    "architecture_from_config",
    "capabilities_from_config",
    "generation_from_config",
    "load_model_manifest",
    "representation_from_config",
    "text_conditioning_from_config",
    "validate_inference_manifest",
    "validate_loaded_codec",
    "validate_grpo_parent_manifest",
    "validate_manifest_weight",
    "validate_sft_parent_manifest",
    "validate_sft_resume_manifest",
    "write_model_manifest",
]
