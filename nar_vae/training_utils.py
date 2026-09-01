"""Shared helpers for EchoDiT training and fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from nar_vae.dacvae_encoding import DACVAE_POSTERIOR_SAMPLING_POLICY
from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_COLUMN,
    REPRESENTATION_CONTRACT_VERSION,
    TEXT_FRONTEND_NAME,
    TEXT_FRONTEND_VERSION,
)
from nar_vae.dataset.sampling import LATENT_NUM_FRAMES_COLUMN, validate_latent_num_frames
from nar_vae.frozen_text_provider import (
    FROZEN_TEXT_REPRESENTATION_NAME,
    FROZEN_TEXT_REPRESENTATION_VERSION,
    FrozenTextProviderSpec,
)
from nar_vae.languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_COUNT,
    LanguagePair,
    language_from_id,
    language_id,
    normalize_language,
    normalize_language_pairs,
    normalize_languages,
    resolve_language_pair_support,
)


@dataclass(frozen=True)
class DurationTrainingOptions:
    """Validated EchoDiT v2 duration architecture and objective settings."""

    enabled: bool
    initialize: bool
    hidden_size: int
    num_layers: int
    uses_speaker: bool
    loss_weight: float
    huber_delta: float
    uses_mas: bool
    alignment_hidden_size: int
    mas_duration_loss_weight: float
    mas_alignment_loss_weight: float


def _boolean_option(config: Mapping[str, object], name: str, default: bool = False) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _integer_option(
    config: Mapping[str, object],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return value


def unwrap_training_model(model: nn.Module) -> nn.Module:
    """Return the canonical module beneath nested DDP/DataParallel/compile wrappers."""
    current = model
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        wrapped = getattr(current, "module", None)
        if isinstance(wrapped, nn.Module) and wrapped is not current:
            current = wrapped
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module) and original is not current:
            current = original
            continue
        break
    return current


def resolve_speaker_training_options(
    config: dict,
    *,
    pretrained_checkpoint: str | None = None,
) -> tuple[bool, bool, int]:
    """Validate and return speaker-conditioning choices shared by both trainers."""
    use_speaker_conditioning = _boolean_option(config, "use_speaker_conditioning")
    initialize_speaker_conditioning = _boolean_option(config, "initialize_speaker_conditioning")
    speaker_patch_size = _integer_option(config, "speaker_patch_size", 4, minimum=1)
    freeze_speaker_encoder = _boolean_option(config, "freeze_speaker_encoder")

    if initialize_speaker_conditioning and not use_speaker_conditioning:
        raise ValueError("initialize_speaker_conditioning requires use_speaker_conditioning: true")
    if initialize_speaker_conditioning and not pretrained_checkpoint:
        raise ValueError(
            "initialize_speaker_conditioning requires a text-only pretrained_checkpoint."
        )
    if initialize_speaker_conditioning and freeze_speaker_encoder:
        raise ValueError(
            "A newly initialized speaker encoder must remain trainable; "
            "set freeze_speaker_encoder: false."
        )
    if use_speaker_conditioning and not pretrained_checkpoint and freeze_speaker_encoder:
        raise ValueError(
            "A from-scratch speaker encoder must remain trainable; "
            "set freeze_speaker_encoder: false."
        )
    return use_speaker_conditioning, initialize_speaker_conditioning, speaker_patch_size


def resolve_language_training_options(
    config: dict,
    *,
    pretrained_checkpoint: str | None = None,
) -> tuple[bool, bool, tuple[str, ...]]:
    """Validate learned language-conditioning settings from a training config."""
    use_language_conditioning = _boolean_option(config, "use_language_conditioning")
    initialize_language_conditioning = _boolean_option(config, "initialize_language_conditioning")
    supported_languages = normalize_languages(config.get("supported_languages"))
    freeze_language_embedding = _boolean_option(config, "freeze_language_embedding")

    if initialize_language_conditioning and not use_language_conditioning:
        raise ValueError(
            "initialize_language_conditioning requires use_language_conditioning: true"
        )
    if not use_language_conditioning and supported_languages != ("en",):
        raise ValueError(
            "supported_languages beyond English require use_language_conditioning: true"
        )
    if initialize_language_conditioning and not pretrained_checkpoint:
        raise ValueError(
            "initialize_language_conditioning requires a monolingual pretrained_checkpoint."
        )
    if initialize_language_conditioning and freeze_language_embedding:
        raise ValueError(
            "A newly initialized language embedding must remain trainable; "
            "set freeze_language_embedding: false."
        )
    if use_language_conditioning and not pretrained_checkpoint and freeze_language_embedding:
        raise ValueError(
            "A from-scratch language embedding must remain trainable; "
            "set freeze_language_embedding: false."
        )
    return (
        use_language_conditioning,
        initialize_language_conditioning,
        supported_languages,
    )


def resolve_reference_language_training_options(
    config: dict,
    *,
    use_speaker_conditioning: bool,
    use_language_conditioning: bool,
    supported_languages: tuple[str, ...] | list[str] | None = None,
    pretrained_checkpoint: str | None = None,
) -> tuple[tuple[str, ...] | None, tuple[LanguagePair, ...], bool]:
    """Validate exact target/reference coverage for multilingual voice cloning."""
    configured_languages = config.get("supported_reference_languages")
    configured_pairs = config.get("supported_language_pairs")
    initialize_capability = _boolean_option(config, "initialize_cross_lingual_capability")
    if configured_languages is None and configured_pairs is None:
        if use_speaker_conditioning and use_language_conditioning:
            raise ValueError(
                "Speaker-conditioned multilingual training requires exact supported_language_pairs."
            )
        if initialize_capability:
            raise ValueError(
                "initialize_cross_lingual_capability requires supported_language_pairs or "
                "supported_reference_languages."
            )
        return None, (), False

    if not use_speaker_conditioning or not use_language_conditioning:
        raise ValueError(
            "Reference-language pair support requires both speaker and language conditioning."
        )
    reference_languages, language_pairs = resolve_language_pair_support(
        supported_languages,
        supported_reference_languages=configured_languages,
        supported_language_pairs=configured_pairs,
    )
    if initialize_capability and not pretrained_checkpoint:
        raise ValueError("initialize_cross_lingual_capability requires a pretrained_checkpoint.")
    return reference_languages, language_pairs, initialize_capability


def resolve_duration_training_options(
    config: dict,
    *,
    use_speaker_conditioning: bool,
    pretrained_checkpoint: str | None = None,
) -> DurationTrainingOptions:
    """Fail closed when a duration head could be saved without being trained."""
    enabled = _boolean_option(config, "use_duration_predictor")
    initialize = _boolean_option(config, "initialize_duration_predictor")
    hidden_size = _integer_option(config, "duration_predictor_hidden_size", 256, minimum=1)
    num_layers = _integer_option(config, "duration_predictor_num_layers", 2, minimum=1)
    uses_speaker = _boolean_option(config, "duration_predictor_use_speaker")
    loss_weight = float(config.get("duration_loss_weight", 0.0))
    huber_delta = float(config.get("duration_huber_delta", 1.0))
    uses_mas = _boolean_option(config, "use_mas_duration")
    alignment_hidden_size = _integer_option(
        config,
        "duration_alignment_hidden_size",
        64,
        minimum=1,
    )
    mas_duration_loss_weight = float(config.get("mas_duration_loss_weight", 0.0))
    mas_alignment_loss_weight = float(config.get("mas_alignment_loss_weight", 0.0))

    if not math.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("duration_huber_delta must be positive.")
    if not all(
        math.isfinite(value)
        for value in (loss_weight, mas_duration_loss_weight, mas_alignment_loss_weight)
    ):
        raise ValueError("Duration loss weights must be finite.")
    if initialize and not enabled:
        raise ValueError("initialize_duration_predictor requires use_duration_predictor: true")
    if initialize and not pretrained_checkpoint:
        raise ValueError("initialize_duration_predictor requires a pretrained_checkpoint.")
    if uses_speaker and not (enabled and use_speaker_conditioning):
        raise ValueError(
            "duration_predictor_use_speaker requires both duration and speaker conditioning."
        )
    if enabled and loss_weight <= 0:
        raise ValueError("A learned duration checkpoint requires a positive duration_loss_weight.")
    if not enabled and loss_weight != 0:
        raise ValueError("duration_loss_weight must be zero when use_duration_predictor is false.")
    if uses_mas and not enabled:
        raise ValueError("use_mas_duration requires use_duration_predictor: true")
    if uses_mas and initialize:
        raise ValueError(
            "MAS cannot be added through initialize_duration_predictor; pretrain the "
            "versioned alignment capability from scratch or use a matching parent checkpoint."
        )
    if uses_mas:
        if alignment_hidden_size <= 0:
            raise ValueError("duration_alignment_hidden_size must be positive.")
        latent_size = config.get("dacvae_latent_dim")
        if latent_size is not None and alignment_hidden_size > int(latent_size):
            raise ValueError("duration_alignment_hidden_size cannot exceed dacvae_latent_dim.")
        if mas_duration_loss_weight <= 0 or mas_alignment_loss_weight <= 0:
            raise ValueError(
                "A MAS checkpoint requires positive mas_duration_loss_weight and "
                "mas_alignment_loss_weight."
            )
    elif mas_duration_loss_weight != 0 or mas_alignment_loss_weight != 0:
        raise ValueError("MAS loss weights must be zero when use_mas_duration is false.")

    return DurationTrainingOptions(
        enabled=enabled,
        initialize=initialize,
        hidden_size=hidden_size,
        num_layers=num_layers,
        uses_speaker=uses_speaker,
        loss_weight=loss_weight,
        huber_delta=huber_delta,
        uses_mas=uses_mas,
        alignment_hidden_size=alignment_hidden_size,
        mas_duration_loss_weight=mas_duration_loss_weight,
        mas_alignment_loss_weight=mas_alignment_loss_weight,
    )


def _dit_block_index(parameter_name: str) -> int | None:
    marker = "dit.blocks."
    if marker not in parameter_name:
        return None
    try:
        return int(parameter_name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None


def freeze_layers(model: nn.Module, config: dict) -> dict[str, float | int]:
    """Apply encoder and leading-DiT-block freezing from a training config."""
    freeze_text = _boolean_option(config, "freeze_text_encoder")
    freeze_speaker = _boolean_option(config, "freeze_speaker_encoder")
    freeze_language = _boolean_option(config, "freeze_language_embedding")
    freeze_first_n = _integer_option(config, "freeze_first_n_layers", 0, minimum=0)

    frozen_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        block_index = _dit_block_index(name)
        should_freeze = (
            (freeze_text and "text_encoder" in name)
            or (freeze_speaker and "speaker_encoder" in name)
            or (freeze_language and "language_embedding" in name)
            or (block_index is not None and block_index < freeze_first_n)
        )
        # Architecture code may already have marked an unreachable or intentionally
        # fixed parameter non-trainable.  User-facing layer-freezing options may add
        # freezes, but must never silently undo that stronger model invariant before
        # the optimizer or DDP wrapper is constructed.
        parameter.requires_grad = parameter.requires_grad and not should_freeze
        if not parameter.requires_grad:
            frozen_params += parameter.numel()
        else:
            trainable_params += parameter.numel()

    total_params = frozen_params + trainable_params
    return {
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "frozen_ratio": frozen_params / total_params if total_params else 0.0,
    }


def _validated_representation_contract(
    value,
    *,
    row_index: int,
    latent_size: int,
    expected_text_frontend_name: str = TEXT_FRONTEND_NAME,
    expected_text_frontend_version: int = TEXT_FRONTEND_VERSION,
) -> dict[str, int | str | None]:
    """Validate one versioned dataset representation without guessing defaults."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Row {row_index} {REPRESENTATION_CONTRACT_COLUMN} must be a mapping.")
    expected_fields = {
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
    }
    if set(value) != expected_fields:
        raise ValueError(
            f"Row {row_index} has an incomplete or unknown representation contract: "
            f"missing={sorted(expected_fields - set(value))}, "
            f"unexpected={sorted(set(value) - expected_fields)}."
        )
    if value["contract_version"] != REPRESENTATION_CONTRACT_VERSION:
        raise ValueError(f"Row {row_index} has an unsupported representation contract version.")
    if (
        value["text_frontend_name"] != expected_text_frontend_name
        or value["text_frontend_version"] != expected_text_frontend_version
    ):
        raise ValueError(f"Row {row_index} was prepared with an incompatible text frontend.")
    for name in ("sample_rate", "hop_length", "latent_width"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"Row {row_index} representation field {name!r} must be a positive integer."
            )
    for name in ("codec_source", "codec_backend"):
        item = value[name]
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(
                f"Row {row_index} representation field {name!r} must be a normalized string."
            )
    revision = value["codec_revision"]
    filename = value["codec_filename"]
    if (revision is None) != (filename is None):
        raise ValueError(f"Row {row_index} codec_revision and codec_filename must be set together.")
    if revision is not None:
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdefABCDEF" for character in revision)
        ):
            raise ValueError(f"Row {row_index} codec_revision must be a full Hub commit SHA.")
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
        ):
            raise ValueError(f"Row {row_index} codec_filename must be repository-relative.")
    artifact_sha256 = value["codec_sha256"]
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ValueError(f"Row {row_index} codec_sha256 must be a lowercase SHA-256.")
    if value["codec_encoding_policy"] != DACVAE_POSTERIOR_SAMPLING_POLICY:
        raise ValueError(
            f"Row {row_index} has an unsupported codec_encoding_policy; re-prepare the "
            "dataset with this NAR-VAE version."
        )
    if value["latent_width"] != latent_size:
        raise ValueError(
            f"Row {row_index} representation latent_width={value['latent_width']} does not "
            f"match dacvae_latent_dim={latent_size}."
        )
    return {name: value[name] for name in sorted(expected_fields)}


def _column_storage_dtype(dataset, name: str) -> str | None:
    """Return the innermost Arrow feature dtype without importing datasets."""
    features = getattr(dataset, "features", None)
    if not isinstance(features, Mapping) or name not in features:
        return None
    feature = features[name]
    visited: set[int] = set()
    while id(feature) not in visited:
        visited.add(id(feature))
        dtype = getattr(feature, "dtype", None)
        if isinstance(dtype, str) and dtype in {"float16", "float32"}:
            return dtype
        nested = getattr(feature, "feature", None)
        if nested is None:
            return dtype if isinstance(dtype, str) else None
        feature = nested
    return None


def _value_storage_dtype(value) -> str | None:
    """Return an exact in-memory floating dtype when the value preserves one."""
    if isinstance(value, torch.Tensor):
        return {
            torch.float16: "float16",
            torch.float32: "float32",
        }.get(value.dtype)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        normalized = str(dtype)
        if normalized in {"float16", "float32"}:
            return normalized
    return None


def validate_tts_dataset(
    dataset,
    *,
    latent_size: int,
    use_speaker_conditioning: bool,
    use_language_conditioning: bool = False,
    supported_languages: tuple[str, ...] | list[str] | None = None,
    supported_reference_languages: tuple[str, ...] | list[str] | None = None,
    supported_language_pairs=None,
    require_language_coverage: bool = False,
    use_mas_duration: bool = False,
    text_conditioning_mode: str = "scratch_tokens",
    conditioning_feature_size: int | None = None,
    frozen_text_provider_spec: FrozenTextProviderSpec | None = None,
    text_vocab_size: int | None = None,
    text_pad_token: int | None = None,
    allow_legacy_representation: bool = True,
    expected_codec_source: str | None = None,
    expected_codec_backend: str | None = None,
    expected_codec_revision: str | None = None,
    expected_codec_filename: str | None = None,
    expected_codec_sha256: str | None = None,
    expected_sample_rate: int | None = None,
    expected_hop_length: int | None = None,
) -> None:
    """Fail before training when prepared latent rows have an unsafe schema."""
    if len(dataset) == 0:
        raise ValueError("The TTS training dataset is empty.")

    column_names = set(getattr(dataset, "column_names", ()) or dataset[0].keys())
    required = {"latents", "conditioning_ids"}
    if text_conditioning_mode not in {"scratch_tokens", "frozen_features"}:
        raise ValueError("text_conditioning_mode must be 'scratch_tokens' or 'frozen_features'.")
    if text_conditioning_mode == "frozen_features":
        if (
            isinstance(conditioning_feature_size, bool)
            or not isinstance(conditioning_feature_size, int)
            or conditioning_feature_size <= 0
        ):
            raise ValueError(
                "frozen_features validation requires a positive conditioning_feature_size."
            )
        if not isinstance(frozen_text_provider_spec, FrozenTextProviderSpec):
            raise ValueError(
                "frozen_features validation requires the exact FrozenTextProviderSpec."
            )
        if frozen_text_provider_spec.conditioning_feature_size != conditioning_feature_size:
            raise ValueError(
                "conditioning_feature_size does not match the frozen text provider spec."
            )
        if (
            isinstance(text_vocab_size, bool)
            or not isinstance(text_vocab_size, int)
            or text_vocab_size <= 0
        ):
            raise ValueError("frozen_features validation requires a positive text_vocab_size.")
        if (
            isinstance(text_pad_token, bool)
            or not isinstance(text_pad_token, int)
            or not 0 <= text_pad_token < text_vocab_size
        ):
            raise ValueError(
                "frozen_features validation requires text_pad_token within text_vocab_size."
            )
        if (
            text_vocab_size != frozen_text_provider_spec.text_vocab_size
            or text_pad_token != frozen_text_provider_spec.pad_token
        ):
            raise ValueError(
                "Training text_vocab_size/pad_token do not match the hashed frozen provider "
                "contract."
            )
        required.update(
            {
                "conditioning_features",
                "conditioning_mask",
                "conditioning_feature_dtype",
                "frozen_text_cache_version",
                "frozen_text_contract_sha256",
                "language_id",
                "token_language_ids",
                "alignment_mask",
            }
        )
    elif frozen_text_provider_spec is not None:
        raise ValueError("scratch_tokens validation cannot receive a frozen text provider spec.")
    if not allow_legacy_representation:
        required.update({"token_language_ids", "alignment_mask"})
    missing = required - column_names
    if missing:
        raise ValueError(f"TTS dataset is missing required columns: {sorted(missing)}")
    frozen_storage_dtype = _column_storage_dtype(dataset, "conditioning_features")
    if (
        text_conditioning_mode == "frozen_features"
        and frozen_storage_dtype is not None
        and frozen_storage_dtype != frozen_text_provider_spec.conditioning_feature_dtype
    ):
        raise ValueError(
            "Dataset conditioning_features Arrow dtype does not match the configured frozen "
            f"cache dtype: {frozen_storage_dtype!r} != "
            f"{frozen_text_provider_spec.conditioning_feature_dtype!r}."
        )
    if use_speaker_conditioning and "speaker_latents" not in column_names:
        raise ValueError(
            "Speaker conditioning is enabled, but the dataset has no speaker_latents column. "
            "Prepare references with the dataset API's speaker_id_column option "
            "or disable speaker conditioning."
        )
    if use_language_conditioning and not {"language", "language_id"}.intersection(column_names):
        raise ValueError(
            "Language conditioning is enabled, but the dataset has no language or "
            "language_id column."
        )

    normalized_supported = normalize_languages(supported_languages)
    supported = set(normalized_supported)
    has_pair_declaration = (
        supported_reference_languages is not None or supported_language_pairs is not None
    )
    if use_speaker_conditioning and use_language_conditioning and not has_pair_declaration:
        raise ValueError(
            "Speaker-conditioned multilingual training requires exact supported_language_pairs."
        )
    if has_pair_declaration and not (use_speaker_conditioning and use_language_conditioning):
        raise ValueError(
            "Reference-language pair support requires both speaker and language conditioning."
        )
    if has_pair_declaration:
        reference_languages, normalized_pairs = resolve_language_pair_support(
            normalized_supported,
            supported_reference_languages=supported_reference_languages,
            supported_language_pairs=supported_language_pairs,
        )
        supported_references: set[str] | None = set(reference_languages)
        supported_pairs: set[tuple[str, str]] | None = {
            pair.as_tuple() for pair in normalized_pairs
        }
    else:
        supported_references = None
        supported_pairs = None
    topology_pairs: set[tuple[str, str]] | None = None
    if require_language_coverage and supported_pairs is not None:
        coverage_hook = getattr(dataset, "reference_pair_coverage", None)
        if callable(coverage_hook):
            raw_topology_pairs = coverage_hook()
        else:
            raw_topology_pairs = getattr(dataset, "available_language_pairs", None)
            if callable(raw_topology_pairs):
                raw_topology_pairs = raw_topology_pairs()
        if raw_topology_pairs is not None:
            raw_topology_pairs = tuple(raw_topology_pairs)
            topology_pairs = (
                {pair.as_tuple() for pair in normalize_language_pairs(raw_topology_pairs)}
                if raw_topology_pairs
                else set()
            )
    seen_languages: set[str] = set()
    seen_reference_languages: set[str] = set()
    seen_language_pairs: set[tuple[str, str]] = set()
    has_frame_metadata = LATENT_NUM_FRAMES_COLUMN in column_names
    has_representation_contract = REPRESENTATION_CONTRACT_COLUMN in column_names
    if not has_representation_contract and not allow_legacy_representation:
        raise ValueError(
            f"TTS dataset has no {REPRESENTATION_CONTRACT_COLUMN}. Re-prepare it with the "
            "current NAR-VAE dataset API or explicitly allow legacy data after an external audit."
        )
    first_representation: dict[str, int | str | None] | None = None

    for index in range(len(dataset)):
        row = dataset[index]
        if has_representation_contract:
            if REPRESENTATION_CONTRACT_COLUMN not in row:
                raise ValueError(f"Row {index} has no {REPRESENTATION_CONTRACT_COLUMN} value.")
            representation = _validated_representation_contract(
                row[REPRESENTATION_CONTRACT_COLUMN],
                row_index=index,
                latent_size=latent_size,
                expected_text_frontend_name=(
                    FROZEN_TEXT_REPRESENTATION_NAME
                    if text_conditioning_mode == "frozen_features"
                    else TEXT_FRONTEND_NAME
                ),
                expected_text_frontend_version=(
                    FROZEN_TEXT_REPRESENTATION_VERSION
                    if text_conditioning_mode == "frozen_features"
                    else TEXT_FRONTEND_VERSION
                ),
            )
            if first_representation is None:
                first_representation = representation
            elif representation != first_representation:
                raise ValueError(
                    f"Row {index} uses a different codec/frontend representation than row 0."
                )
        target = torch.as_tensor(row["latents"])
        if (
            target.ndim != 2
            or target.shape[0] != latent_size
            or target.shape[1] == 0
            or not torch.isfinite(target).all()
        ):
            raise ValueError(
                f"Row {index} has invalid latents; expected finite [{latent_size}, T>0]."
            )
        if has_frame_metadata and LATENT_NUM_FRAMES_COLUMN not in row:
            raise ValueError(f"Row {index} has no {LATENT_NUM_FRAMES_COLUMN} value.")
        if LATENT_NUM_FRAMES_COLUMN in row:
            declared_frames = validate_latent_num_frames(
                row[LATENT_NUM_FRAMES_COLUMN],
                name=f"Row {index} {LATENT_NUM_FRAMES_COLUMN}",
            )
            actual_frames = int(target.shape[1])
            if declared_frames != actual_frames:
                raise ValueError(
                    f"Row {index} has {LATENT_NUM_FRAMES_COLUMN}={declared_frames}, "
                    f"but latents contain {actual_frames} frames."
                )
        conditioning_ids = torch.as_tensor(row["conditioning_ids"])
        conditioning_token_count = len(conditioning_ids)
        if conditioning_token_count == 0:
            raise ValueError(f"Row {index} has empty conditioning_ids.")
        row_features = row.get("conditioning_features")
        if text_conditioning_mode == "frozen_features":
            assert frozen_text_provider_spec is not None
            assert text_vocab_size is not None
            assert text_pad_token is not None
            if conditioning_ids.ndim != 1 or conditioning_ids.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise ValueError(f"Row {index} conditioning_ids must be a rank-one integer axis.")
            if bool((conditioning_ids < 0).any()) or bool(
                (conditioning_ids >= text_vocab_size).any()
            ):
                raise ValueError(f"Row {index} conditioning_ids must be in [0, {text_vocab_size}).")
            if bool((conditioning_ids == text_pad_token).any()):
                raise ValueError(
                    f"Row {index} contains provider PAD on its unpadded conditioning axis."
                )
            expected_contract = frozen_text_provider_spec.contract_sha256
            if row.get("frozen_text_contract_sha256") != expected_contract:
                raise ValueError(
                    f"Row {index} frozen_text_contract_sha256 does not match the configured "
                    "provider contract."
                )
            row_cache_version = row.get("frozen_text_cache_version")
            if (
                isinstance(row_cache_version, bool)
                or not isinstance(row_cache_version, int)
                or row_cache_version != frozen_text_provider_spec.frozen_text_cache_version
            ):
                raise ValueError(
                    f"Row {index} frozen_text_cache_version does not match the configured "
                    "provider cache version."
                )
            expected_dtype = frozen_text_provider_spec.conditioning_feature_dtype
            if row.get("conditioning_feature_dtype") != expected_dtype:
                raise ValueError(
                    f"Row {index} conditioning_feature_dtype does not match the configured "
                    f"cache dtype {expected_dtype!r}."
                )
            value_dtype = _value_storage_dtype(row_features)
            if frozen_storage_dtype is None and value_dtype != expected_dtype:
                raise ValueError(
                    f"Row {index} conditioning_features does not preserve the configured "
                    f"{expected_dtype} storage dtype."
                )
            features = torch.as_tensor(row_features)
            if (
                features.ndim != 2
                or features.shape[0] != conditioning_token_count
                or features.shape[1] != conditioning_feature_size
                or not torch.is_floating_point(features)
                or not torch.isfinite(features).all()
            ):
                raise ValueError(
                    f"Row {index} has invalid conditioning_features; expected finite "
                    f"[{conditioning_token_count}, {conditioning_feature_size}] floating states."
                )
            row_conditioning_mask = torch.as_tensor(row["conditioning_mask"])
            if (
                row_conditioning_mask.ndim != 1
                or row_conditioning_mask.dtype != torch.bool
                or row_conditioning_mask.numel() != conditioning_token_count
                or not bool(row_conditioning_mask.all())
            ):
                raise ValueError(
                    f"Row {index} conditioning_mask must be an all-true boolean provider axis."
                )
            row_language_id = row["language_id"]
            if (
                isinstance(row_language_id, bool)
                or not isinstance(row_language_id, int)
                or row_language_id <= 0
            ):
                raise ValueError(f"Row {index} language_id must be a positive integer.")
        elif row_features is not None:
            raise ValueError(
                f"Row {index} contains conditioning_features but scratch_tokens mode is active."
            )
        token_language_values = row.get("token_language_ids")
        if (
            token_language_values is not None
            and len(token_language_values) != conditioning_token_count
        ):
            raise ValueError(f"Row {index} token_language_ids must match conditioning_ids length.")
        if text_conditioning_mode == "frozen_features":
            token_language_tensor = torch.as_tensor(token_language_values)
            if (
                token_language_tensor.ndim != 1
                or token_language_tensor.dtype
                not in {
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }
                or bool((token_language_tensor < 0).any())
                or bool((token_language_tensor > LANGUAGE_COUNT).any())
            ):
                raise ValueError(
                    f"Row {index} token_language_ids must be rank-one integer IDs in "
                    f"[0, {LANGUAGE_COUNT}]."
                )
            if int(token_language_tensor[0]) != 0 or int(token_language_tensor[-1]) != 0:
                raise ValueError(
                    f"Row {index} BOS/EOS provider tokens must use null language ID 0."
                )
        row_alignment_mask = row.get("alignment_mask")
        if row_alignment_mask is not None:
            if len(row_alignment_mask) != conditioning_token_count:
                raise ValueError(f"Row {index} alignment_mask must match conditioning_ids length.")
            row_alignment_tensor = torch.as_tensor(row_alignment_mask)
            if text_conditioning_mode == "frozen_features":
                row_conditioning_mask = torch.as_tensor(row["conditioning_mask"])
                if (
                    row_alignment_tensor.ndim != 1
                    or row_alignment_tensor.dtype != torch.bool
                    or bool((row_alignment_tensor & ~row_conditioning_mask).any())
                    or bool(row_alignment_tensor[0])
                    or bool(row_alignment_tensor[-1])
                ):
                    raise ValueError(
                        f"Row {index} alignment_mask must be a rank-one boolean subset of "
                        "conditioning_mask with non-aligning BOS/EOS edges."
                    )
            alignable_token_count = int(row_alignment_tensor.sum().item())
            if alignable_token_count <= 0:
                raise ValueError(f"Row {index} alignment_mask selects no acoustic tokens.")
            if text_conditioning_mode == "frozen_features":
                token_language_tensor = torch.as_tensor(token_language_values)
                if bool((row_alignment_tensor & (token_language_tensor == 0)).any()):
                    raise ValueError(
                        f"Row {index} gives an alignable provider token null language ID 0."
                    )
                declared_language_ids = {language_id(code) for code in normalized_supported}
                observed_language_ids = {
                    int(value) for value in token_language_tensor.tolist() if int(value) != 0
                }
                if not observed_language_ids.issubset(declared_language_ids):
                    raise ValueError(
                        f"Row {index} token_language_ids include undeclared languages: "
                        f"{sorted(observed_language_ids - declared_language_ids)}."
                    )
        else:
            alignable_token_count = conditioning_token_count
        if use_mas_duration and int(target.shape[1]) < alignable_token_count:
            raise ValueError(
                f"Row {index} has {target.shape[1]} latent frames for "
                f"{alignable_token_count} alignable tokens. MAS requires at least one "
                "frame for every token selected by alignment_mask."
            )

        if "language" in row:
            row_language = normalize_language(row["language"])
            if "language_id" in row and int(row["language_id"]) != language_id(row_language):
                raise ValueError(f"Row {index} has conflicting language metadata.")
        elif "language_id" in row:
            row_language = language_from_id(int(row["language_id"])).code
        else:
            row_language = DEFAULT_LANGUAGE

        if use_language_conditioning:
            if row_language not in supported:
                raise ValueError(
                    f"Row {index} uses unsupported language {row_language!r}; "
                    f"expected one of {sorted(supported)}."
                )
            seen_languages.add(row_language)
        elif row_language != DEFAULT_LANGUAGE:
            raise ValueError(
                f"Row {index} uses language {row_language!r}, but language conditioning is disabled."
            )

        if use_speaker_conditioning:
            reference = row.get("speaker_latents")
            if reference is None:
                raise ValueError(f"Row {index} has no speaker_latents reference.")
            reference = torch.as_tensor(reference)
            if (
                reference.ndim != 2
                or reference.shape[0] != latent_size
                or reference.shape[1] == 0
                or not torch.isfinite(reference).all()
            ):
                raise ValueError(
                    f"Row {index} has invalid speaker_latents; "
                    f"expected finite [{latent_size}, T>0]."
                )

            # Source-language metadata is intentionally validated separately
            # from target language so cross-lingual rows cannot conflate them.
            if "speaker_language" in row:
                reference_language = normalize_language(row["speaker_language"])
            elif "speaker_language_id" in row:
                reference_language = language_from_id(int(row["speaker_language_id"])).code
            else:
                reference_language = row_language
            if "speaker_language" in row and "speaker_language_id" in row:
                if int(row["speaker_language_id"]) != language_id(reference_language):
                    raise ValueError(f"Row {index} has conflicting speaker language metadata.")
            if supported_references is None and reference_language != row_language:
                raise ValueError(
                    f"Row {index} is cross-lingual ({reference_language!r} reference, "
                    f"{row_language!r} target) but language-pair support is not declared."
                )
            if supported_references is not None:
                row_pair = (row_language, reference_language)
                if supported_pairs is not None and row_pair not in supported_pairs:
                    raise ValueError(
                        f"Row {index} uses unsupported target/reference language pair "
                        f"{row_pair!r}; expected one of {sorted(supported_pairs)}."
                    )
                seen_reference_languages.add(reference_language)
                seen_language_pairs.add(row_pair)

    if first_representation is not None:
        expected_values = {
            "codec_source": expected_codec_source,
            "codec_backend": (
                expected_codec_backend if expected_codec_backend not in (None, "auto") else None
            ),
            "codec_revision": expected_codec_revision,
            "codec_filename": expected_codec_filename,
            "codec_sha256": expected_codec_sha256,
            "sample_rate": expected_sample_rate,
            "hop_length": expected_hop_length,
        }
        for name, expected in expected_values.items():
            if expected is not None and first_representation[name] != expected:
                raise ValueError(
                    f"Dataset representation {name}={first_representation[name]!r} does not "
                    f"match the training configuration ({expected!r})."
                )

    if require_language_coverage and use_language_conditioning and seen_languages != supported:
        raise ValueError(
            "Dataset target-language coverage does not match supported_languages: "
            f"observed={sorted(seen_languages)}, declared={sorted(supported)}."
        )
    coverage_reference_languages = (
        {reference for _, reference in topology_pairs}
        if topology_pairs is not None
        else seen_reference_languages
    )
    coverage_language_pairs = topology_pairs if topology_pairs is not None else seen_language_pairs
    if (
        require_language_coverage
        and supported_references is not None
        and coverage_reference_languages != supported_references
    ):
        raise ValueError(
            "Dataset reference-language coverage does not match "
            "supported_reference_languages: "
            f"observed={sorted(coverage_reference_languages)}, "
            f"declared={sorted(supported_references)}."
        )
    if (
        require_language_coverage
        and supported_pairs is not None
        and coverage_language_pairs != supported_pairs
    ):
        raise ValueError(
            "Dataset target/reference language-pair coverage does not match "
            "supported_language_pairs: "
            f"observed={sorted(coverage_language_pairs)}, declared={sorted(supported_pairs)}."
        )


__all__ = [
    "DurationTrainingOptions",
    "freeze_layers",
    "resolve_language_training_options",
    "resolve_duration_training_options",
    "resolve_reference_language_training_options",
    "resolve_speaker_training_options",
    "unwrap_training_model",
    "validate_tts_dataset",
]
