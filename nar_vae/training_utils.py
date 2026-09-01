"""Shared helpers for EchoDiT training and fine-tuning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_COLUMN,
    REPRESENTATION_CONTRACT_VERSION,
    TEXT_FRONTEND_NAME,
    TEXT_FRONTEND_VERSION,
)
from nar_vae.dataset.sampling import LATENT_NUM_FRAMES_COLUMN, validate_latent_num_frames
from nar_vae.languages import (
    DEFAULT_LANGUAGE,
    LanguagePair,
    language_from_id,
    language_id,
    normalize_language,
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
        value["text_frontend_name"] != TEXT_FRONTEND_NAME
        or value["text_frontend_version"] != TEXT_FRONTEND_VERSION
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
    if value["latent_width"] != latent_size:
        raise ValueError(
            f"Row {row_index} representation latent_width={value['latent_width']} does not "
            f"match dacvae_latent_dim={latent_size}."
        )
    return {name: value[name] for name in sorted(expected_fields)}


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
    missing = required - column_names
    if missing:
        raise ValueError(f"TTS dataset is missing required columns: {sorted(missing)}")
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
        conditioning_token_count = len(row["conditioning_ids"])
        if conditioning_token_count == 0:
            raise ValueError(f"Row {index} has empty conditioning_ids.")
        if use_mas_duration and int(target.shape[1]) < conditioning_token_count:
            raise ValueError(
                f"Row {index} has {target.shape[1]} latent frames for "
                f"{conditioning_token_count} conditioning tokens. MAS requires at least one "
                "frame for every unmasked conditioning token, including boundary tokens."
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
    if (
        require_language_coverage
        and supported_references is not None
        and seen_reference_languages != supported_references
    ):
        raise ValueError(
            "Dataset reference-language coverage does not match "
            "supported_reference_languages: "
            f"observed={sorted(seen_reference_languages)}, "
            f"declared={sorted(supported_references)}."
        )
    if (
        require_language_coverage
        and supported_pairs is not None
        and seen_language_pairs != supported_pairs
    ):
        raise ValueError(
            "Dataset target/reference language-pair coverage does not match "
            "supported_language_pairs: "
            f"observed={sorted(seen_language_pairs)}, declared={sorted(supported_pairs)}."
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
