"""Checkpoint discovery and compatibility helpers for EchoDiT inference."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nar_vae.languages import (
    LANGUAGE_CONDITIONING_VERSION,
    LANGUAGE_COUNT,
    LANGUAGE_REGISTRY_VERSION,
    language_from_id,
)
from nar_vae.models.duration import (
    DURATION_PREDICTOR_VERSION,
    ECHODIT_ARCHITECTURE_VERSION,
    MONOTONIC_ALIGNMENT_VERSION,
)
from nar_vae.voice import (
    CROSS_LINGUAL_CAPABILITY_VERSION,
    SPEAKER_CONDITIONING_VERSION,
    SPEAKER_PATCH_LAYOUT_VERSION,
)

SPEAKER_STATE_KEYS = {
    "null_speaker_embed",
    "speaker_conditioning_version",
    "speaker_patch_layout_version",
    "speaker_patch_size_metadata",
}
LANGUAGE_EMBEDDING_KEY = "dit.text_encoder.language_embedding.weight"
LANGUAGE_STATE_KEYS = {
    LANGUAGE_EMBEDDING_KEY,
    "language_conditioning_version",
    "language_registry_version",
    "language_count_metadata",
    "supported_language_ids_metadata",
}
CROSS_LINGUAL_STATE_KEYS = {
    "cross_lingual_capability_version",
    "reference_language_registry_version",
    "supported_reference_language_ids_metadata",
}
DURATION_METADATA_KEYS = {
    "echodit_architecture_version",
    "duration_predictor_version",
    "duration_predictor_hidden_size_metadata",
    "duration_predictor_num_layers_metadata",
    "duration_predictor_uses_speaker_metadata",
}
DURATION_PARAMETER_PREFIX = "duration_predictor."
MONOTONIC_ALIGNMENT_METADATA_KEYS = {
    "duration_alignment_version",
    "duration_alignment_hidden_size_metadata",
}
MONOTONIC_ALIGNMENT_PARAMETER_PREFIX = "duration_alignment."
MONOTONIC_FRAME_PROJECTION_KEY = "dit.frame_text_proj.weight"
_HUB_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


class LegacySpeakerCheckpointError(RuntimeError):
    """Raised when speaker weights predate the versioned patch layout."""


class LegacyLanguageCheckpointError(RuntimeError):
    """Raised when language-conditioned weights lack stable registry metadata."""


class LegacyCrossLingualCheckpointError(RuntimeError):
    """Raised when reference-language capability metadata is incomplete or invalid."""


class LegacyDurationCheckpointError(RuntimeError):
    """Raised when duration weights lack a complete, supported architecture contract."""


class LegacyMonotonicAlignmentCheckpointError(RuntimeError):
    """Raised when MAS weights lack a complete, supported capability contract."""


@dataclass(frozen=True)
class HubCheckpointSource:
    """Explicit, immutable Hugging Face Hub source for one flow checkpoint pair.

    Plain strings and paths are always treated as local sources. Remote loading is
    available only through this type so a mutable branch or an implicit filename
    cannot silently change the weights used for inference.
    """

    repo_id: str
    revision: str
    base_filename: str
    ema_filename: str
    manifest_filename: str = "nar_vae_manifest.json"

    def __post_init__(self) -> None:
        repo_id = self.repo_id.strip()
        if repo_id != self.repo_id or repo_id.count("/") != 1 or not all(repo_id.split("/")):
            raise ValueError("repo_id must use the non-empty 'owner/name' Hub format.")
        if not _HUB_COMMIT_PATTERN.fullmatch(self.revision):
            raise ValueError(
                "revision must be an explicit 40-character Hub commit hash; "
                "branches and tags are not pinned checkpoint sources."
            )
        for field_name in ("base_filename", "ema_filename", "manifest_filename"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be an explicit non-empty filename.")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{field_name} must be a repository-relative filename.")
        if self.base_filename == self.ema_filename:
            raise ValueError("base_filename and ema_filename must identify different artifacts.")
        if self.manifest_filename in {self.base_filename, self.ema_filename}:
            raise ValueError("manifest_filename must identify a separate JSON artifact.")


@dataclass(frozen=True)
class CheckpointProvenance:
    """Resolved source facts retained alongside loaded checkpoint tensors."""

    kind: str
    source: str
    requested_revision: str | None
    resolved_revision: str | None
    base_filename: str
    ema_filename: str | None
    selected_filename: str
    path: Path
    base_path: Path | None
    manifest_filename: str | None = None
    manifest_path: Path | None = None

    @property
    def commit(self) -> str | None:
        """Return the resolved Hub commit, or ``None`` for local artifacts."""
        return self.resolved_revision


@dataclass(frozen=True)
class _ResolvedFlowCheckpoint:
    path: Path
    base_path: Path | None
    is_ema: bool
    provenance: CheckpointProvenance


@dataclass(frozen=True)
class LanguageCheckpointInfo:
    """Validated multilingual capability stored in one checkpoint."""

    enabled: bool
    supported_languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceLanguageCheckpointInfo:
    """Validated reference-audio language coverage stored in one checkpoint."""

    enabled: bool
    supported_languages: tuple[str, ...] = ()


@dataclass(frozen=True)
class DurationCheckpointInfo:
    """Validated learned-duration capability stored in one checkpoint."""

    enabled: bool
    hidden_size: int = 0
    num_layers: int = 0
    uses_speaker: bool = False


@dataclass(frozen=True)
class MonotonicAlignmentCheckpointInfo:
    """Validated monotonic-alignment capability stored independently of duration v2."""

    enabled: bool
    hidden_size: int = 0
    version: int = 0


def _scalar_metadata(
    state_dict: Mapping[str, torch.Tensor],
    key: str,
    error_type: type[RuntimeError],
) -> int:
    value = state_dict[key]
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if (
        not isinstance(value, torch.Tensor)
        or value.numel() != 1
        or value.dtype not in integer_dtypes
    ):
        raise error_type(f"Checkpoint metadata {key!r} must be an integer scalar tensor.")
    return int(value.item())


def _duration_state_keys(state_dict: Mapping[str, torch.Tensor]) -> set[str]:
    return DURATION_METADATA_KEYS | {
        key for key in state_dict if key.startswith(DURATION_PARAMETER_PREFIX)
    }


def _monotonic_alignment_state_keys(state_dict: Mapping[str, torch.Tensor]) -> set[str]:
    keys = MONOTONIC_ALIGNMENT_METADATA_KEYS | {
        key for key in state_dict if key.startswith(MONOTONIC_ALIGNMENT_PARAMETER_PREFIX)
    }
    if MONOTONIC_FRAME_PROJECTION_KEY in state_dict:
        keys.add(MONOTONIC_FRAME_PROJECTION_KEY)
    return keys


def inspect_duration_capability(
    state_dict: Mapping[str, torch.Tensor],
) -> DurationCheckpointInfo:
    """Validate EchoDiT v2 duration tensors without inferring capability from config."""
    parameter_keys = {key for key in state_dict if key.startswith(DURATION_PARAMETER_PREFIX)}
    present_metadata = DURATION_METADATA_KEYS.intersection(state_dict)
    if not parameter_keys:
        if present_metadata:
            raise LegacyDurationCheckpointError(
                "Duration metadata is present without duration predictor weights."
            )
        return DurationCheckpointInfo(enabled=False)

    missing_metadata = DURATION_METADATA_KEYS - set(state_dict)
    if missing_metadata:
        raise LegacyDurationCheckpointError(
            "Learned-duration weights require complete EchoDiT v2 metadata. "
            f"Missing: {sorted(missing_metadata)}."
        )

    architecture_version = _scalar_metadata(
        state_dict,
        "echodit_architecture_version",
        LegacyDurationCheckpointError,
    )
    predictor_version = _scalar_metadata(
        state_dict,
        "duration_predictor_version",
        LegacyDurationCheckpointError,
    )
    hidden_size = _scalar_metadata(
        state_dict,
        "duration_predictor_hidden_size_metadata",
        LegacyDurationCheckpointError,
    )
    num_layers = _scalar_metadata(
        state_dict,
        "duration_predictor_num_layers_metadata",
        LegacyDurationCheckpointError,
    )
    uses_speaker_value = _scalar_metadata(
        state_dict,
        "duration_predictor_uses_speaker_metadata",
        LegacyDurationCheckpointError,
    )
    if architecture_version != ECHODIT_ARCHITECTURE_VERSION:
        raise LegacyDurationCheckpointError(
            f"Unsupported EchoDiT architecture version: {architecture_version}."
        )
    if predictor_version != DURATION_PREDICTOR_VERSION:
        raise LegacyDurationCheckpointError(
            f"Unsupported duration predictor version: {predictor_version}."
        )
    if hidden_size <= 0 or num_layers <= 0:
        raise LegacyDurationCheckpointError(
            "Duration predictor hidden size and layer count must be positive."
        )
    if uses_speaker_value not in (0, 1):
        raise LegacyDurationCheckpointError(
            "duration_predictor_uses_speaker_metadata must be zero or one."
        )

    text_projection = state_dict.get("duration_predictor.text_projection.weight")
    output_projection = state_dict.get("duration_predictor.output_projection.weight")
    text_embedding = state_dict.get("dit.text_encoder.text_embedding.weight")
    if not isinstance(text_projection, torch.Tensor) or text_projection.ndim != 2:
        raise LegacyDurationCheckpointError("Duration text projection must be rank 2.")
    if not isinstance(output_projection, torch.Tensor) or output_projection.ndim != 2:
        raise LegacyDurationCheckpointError("Duration output projection must be rank 2.")
    if not isinstance(text_embedding, torch.Tensor) or text_embedding.ndim != 2:
        raise LegacyDurationCheckpointError(
            "Duration checkpoints must include the EchoDiT text embedding."
        )
    if tuple(text_projection.shape) != (hidden_size, text_embedding.shape[1]):
        raise LegacyDurationCheckpointError(
            "Duration text projection does not match its stored hidden/text dimensions."
        )
    if tuple(output_projection.shape) != (1, hidden_size):
        raise LegacyDurationCheckpointError(
            "Duration output projection does not match the stored hidden size."
        )

    block_indices: set[int] = set()
    for key in parameter_keys:
        if not key.startswith("duration_predictor.blocks."):
            continue
        try:
            block_indices.add(int(key.split(".", 3)[2]))
        except (IndexError, ValueError) as exc:
            raise LegacyDurationCheckpointError(
                f"Invalid duration predictor block key: {key!r}."
            ) from exc
    if block_indices != set(range(num_layers)):
        raise LegacyDurationCheckpointError(
            "Duration predictor block tensors do not match the stored layer count."
        )

    speaker_projection = state_dict.get("duration_predictor.speaker_projection.weight")
    uses_speaker = bool(uses_speaker_value)
    if uses_speaker:
        infer_speaker_conditioning_from_state_dict(state_dict, False)
        speaker_encoder_projection = state_dict.get("dit.speaker_encoder.in_proj.weight")
        if not isinstance(speaker_projection, torch.Tensor) or speaker_projection.ndim != 2:
            raise LegacyDurationCheckpointError(
                "Speaker-conditioned duration metadata requires its speaker projection."
            )
        if (
            not isinstance(speaker_encoder_projection, torch.Tensor)
            or speaker_encoder_projection.ndim != 2
            or tuple(speaker_projection.shape) != (hidden_size, speaker_encoder_projection.shape[0])
        ):
            raise LegacyDurationCheckpointError(
                "Duration speaker projection does not match the speaker encoder width."
            )
    elif speaker_projection is not None:
        raise LegacyDurationCheckpointError(
            "Text-only duration metadata cannot include a speaker projection."
        )

    return DurationCheckpointInfo(
        enabled=True,
        hidden_size=hidden_size,
        num_layers=num_layers,
        uses_speaker=uses_speaker,
    )


def inspect_monotonic_alignment_capability(
    state_dict: Mapping[str, torch.Tensor],
) -> MonotonicAlignmentCheckpointInfo:
    """Validate the independently versioned MAS head from checkpoint tensors."""
    parameter_keys = {
        key for key in state_dict if key.startswith(MONOTONIC_ALIGNMENT_PARAMETER_PREFIX)
    }
    present_metadata = MONOTONIC_ALIGNMENT_METADATA_KEYS.intersection(state_dict)
    frame_projection = state_dict.get(MONOTONIC_FRAME_PROJECTION_KEY)
    if not parameter_keys:
        if present_metadata:
            raise LegacyMonotonicAlignmentCheckpointError(
                "Monotonic-alignment metadata is present without alignment-head state."
            )
        if frame_projection is not None:
            raise LegacyMonotonicAlignmentCheckpointError(
                "MAS frame-regulator state is present without alignment-head state."
            )
        return MonotonicAlignmentCheckpointInfo(enabled=False)

    missing_metadata = MONOTONIC_ALIGNMENT_METADATA_KEYS - set(state_dict)
    if missing_metadata:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS weights require complete monotonic-alignment metadata. "
            f"Missing: {sorted(missing_metadata)}."
        )
    duration = inspect_duration_capability(state_dict)
    if not duration.enabled:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS capability requires a versioned learned-duration predictor."
        )

    version = _scalar_metadata(
        state_dict,
        "duration_alignment_version",
        LegacyMonotonicAlignmentCheckpointError,
    )
    hidden_size = _scalar_metadata(
        state_dict,
        "duration_alignment_hidden_size_metadata",
        LegacyMonotonicAlignmentCheckpointError,
    )
    if version != MONOTONIC_ALIGNMENT_VERSION:
        raise LegacyMonotonicAlignmentCheckpointError(
            f"Unsupported monotonic-alignment version: {version}."
        )
    if hidden_size <= 0:
        raise LegacyMonotonicAlignmentCheckpointError(
            "Monotonic-alignment hidden size must be positive."
        )

    expected_keys = {
        "duration_alignment.text_statistics.weight",
        "duration_alignment.text_statistics.bias",
        "duration_alignment.latent_projection",
    }
    if parameter_keys != expected_keys:
        raise LegacyMonotonicAlignmentCheckpointError(
            "Monotonic-alignment state does not match the version-1 tensor contract. "
            f"Missing: {sorted(expected_keys - parameter_keys)}; "
            f"unexpected: {sorted(parameter_keys - expected_keys)}."
        )

    text_embedding = state_dict.get("dit.text_encoder.text_embedding.weight")
    input_projection = state_dict.get("dit.in_proj.weight")
    statistics_weight = state_dict["duration_alignment.text_statistics.weight"]
    statistics_bias = state_dict["duration_alignment.text_statistics.bias"]
    latent_projection = state_dict["duration_alignment.latent_projection"]
    tensors = (statistics_weight, statistics_bias, latent_projection, frame_projection)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise LegacyMonotonicAlignmentCheckpointError(
            "Monotonic-alignment and frame-regulator state must contain tensors."
        )
    if not isinstance(text_embedding, torch.Tensor) or text_embedding.ndim != 2:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS checkpoints must include the EchoDiT text embedding."
        )
    if not isinstance(input_projection, torch.Tensor) or input_projection.ndim != 2:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS checkpoints must include the EchoDiT latent input projection."
        )
    if frame_projection.ndim != 2 or tuple(frame_projection.shape) != (
        input_projection.shape[0],
        text_embedding.shape[1],
    ):
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS frame-regulator projection does not match the model/text dimensions."
        )
    if hidden_size > input_projection.shape[1]:
        raise LegacyMonotonicAlignmentCheckpointError(
            "Monotonic-alignment hidden size cannot exceed the latent width."
        )
    expected_shapes = (
        (hidden_size * 2, text_embedding.shape[1]),
        (hidden_size * 2,),
        (hidden_size, input_projection.shape[1]),
    )
    if tuple(statistics_weight.shape) != expected_shapes[0]:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS text-statistics weight does not match stored hidden/text dimensions."
        )
    if tuple(statistics_bias.shape) != expected_shapes[1]:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS text-statistics bias does not match the stored hidden size."
        )
    if tuple(latent_projection.shape) != expected_shapes[2]:
        raise LegacyMonotonicAlignmentCheckpointError(
            "MAS latent projection does not match stored hidden/latent dimensions."
        )
    if any(not torch.is_floating_point(tensor) for tensor in tensors):
        raise LegacyMonotonicAlignmentCheckpointError(
            "Monotonic-alignment state must use floating-point tensors."
        )
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise LegacyMonotonicAlignmentCheckpointError(
            "Monotonic-alignment state must contain only finite values."
        )

    return MonotonicAlignmentCheckpointInfo(
        enabled=True,
        hidden_size=hidden_size,
        version=version,
    )


def _language_id_metadata(
    state_dict: Mapping[str, torch.Tensor],
    key: str,
    error_type: type[RuntimeError],
) -> tuple[int, ...]:
    value = state_dict[key]
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype not in integer_dtypes:
        raise error_type(f"Checkpoint metadata {key!r} must be a rank-1 integer tensor.")
    ids = tuple(int(item) for item in value.tolist())
    if not ids or len(set(ids)) != len(ids):
        raise error_type(f"Checkpoint metadata {key!r} must be non-empty and unique.")
    return ids


def inspect_language_conditioning(
    state_dict: Mapping[str, torch.Tensor],
) -> LanguageCheckpointInfo:
    """Validate versioned language metadata and return supported codes."""
    embedding = state_dict.get(LANGUAGE_EMBEDDING_KEY)
    metadata_keys = LANGUAGE_STATE_KEYS - {LANGUAGE_EMBEDDING_KEY}
    present_metadata = metadata_keys.intersection(state_dict)
    if embedding is None:
        if present_metadata:
            raise LegacyLanguageCheckpointError(
                "Language metadata is present without a language embedding; "
                "the checkpoint is internally inconsistent."
            )
        return LanguageCheckpointInfo(enabled=False)

    missing = metadata_keys - set(state_dict)
    if missing:
        raise LegacyLanguageCheckpointError(
            "This multilingual checkpoint predates versioned language metadata. "
            f"Missing: {sorted(missing)}."
        )
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
        raise LegacyLanguageCheckpointError("Language embedding must be a rank-2 tensor.")

    conditioning_version = _scalar_metadata(
        state_dict,
        "language_conditioning_version",
        LegacyLanguageCheckpointError,
    )
    registry_version = _scalar_metadata(
        state_dict,
        "language_registry_version",
        LegacyLanguageCheckpointError,
    )
    language_count = _scalar_metadata(
        state_dict,
        "language_count_metadata",
        LegacyLanguageCheckpointError,
    )
    if conditioning_version != LANGUAGE_CONDITIONING_VERSION:
        raise LegacyLanguageCheckpointError(
            f"Unsupported language conditioning version: {conditioning_version}."
        )
    if registry_version != LANGUAGE_REGISTRY_VERSION:
        raise LegacyLanguageCheckpointError(
            f"Unsupported language registry version: {registry_version}."
        )
    if language_count != LANGUAGE_COUNT or embedding.shape[0] != LANGUAGE_COUNT + 1:
        raise LegacyLanguageCheckpointError(
            "Checkpoint language count does not match this library's stable registry."
        )

    text_embedding = state_dict.get("dit.text_encoder.text_embedding.weight")
    if not isinstance(text_embedding, torch.Tensor) or text_embedding.ndim != 2:
        raise LegacyLanguageCheckpointError(
            "Language-conditioned checkpoints must include a rank-2 text embedding."
        )
    if embedding.shape[1] != text_embedding.shape[1]:
        raise LegacyLanguageCheckpointError(
            "Language and text embedding widths do not match in this checkpoint."
        )

    supported_ids = _language_id_metadata(
        state_dict,
        "supported_language_ids_metadata",
        LegacyLanguageCheckpointError,
    )
    try:
        supported_languages = tuple(language_from_id(value).code for value in supported_ids)
    except ValueError as exc:
        raise LegacyLanguageCheckpointError(str(exc)) from exc
    return LanguageCheckpointInfo(
        enabled=True,
        supported_languages=supported_languages,
    )


def inspect_reference_language_capability(
    state_dict: Mapping[str, torch.Tensor],
) -> ReferenceLanguageCheckpointInfo:
    """Validate explicit source-reference language coverage without inferring it."""
    present = CROSS_LINGUAL_STATE_KEYS.intersection(state_dict)
    if not present:
        return ReferenceLanguageCheckpointInfo(enabled=False)
    missing = CROSS_LINGUAL_STATE_KEYS - set(state_dict)
    if missing:
        raise LegacyCrossLingualCheckpointError(
            f"Cross-lingual metadata is incomplete. Missing: {sorted(missing)}."
        )
    if "null_speaker_embed" not in state_dict:
        raise LegacyCrossLingualCheckpointError(
            "Reference-language metadata requires versioned speaker conditioning."
        )
    if LANGUAGE_EMBEDDING_KEY not in state_dict:
        raise LegacyCrossLingualCheckpointError(
            "Reference-language metadata requires versioned target-language conditioning."
        )
    infer_speaker_conditioning_from_state_dict(state_dict, False)
    inspect_language_conditioning(state_dict)

    capability_version = _scalar_metadata(
        state_dict,
        "cross_lingual_capability_version",
        LegacyCrossLingualCheckpointError,
    )
    registry_version = _scalar_metadata(
        state_dict,
        "reference_language_registry_version",
        LegacyCrossLingualCheckpointError,
    )
    if capability_version != CROSS_LINGUAL_CAPABILITY_VERSION:
        raise LegacyCrossLingualCheckpointError(
            f"Unsupported cross-lingual capability version: {capability_version}."
        )
    if registry_version != LANGUAGE_REGISTRY_VERSION:
        raise LegacyCrossLingualCheckpointError(
            f"Unsupported reference-language registry version: {registry_version}."
        )
    supported_ids = _language_id_metadata(
        state_dict,
        "supported_reference_language_ids_metadata",
        LegacyCrossLingualCheckpointError,
    )
    try:
        supported_languages = tuple(language_from_id(value).code for value in supported_ids)
    except ValueError as exc:
        raise LegacyCrossLingualCheckpointError(str(exc)) from exc
    return ReferenceLanguageCheckpointInfo(
        enabled=True,
        supported_languages=supported_languages,
    )


def infer_speaker_conditioning_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    fallback: bool,
) -> bool:
    """Infer and validate versioned speaker capability from checkpoint tensors."""
    if "null_speaker_embed" in state_dict:
        conditioning_version = state_dict.get("speaker_conditioning_version")
        patch_layout_version = state_dict.get("speaker_patch_layout_version")
        patch_size = state_dict.get("speaker_patch_size_metadata")
        if conditioning_version is None or patch_layout_version is None or patch_size is None:
            raise LegacySpeakerCheckpointError(
                "This speaker-conditioned checkpoint predates versioned speaker patching. "
                "Its reference layout is ambiguous; migrate or retrain it before cloning."
            )
        null_speaker = state_dict["null_speaker_embed"]
        if not isinstance(null_speaker, torch.Tensor) or null_speaker.ndim != 3:
            raise LegacySpeakerCheckpointError("null_speaker_embed must be a rank-3 tensor.")
        if null_speaker.shape[0] != 1:
            raise LegacySpeakerCheckpointError(
                "null_speaker_embed must contain one reusable reference."
            )
        conditioning_version_value = _scalar_metadata(
            state_dict,
            "speaker_conditioning_version",
            LegacySpeakerCheckpointError,
        )
        patch_layout_version_value = _scalar_metadata(
            state_dict,
            "speaker_patch_layout_version",
            LegacySpeakerCheckpointError,
        )
        patch_size_value = _scalar_metadata(
            state_dict,
            "speaker_patch_size_metadata",
            LegacySpeakerCheckpointError,
        )
        if conditioning_version_value != SPEAKER_CONDITIONING_VERSION:
            raise LegacySpeakerCheckpointError(
                f"Unsupported speaker conditioning version: {conditioning_version_value}."
            )
        if patch_layout_version_value != SPEAKER_PATCH_LAYOUT_VERSION:
            raise LegacySpeakerCheckpointError(
                f"Unsupported speaker patch layout version: {patch_layout_version_value}."
            )
        if patch_size_value <= 0:
            raise LegacySpeakerCheckpointError("Speaker patch size metadata must be positive.")
        if null_speaker.shape[-1] != patch_size_value:
            raise LegacySpeakerCheckpointError(
                "Speaker patch size metadata does not match null_speaker_embed."
            )
        speaker_projection = state_dict.get("dit.speaker_encoder.in_proj.weight")
        if not isinstance(speaker_projection, torch.Tensor) or speaker_projection.ndim != 2:
            raise LegacySpeakerCheckpointError(
                "Speaker-conditioned checkpoints must include the speaker encoder projection."
            )
        expected_projection_width = null_speaker.shape[1] * patch_size_value
        if speaker_projection.shape[1] != expected_projection_width:
            raise LegacySpeakerCheckpointError(
                "Speaker encoder projection width does not match the stored latent/patch layout."
            )
        return True

    orphaned_metadata = SPEAKER_STATE_KEYS.intersection(state_dict)
    if orphaned_metadata:
        raise LegacySpeakerCheckpointError(
            "Speaker version metadata is present without null_speaker_embed; "
            "the checkpoint is internally inconsistent."
        )
    if any(key.startswith("dit.speaker_encoder.") for key in state_dict):
        return False
    return fallback


def _path_relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _resolve_local_flow_checkpoint(
    source: str | os.PathLike[str],
    *,
    prefer_ema: bool,
) -> _ResolvedFlowCheckpoint:
    source_string = os.fspath(source)
    source_path = Path(source_string).expanduser()
    search_root = source_path if source_path.is_dir() else source_path.parent

    if source_path.is_file():
        path = source_path
    elif source_path.is_dir():
        preferred_names = ("pytorch_model_ema.bin", "ema_model.bin") if prefer_ema else ()
        candidates = tuple(source_path / name for name in preferred_names) + (
            source_path / "pytorch_model.bin",
        )
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"No EchoDiT checkpoint found under {source_path}. Expected "
                "'pytorch_model_ema.bin', 'ema_model.bin', or 'pytorch_model.bin'."
            )
    else:
        raise FileNotFoundError(
            f"Checkpoint not found: {source_path}. Plain strings are local paths; "
            "use HubCheckpointSource for an explicit revision-pinned Hub artifact."
        )

    is_ema = _is_ema_checkpoint(path)
    base_path = _base_checkpoint_path(path) if is_ema else path
    provenance = CheckpointProvenance(
        kind="local",
        source=source_string,
        requested_revision=None,
        resolved_revision=None,
        base_filename=_path_relative_to(base_path, search_root),
        ema_filename=_path_relative_to(path, search_root) if is_ema else None,
        selected_filename=_path_relative_to(path, search_root),
        path=path.resolve(),
        base_path=base_path.resolve() if base_path.is_file() else None,
        manifest_filename="nar_vae_manifest.json",
        manifest_path=(search_root / "nar_vae_manifest.json").resolve(),
    )
    return _ResolvedFlowCheckpoint(
        path=path,
        base_path=base_path if is_ema else None,
        is_ema=is_ema,
        provenance=provenance,
    )


def _commit_from_hub_cache_path(path: Path) -> str | None:
    for candidate in (path, path.resolve()):
        parts = candidate.parts
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots" and _HUB_COMMIT_PATTERN.fullmatch(parts[index + 1]):
                return parts[index + 1]
    return None


def _resolve_hub_flow_checkpoint(
    source: HubCheckpointSource,
    *,
    prefer_ema: bool,
) -> _ResolvedFlowCheckpoint:
    from huggingface_hub import hf_hub_download

    base_path = Path(
        hf_hub_download(
            repo_id=source.repo_id,
            filename=source.base_filename,
            revision=source.revision,
        )
    )
    if prefer_ema:
        path = Path(
            hf_hub_download(
                repo_id=source.repo_id,
                filename=source.ema_filename,
                revision=source.revision,
            )
        )
        selected_filename = source.ema_filename
    else:
        path = base_path
        selected_filename = source.base_filename

    manifest_path = Path(
        hf_hub_download(
            repo_id=source.repo_id,
            filename=source.manifest_filename,
            revision=source.revision,
        )
    )

    resolved_commits = tuple(
        commit
        for commit in (
            _commit_from_hub_cache_path(path),
            _commit_from_hub_cache_path(base_path),
            _commit_from_hub_cache_path(manifest_path),
        )
        if commit is not None
    )
    mismatched_commit = next(
        (commit for commit in resolved_commits if commit.lower() != source.revision.lower()),
        None,
    )
    if mismatched_commit is not None:
        raise RuntimeError(
            "Hugging Face Hub resolved a different commit than the pinned checkpoint source: "
            f"{mismatched_commit!r} != {source.revision!r}."
        )
    resolved_revision = resolved_commits[0] if resolved_commits else None
    resolved_revision = resolved_revision or source.revision
    provenance = CheckpointProvenance(
        kind="huggingface_hub",
        source=source.repo_id,
        requested_revision=source.revision,
        resolved_revision=resolved_revision,
        base_filename=source.base_filename,
        ema_filename=source.ema_filename,
        selected_filename=selected_filename,
        path=path.resolve(),
        base_path=base_path.resolve(),
        manifest_filename=source.manifest_filename,
        manifest_path=manifest_path.resolve(),
    )
    return _ResolvedFlowCheckpoint(
        path=path,
        base_path=base_path if prefer_ema else None,
        is_ema=prefer_ema,
        provenance=provenance,
    )


def _resolve_flow_checkpoint(
    source: str | os.PathLike[str] | HubCheckpointSource,
    *,
    prefer_ema: bool,
) -> _ResolvedFlowCheckpoint:
    if isinstance(source, HubCheckpointSource):
        return _resolve_hub_flow_checkpoint(source, prefer_ema=prefer_ema)
    return _resolve_local_flow_checkpoint(source, prefer_ema=prefer_ema)


def resolve_flow_checkpoint(
    source: str | os.PathLike[str] | HubCheckpointSource,
    *,
    prefer_ema: bool = True,
) -> Path:
    """Resolve a local path or an explicit revision-pinned Hub source.

    A plain ``owner/name`` string is a local path and never causes network access.
    Use :class:`HubCheckpointSource` to opt into Hub downloads.
    """
    return _resolve_flow_checkpoint(source, prefer_ema=prefer_ema).path


def _unwrap_state_dict(artifact: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(artifact, Mapping):
        raise ValueError("Expected an EchoDiT checkpoint containing a state-dict mapping.")

    for wrapper_key in ("model_state_dict", "state_dict", "shadow"):
        wrapped = artifact.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            artifact = wrapped
            break

    if not artifact or not all(isinstance(key, str) for key in artifact):
        raise ValueError("EchoDiT checkpoint state dict is empty or has invalid keys.")
    return artifact


def _is_ema_checkpoint(path: Path) -> bool:
    return path.name in {"ema_model.bin", "pytorch_model_ema.bin"} or "_ema" in path.stem


def _base_checkpoint_path(path: Path) -> Path:
    if path.name in {"ema_model.bin", "pytorch_model_ema.bin"}:
        return path.with_name("pytorch_model.bin")
    return Path(str(path).replace("_ema", ""))


def load_pretrained_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | os.PathLike[str],
    *,
    strict: bool = True,
    initialize_speaker_conditioning: bool = False,
    initialize_language_conditioning: bool = False,
    initialize_cross_lingual_capability: bool = False,
    initialize_duration_predictor: bool = False,
    preload_validator: Callable[[Path], None] | None = None,
) -> Any:
    """Load training weights with the same speaker-version checks as inference."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if preload_validator is not None:
        # Re-authenticate the resolved bytes at the deserialization boundary.
        # Fresh SFT uses this to prevent a validated parent from being replaced
        # between startup lineage checks and torch.load.
        preload_validator(path.resolve())

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = _unwrap_state_dict(artifact)
    normalized_state = OrderedDict(
        (
            key.removeprefix("module."),
            value,
        )
        for key, value in state_dict.items()
    )
    checkpoint_reference_languages = inspect_reference_language_capability(normalized_state)
    checkpoint_duration = inspect_duration_capability(normalized_state)
    checkpoint_alignment = inspect_monotonic_alignment_capability(normalized_state)

    model_keys = set(model.state_dict())
    model_uses_speaker = "null_speaker_embed" in model_keys
    if model_uses_speaker:
        checkpoint_uses_speaker = infer_speaker_conditioning_from_state_dict(
            normalized_state,
            False,
        )
        if initialize_speaker_conditioning:
            if checkpoint_uses_speaker:
                raise ValueError(
                    "The pretrained checkpoint is already speaker-conditioned; "
                    "disable initialize_speaker_conditioning."
                )
            strict = False
        elif not checkpoint_uses_speaker:
            raise RuntimeError(
                "The model enables speaker conditioning, but the pretrained checkpoint "
                "does not contain versioned speaker state. Set "
                "initialize_speaker_conditioning: true to add a new trainable speaker path."
            )
    else:
        normalized_state = OrderedDict(
            (key, value) for key, value in normalized_state.items() if key not in SPEAKER_STATE_KEYS
        )

    model_uses_language = LANGUAGE_EMBEDDING_KEY in model_keys
    model_language = inspect_language_conditioning(model.state_dict())
    checkpoint_language = inspect_language_conditioning(normalized_state)
    if model_uses_language:
        if initialize_language_conditioning:
            if checkpoint_language.enabled:
                raise ValueError(
                    "The pretrained checkpoint is already language-conditioned; "
                    "disable initialize_language_conditioning."
                )
            strict = False
        elif not checkpoint_language.enabled:
            raise RuntimeError(
                "The model enables language conditioning, but the pretrained checkpoint "
                "does not contain versioned language state. Set "
                "initialize_language_conditioning: true to add a new trainable language path."
            )
        elif model_language.supported_languages != checkpoint_language.supported_languages:
            raise RuntimeError(
                "The model and pretrained checkpoint declare different supported languages: "
                f"{model_language.supported_languages!r} != "
                f"{checkpoint_language.supported_languages!r}."
            )
    else:
        normalized_state = OrderedDict(
            (key, value)
            for key, value in normalized_state.items()
            if key not in LANGUAGE_STATE_KEYS
        )

    model_reference_languages = inspect_reference_language_capability(model.state_dict())
    if model_reference_languages.enabled:
        if initialize_cross_lingual_capability:
            if checkpoint_reference_languages.enabled:
                raise ValueError(
                    "The pretrained checkpoint already declares reference-language coverage; "
                    "disable initialize_cross_lingual_capability."
                )
            strict = False
        elif not checkpoint_reference_languages.enabled:
            raise RuntimeError(
                "The model declares cross-lingual reference coverage, but the pretrained "
                "checkpoint does not. Set initialize_cross_lingual_capability: true while "
                "fine-tuning on validated cross-lingual rows."
            )
        elif (
            model_reference_languages.supported_languages
            != checkpoint_reference_languages.supported_languages
        ):
            raise RuntimeError(
                "The model and pretrained checkpoint declare different reference languages: "
                f"{model_reference_languages.supported_languages!r} != "
                f"{checkpoint_reference_languages.supported_languages!r}."
            )
    elif checkpoint_reference_languages.enabled and model_uses_speaker and model_uses_language:
        raise RuntimeError(
            "Cross-lingual checkpoint metadata cannot be disabled while preserving speaker and "
            "language conditioning."
        )
    else:
        normalized_state = OrderedDict(
            (key, value)
            for key, value in normalized_state.items()
            if key not in CROSS_LINGUAL_STATE_KEYS
        )

    model_duration = inspect_duration_capability(model.state_dict())
    model_alignment = inspect_monotonic_alignment_capability(model.state_dict())
    if model_duration.enabled:
        if initialize_duration_predictor:
            if model_alignment.enabled:
                raise RuntimeError(
                    "A MAS-duration head cannot be initialized onto a parent checkpoint. "
                    "Pretrain it from scratch or use a parent with matching versioned MAS state."
                )
            if checkpoint_duration.enabled:
                raise ValueError(
                    "The pretrained checkpoint already has a duration predictor; "
                    "disable initialize_duration_predictor."
                )
            strict = False
        elif not checkpoint_duration.enabled:
            raise RuntimeError(
                "The model enables learned duration, but the pretrained checkpoint does not. "
                "Set initialize_duration_predictor: true to train a new EchoDiT v2 head."
            )
        elif model_duration != checkpoint_duration:
            raise RuntimeError(
                "The model and pretrained checkpoint declare different duration architectures: "
                f"{model_duration!r} != {checkpoint_duration!r}."
            )
    elif checkpoint_duration.enabled:
        raise RuntimeError(
            "A checkpoint with learned duration cannot be loaded into a legacy EchoDiT model."
        )

    if model_alignment.enabled:
        if not checkpoint_alignment.enabled:
            raise RuntimeError(
                "The model enables MAS duration, but the parent checkpoint does not contain "
                "versioned monotonic-alignment state. MAS must be scratch-pretrained or match "
                "the parent architecture."
            )
        if model_alignment != checkpoint_alignment:
            raise RuntimeError(
                "The model and pretrained checkpoint declare different MAS architectures: "
                f"{model_alignment!r} != {checkpoint_alignment!r}."
            )
    elif checkpoint_alignment.enabled:
        raise RuntimeError(
            "A checkpoint with versioned MAS duration cannot be loaded into a model without "
            "that alignment capability."
        )

    result = model.load_state_dict(normalized_state, strict=strict)
    if (
        initialize_speaker_conditioning
        or initialize_language_conditioning
        or initialize_cross_lingual_capability
        or initialize_duration_predictor
    ):
        missing = set(result.missing_keys)
        expected_missing = set()
        if initialize_speaker_conditioning:
            expected_missing.update(SPEAKER_STATE_KEYS)
        if initialize_language_conditioning:
            expected_missing.update(LANGUAGE_STATE_KEYS)
        if initialize_cross_lingual_capability:
            expected_missing.update(CROSS_LINGUAL_STATE_KEYS)
        if initialize_duration_predictor:
            expected_missing.update(_duration_state_keys(model.state_dict()))
        if missing != expected_missing or result.unexpected_keys:
            raise RuntimeError(
                "The pretrained checkpoint is incompatible with conditioning initialization. "
                f"Missing: {sorted(missing)}; "
                f"unexpected: {sorted(result.unexpected_keys)}"
            )
    return result


@dataclass
class FlowCheckpoint:
    """Loaded checkpoint tensors and architecture facts needed before model creation."""

    path: Path
    state_dict: Mapping[str, torch.Tensor]
    base_state_dict: Mapping[str, torch.Tensor] | None = None
    is_ema: bool = False
    provenance: CheckpointProvenance | None = None

    @classmethod
    def load(
        cls,
        source: str | os.PathLike[str] | HubCheckpointSource,
        *,
        prefer_ema: bool = True,
        preload_validator: Callable[[CheckpointProvenance], None] | None = None,
    ) -> "FlowCheckpoint":
        resolved = _resolve_flow_checkpoint(source, prefer_ema=prefer_ema)
        if preload_validator is not None:
            # Security-sensitive callers can authenticate every resolved artifact before
            # torch deserializes even a weights-only checkpoint.
            preload_validator(resolved.provenance)
        path = resolved.path
        is_ema = resolved.is_ema
        state_dict = _unwrap_state_dict(torch.load(path, map_location="cpu", weights_only=True))

        base_state_dict = None
        if is_ema:
            base_path = resolved.base_path or _base_checkpoint_path(path)
            if not base_path.is_file():
                raise FileNotFoundError(
                    f"A partial EMA checkpoint requires its full base checkpoint at {base_path}."
                )
            base_state_dict = _unwrap_state_dict(
                torch.load(base_path, map_location="cpu", weights_only=True)
            )

        return cls(
            path=path,
            state_dict=state_dict,
            base_state_dict=base_state_dict,
            is_ema=is_ema,
            provenance=resolved.provenance,
        )

    @property
    def architecture_state_dict(self) -> Mapping[str, torch.Tensor]:
        return self.base_state_dict or self.state_dict

    def infer_text_vocab_size(self, fallback: int) -> int:
        embedding = self.architecture_state_dict.get("dit.text_encoder.text_embedding.weight")
        if isinstance(embedding, torch.Tensor) and embedding.ndim == 2:
            return int(embedding.shape[0])
        return fallback

    def infer_speaker_conditioning(self, fallback: bool) -> bool:
        return infer_speaker_conditioning_from_state_dict(
            self.architecture_state_dict,
            fallback,
        )

    def infer_speaker_patch_size(self, fallback: int) -> int:
        """Return the validated speaker patch size stored with clone-capable weights."""
        value = self.architecture_state_dict.get("speaker_patch_size_metadata")
        if value is None:
            return fallback
        patch_size = int(value.item())
        if patch_size <= 0:
            raise LegacySpeakerCheckpointError("Speaker patch size metadata must be positive.")
        return patch_size

    def language_capability(self) -> LanguageCheckpointInfo:
        """Return validated language support stored with the architecture weights."""
        return inspect_language_conditioning(self.architecture_state_dict)

    def reference_language_capability(self) -> ReferenceLanguageCheckpointInfo:
        """Return validated reference-audio language coverage from architecture weights."""
        return inspect_reference_language_capability(self.architecture_state_dict)

    def duration_capability(self) -> DurationCheckpointInfo:
        """Return the strictly versioned learned-duration architecture facts."""
        return inspect_duration_capability(self.architecture_state_dict)

    def monotonic_alignment_capability(self) -> MonotonicAlignmentCheckpointInfo:
        """Return the strictly versioned MAS architecture facts."""
        return inspect_monotonic_alignment_capability(self.architecture_state_dict)

    def infer_language_conditioning(self, fallback: bool = False) -> bool:
        capability = self.language_capability()
        return capability.enabled if capability.enabled else fallback

    def infer_supported_languages(self) -> tuple[str, ...]:
        return self.language_capability().supported_languages

    def load_into(self, model: torch.nn.Module) -> None:
        """Strictly load a base checkpoint, then overlay partial EMA weights if used."""
        model_keys = set(model.state_dict())

        def compatible_state_dict(
            state_dict: Mapping[str, torch.Tensor],
        ) -> Mapping[str, torch.Tensor]:
            ignored = (
                SPEAKER_STATE_KEYS
                | LANGUAGE_STATE_KEYS
                | CROSS_LINGUAL_STATE_KEYS
                | _duration_state_keys(state_dict)
            ) - model_keys
            if not ignored:
                return state_dict
            return {key: value for key, value in state_dict.items() if key not in ignored}

        if self.base_state_dict is not None:
            model.load_state_dict(compatible_state_dict(self.base_state_dict), strict=True)
            incompatible = model.load_state_dict(
                compatible_state_dict(self.state_dict),
                strict=False,
            )
            if incompatible.unexpected_keys:
                raise RuntimeError(
                    f"EMA checkpoint contains unexpected keys: {incompatible.unexpected_keys}"
                )
            return
        model.load_state_dict(compatible_state_dict(self.state_dict), strict=True)


__all__ = [
    "CheckpointProvenance",
    "DURATION_METADATA_KEYS",
    "DurationCheckpointInfo",
    "FlowCheckpoint",
    "HubCheckpointSource",
    "CROSS_LINGUAL_STATE_KEYS",
    "LANGUAGE_EMBEDDING_KEY",
    "LANGUAGE_STATE_KEYS",
    "LegacySpeakerCheckpointError",
    "LegacyLanguageCheckpointError",
    "LegacyCrossLingualCheckpointError",
    "LegacyDurationCheckpointError",
    "LegacyMonotonicAlignmentCheckpointError",
    "LanguageCheckpointInfo",
    "ReferenceLanguageCheckpointInfo",
    "MONOTONIC_ALIGNMENT_METADATA_KEYS",
    "MonotonicAlignmentCheckpointInfo",
    "SPEAKER_STATE_KEYS",
    "infer_speaker_conditioning_from_state_dict",
    "inspect_language_conditioning",
    "inspect_duration_capability",
    "inspect_monotonic_alignment_capability",
    "inspect_reference_language_capability",
    "load_pretrained_checkpoint",
    "resolve_flow_checkpoint",
]
