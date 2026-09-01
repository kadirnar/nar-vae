"""One-latent-per-utterance storage and deterministic dynamic references."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

from nar_vae.languages import LanguagePair, normalize_language, normalize_language_pairs

from .representation import (
    PREPARED_ROW_VERSION,
    PREPARED_ROW_VERSION_COLUMN,
    REPRESENTATION_CONTRACT_COLUMN,
)
from .sampling import LATENT_NUM_FRAMES_COLUMN, infer_latent_num_frames

AUDIO_SHA256_COLUMN = "audio_sha256"
CONDITIONING_NUM_TOKENS_COLUMN = "conditioning_num_tokens"
SPEAKER_NUM_FRAMES_COLUMN = "speaker_num_frames"
UTTERANCE_ID_COLUMN = "utterance_id"
SPEAKER_ID_COLUMN = "speaker_id"

_SHA256_DIGEST_LENGTH = 64
_REQUIRED_V2_COLUMNS = frozenset(
    {
        "latents",
        LATENT_NUM_FRAMES_COLUMN,
        "conditioning_ids",
        "token_language_ids",
        "alignment_mask",
        CONDITIONING_NUM_TOKENS_COLUMN,
        "language",
        SPEAKER_ID_COLUMN,
        UTTERANCE_ID_COLUMN,
        AUDIO_SHA256_COLUMN,
        PREPARED_ROW_VERSION_COLUMN,
        REPRESENTATION_CONTRACT_COLUMN,
    }
)


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a normalized non-empty string.")
    return value


def namespace_speaker_id(namespace: str, speaker_id: Any) -> str:
    """Namespace a source-local grouping key before corpora are combined."""
    namespace = _nonempty_string(namespace, name="dataset namespace")
    raw = _nonempty_string(str(speaker_id), name="speaker_id")
    return f"{namespace}:{raw}"


def canonical_audio_sha256(audio: Any, sample_rate: int) -> str:
    """Hash decoded PCM plus its rate for exact duplicate/leak detection."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0:
        raise ValueError("audio sample_rate must be a positive integer.")
    samples = np.asarray(audio)
    if samples.ndim == 2:
        # Hugging Face audio is normally [samples]. Accept either common channel
        # orientation deterministically for external manifests.
        channel_axis = 0 if samples.shape[0] <= samples.shape[1] else 1
        samples = samples.astype(np.float32, copy=False).mean(axis=channel_axis)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("audio must contain a non-empty mono or multichannel waveform.")
    samples = np.ascontiguousarray(samples, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"nar-vae-audio-v1\0")
    digest.update(int(sample_rate).to_bytes(8, "big", signed=False))
    digest.update(samples.tobytes())
    return digest.hexdigest()


def stable_utterance_id(
    namespace: str,
    *,
    source_id: Any | None,
    audio_sha256: str,
    language: str,
    text: str,
) -> str:
    """Return a globally namespaced immutable utterance identity."""
    namespace = _nonempty_string(namespace, name="dataset namespace")
    if source_id is not None:
        source = _nonempty_string(str(source_id), name="source utterance_id")
        return f"{namespace}:{source}"
    audio_sha256 = _validated_sha256(audio_sha256, name="audio_sha256")
    payload = json.dumps(
        {
            "audio_sha256": audio_sha256,
            "language": normalize_language(language),
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{namespace}:sha256-{hashlib.sha256(payload).hexdigest()}"


def _validated_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256.")
    return value


def attach_utterance_metadata(
    row: dict[str, Any],
    *,
    dataset_namespace: str,
    speaker_id: Any,
    utterance_id: Any | None,
    audio_sha256: str,
    text: str,
) -> dict[str, Any]:
    """Attach v2 identity and cheap batching metadata to a prepared row."""
    if "language" not in row:
        raise ValueError("A prepared utterance row must declare language.")
    if "conditioning_ids" not in row:
        raise ValueError("A prepared utterance row must contain conditioning_ids.")
    token_count = len(row["conditioning_ids"])
    if token_count <= 0:
        raise ValueError("conditioning_ids must not be empty.")
    if len(row.get("token_language_ids", ())) != token_count:
        raise ValueError("token_language_ids must match conditioning_ids length.")
    if len(row.get("alignment_mask", ())) != token_count:
        raise ValueError("alignment_mask must match conditioning_ids length.")

    audio_sha256 = _validated_sha256(audio_sha256, name="audio_sha256")
    row[SPEAKER_ID_COLUMN] = namespace_speaker_id(dataset_namespace, speaker_id)
    row[UTTERANCE_ID_COLUMN] = stable_utterance_id(
        dataset_namespace,
        source_id=utterance_id,
        audio_sha256=audio_sha256,
        language=row["language"],
        text=text,
    )
    row[AUDIO_SHA256_COLUMN] = audio_sha256
    row[CONDITIONING_NUM_TOKENS_COLUMN] = token_count
    frames = infer_latent_num_frames(row["latents"])
    row[LATENT_NUM_FRAMES_COLUMN] = frames
    # On disk this records how many frames the utterance can provide as a
    # reference candidate. DynamicReferenceDataset replaces it with the actual
    # deterministic crop length for each target row and epoch.
    row[SPEAKER_NUM_FRAMES_COLUMN] = frames
    row[PREPARED_ROW_VERSION_COLUMN] = PREPARED_ROW_VERSION
    return row


def validate_utterance_store(
    dataset: Any, *, reject_duplicate_audio: bool = True
) -> dict[str, Any]:
    """Validate v2 row topology before dynamic reference construction."""
    if len(dataset) == 0:
        raise ValueError("The prepared utterance store is empty.")
    columns = set(getattr(dataset, "column_names", ()) or dataset[0].keys())
    missing = _REQUIRED_V2_COLUMNS - columns
    if missing:
        raise ValueError(f"Prepared utterance store is missing columns: {sorted(missing)}")

    utterance_ids: set[str] = set()
    audio_hashes: dict[str, str] = {}
    speakers: set[str] = set()
    languages: set[str] = set()
    for index in range(len(dataset)):
        row = dataset[index]
        if row.get(PREPARED_ROW_VERSION_COLUMN) != PREPARED_ROW_VERSION:
            raise ValueError(f"Row {index} is not a prepared-row v{PREPARED_ROW_VERSION} record.")
        utterance_id = _nonempty_string(row.get(UTTERANCE_ID_COLUMN), name="utterance_id")
        if utterance_id in utterance_ids:
            raise ValueError(f"Duplicate utterance_id: {utterance_id!r}.")
        utterance_ids.add(utterance_id)
        speaker = _nonempty_string(row.get(SPEAKER_ID_COLUMN), name="speaker_id")
        speakers.add(speaker)
        language = normalize_language(row.get("language"))
        languages.add(language)
        audio_hash = _validated_sha256(row.get(AUDIO_SHA256_COLUMN), name="audio_sha256")
        if reject_duplicate_audio and audio_hash in audio_hashes:
            raise ValueError(
                f"Duplicate audio across {audio_hashes[audio_hash]!r} and {utterance_id!r}."
            )
        audio_hashes[audio_hash] = utterance_id

        latents = np.asarray(row["latents"])
        if latents.ndim != 2 or latents.shape[1] <= 0 or not np.isfinite(latents).all():
            raise ValueError(f"Row {index} has invalid [channels, frames] latents.")
        frames = infer_latent_num_frames(latents)
        if row[LATENT_NUM_FRAMES_COLUMN] != frames:
            raise ValueError(f"Row {index} latent_num_frames does not match its latents.")
        token_count = len(row["conditioning_ids"])
        if row[CONDITIONING_NUM_TOKENS_COLUMN] != token_count:
            raise ValueError(f"Row {index} conditioning_num_tokens is incorrect.")
        if len(row["token_language_ids"]) != token_count:
            raise ValueError(f"Row {index} token_language_ids length is incorrect.")
        if len(row["alignment_mask"]) != token_count or not any(row["alignment_mask"]):
            raise ValueError(f"Row {index} alignment_mask is invalid.")

    return {
        "utterances": len(utterance_ids),
        "speakers": len(speakers),
        "languages": tuple(sorted(languages)),
    }


def _stable_random_u64(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


class DynamicReferenceDataset:
    """Attach deterministic cached-latent voice references at item access.

    Speaker and utterance IDs never leave this wrapper through the model
    collator. Reference choices depend only on stable metadata, ``seed`` and
    ``epoch``; worker count, rank and access order cannot change them.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        supported_language_pairs: Sequence[LanguagePair | Sequence[str]] | None = None,
        seed: int = 0,
        min_reference_seconds: float = 3.0,
        short_reference_max_seconds: float = 8.0,
        max_reference_seconds: float = 12.0,
        short_reference_probability: float = 0.8,
        speaker_patch_size: int = 4,
        strict: bool = True,
    ) -> None:
        validate_utterance_store(dataset)
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        for name, value in (
            ("min_reference_seconds", min_reference_seconds),
            ("short_reference_max_seconds", short_reference_max_seconds),
            ("max_reference_seconds", max_reference_seconds),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not min_reference_seconds <= short_reference_max_seconds <= max_reference_seconds:
            raise ValueError("Reference durations must satisfy min <= short_reference_max <= max.")
        if not 0 <= short_reference_probability <= 1:
            raise ValueError("short_reference_probability must be between zero and one.")
        if isinstance(speaker_patch_size, bool) or not isinstance(speaker_patch_size, Integral):
            raise ValueError("speaker_patch_size must be a positive integer.")
        if speaker_patch_size <= 0:
            raise ValueError("speaker_patch_size must be a positive integer.")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a boolean.")

        self.dataset = dataset
        self.seed = int(seed)
        self.min_reference_seconds = float(min_reference_seconds)
        self.short_reference_max_seconds = float(short_reference_max_seconds)
        self.max_reference_seconds = float(max_reference_seconds)
        self.short_reference_probability = float(short_reference_probability)
        self.speaker_patch_size = int(speaker_patch_size)
        self.strict = strict
        self._epoch = multiprocessing.Value("q", 0, lock=True)

        normalized_pairs = (
            normalize_language_pairs(supported_language_pairs)
            if supported_language_pairs is not None
            else ()
        )
        self.supported_language_pairs = normalized_pairs
        self._references_by_speaker_language: dict[tuple[str, str], list[int]] = defaultdict(list)
        self._metadata: list[tuple[str, str, str, str]] = []
        self._reference_frame_counts: list[int] = []
        self._reference_frame_rates: list[float] = []
        for index in range(len(dataset)):
            row = dataset[index]
            speaker = row[SPEAKER_ID_COLUMN]
            language = normalize_language(row["language"])
            utterance = row[UTTERANCE_ID_COLUMN]
            audio_hash = row[AUDIO_SHA256_COLUMN]
            self._metadata.append((speaker, language, utterance, audio_hash))
            self._reference_frame_counts.append(int(row[LATENT_NUM_FRAMES_COLUMN]))
            self._reference_frame_rates.append(self._frame_rate(row))
            self._references_by_speaker_language[(speaker, language)].append(index)

        self._candidate_languages: list[tuple[str, ...]] = []
        available_pairs: set[tuple[str, str]] = set()
        targets_without_references: list[tuple[str, str]] = []
        for target_index, (speaker, target_language, utterance, audio_hash) in enumerate(
            self._metadata
        ):
            if normalized_pairs:
                allowed = tuple(
                    pair.reference
                    for pair in normalized_pairs
                    if pair.target == target_language and pair.reference is not None
                )
            else:
                allowed = (target_language,)
            available = tuple(
                language
                for language in dict.fromkeys(allowed)
                if any(
                    candidate != target_index
                    and self._metadata[candidate][2] != utterance
                    and self._metadata[candidate][3] != audio_hash
                    for candidate in self._references_by_speaker_language.get(
                        (speaker, language), ()
                    )
                )
            )
            available_pairs.update((target_language, reference) for reference in available)
            if not available:
                targets_without_references.append((utterance, target_language))
            self._candidate_languages.append(available)

        if normalized_pairs:
            declared_pairs = {pair.as_tuple() for pair in normalized_pairs}
            missing_pairs = declared_pairs - available_pairs
            if missing_pairs:
                raise ValueError(
                    "Declared target/reference language pairs have no valid cached-latent "
                    f"example: {sorted(missing_pairs)}."
                )
            # Do not expose undeclared pairs even if the underlying speakers can
            # form them; checkpoint capability follows the explicit contract.
            available_pairs &= declared_pairs
        self.available_language_pairs = tuple(
            LanguagePair(target, reference) for target, reference in sorted(available_pairs)
        )
        if targets_without_references and strict:
            utterance, target_language = targets_without_references[0]
            raise ValueError(
                f"Utterance {utterance!r} has no same-speaker, different-utterance "
                f"reference for target language {target_language!r}."
            )

    def reference_pair_coverage(self) -> tuple[LanguagePair, ...]:
        """Return deterministic candidate coverage, independent of sampled epoch."""
        return self.available_language_pairs

    @property
    def epoch(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    @property
    def column_names(self) -> list[str]:
        base = list(getattr(self.dataset, "column_names", ()) or self.dataset[0].keys())
        for column in (
            "speaker_latents",
            "speaker_language",
            "reference_utterance_id",
            "reference_audio_sha256",
            "reference_segment_lengths",
            SPEAKER_NUM_FRAMES_COLUMN,
        ):
            if column not in base:
                base.append(column)
        return base

    @property
    def features(self):
        """Preserve Arrow storage dtypes for strict training preflight."""
        return getattr(self.dataset, "features", None)

    @property
    def _fingerprint(self) -> str | None:
        """Expose the immutable base-data fingerprint to Trainer tooling."""
        return getattr(self.dataset, "_fingerprint", None)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"version": 1, "seed": self.seed, "epoch": self.epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("Unsupported dynamic-reference dataset state.")
        if state.get("seed") != self.seed:
            raise ValueError("Dynamic-reference dataset seed does not match the saved state.")
        self.set_epoch(state.get("epoch"))

    def __len__(self) -> int:
        return len(self.dataset)

    def get_length_metadata(self, column: str) -> tuple[int, ...]:
        """Return deterministic cost metadata for the active epoch.

        ``speaker_num_frames`` is recomputed because the reference crop changes
        with :meth:`set_epoch`; target and token lengths are immutable columns.
        """
        if column not in {
            LATENT_NUM_FRAMES_COLUMN,
            CONDITIONING_NUM_TOKENS_COLUMN,
            SPEAKER_NUM_FRAMES_COLUMN,
        }:
            raise KeyError(column)
        if column == SPEAKER_NUM_FRAMES_COLUMN:
            values = (
                self._planned_crop(
                    self._reference_index(index)[0],
                    self._metadata[index][2],
                )[0]
                for index in range(len(self))
            )
        else:
            try:
                values = iter(self.dataset[column])
            except (KeyError, TypeError, AttributeError):
                values = (self.dataset[index][column] for index in range(len(self)))
        normalized = tuple(int(value) for value in values)
        if any(value <= 0 for value in normalized):
            raise ValueError(f"{column} must contain only positive lengths.")
        return normalized

    def _reference_index(self, target_index: int) -> tuple[int, str]:
        speaker, _, utterance, audio_hash = self._metadata[target_index]
        languages = self._candidate_languages[target_index]
        if not languages:
            raise IndexError(f"Target row {target_index} has no valid voice reference.")
        random_value = _stable_random_u64(self.seed, self.epoch, utterance, "language")
        reference_language = languages[random_value % len(languages)]
        candidates = [
            index
            for index in self._references_by_speaker_language[(speaker, reference_language)]
            if index != target_index
            and self._metadata[index][2] != utterance
            and self._metadata[index][3] != audio_hash
        ]
        random_value = _stable_random_u64(self.seed, self.epoch, utterance, "utterance")
        return candidates[random_value % len(candidates)], reference_language

    @staticmethod
    def _frame_rate(row: Mapping[str, Any]) -> float:
        contract = row.get(REPRESENTATION_CONTRACT_COLUMN)
        if not isinstance(contract, Mapping):
            raise ValueError("A dynamic reference row has no representation contract.")
        sample_rate = contract.get("sample_rate")
        hop_length = contract.get("hop_length")
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, Integral)
            or sample_rate <= 0
            or isinstance(hop_length, bool)
            or not isinstance(hop_length, Integral)
            or hop_length <= 0
        ):
            raise ValueError("A dynamic reference row has invalid codec frame-rate metadata.")
        return float(sample_rate) / float(hop_length)

    def _planned_crop(self, reference_index: int, target_utterance: str) -> tuple[int, int]:
        available_frames = self._reference_frame_counts[reference_index]
        frame_rate = self._reference_frame_rates[reference_index]
        minimum_frames = max(
            self.speaker_patch_size,
            math.ceil(self.min_reference_seconds * frame_rate),
        )
        if available_frames < minimum_frames and self.strict:
            raise ValueError(
                f"Reference {self._metadata[reference_index][2]!r} has {available_frames} frames; "
                f"at least {minimum_frames} are required."
            )

        probability = _stable_random_u64(
            self.seed, self.epoch, target_utterance, "duration-bucket"
        ) / float(2**64 - 1)
        if probability < self.short_reference_probability:
            lower = self.min_reference_seconds
            upper = self.short_reference_max_seconds
        else:
            lower = self.short_reference_max_seconds
            upper = self.max_reference_seconds
        fraction = _stable_random_u64(
            self.seed, self.epoch, target_utterance, "duration-value"
        ) / float(2**64 - 1)
        seconds = lower + (upper - lower) * fraction
        requested = max(self.speaker_patch_size, round(seconds * frame_rate))
        crop_frames = min(available_frames, requested)
        if crop_frames < self.speaker_patch_size:
            raise ValueError("Reference latent is shorter than one speaker patch.")
        maximum_start = available_frames - crop_frames
        start = (
            _stable_random_u64(self.seed, self.epoch, target_utterance, "crop")
            % (maximum_start + 1)
            if maximum_start
            else 0
        )
        return crop_frames, start

    def _crop_reference(
        self,
        row: Mapping[str, Any],
        reference_index: int,
        target_utterance: str,
    ) -> np.ndarray:
        latent = np.asarray(row["latents"], dtype=np.float32)
        crop_frames, start = self._planned_crop(reference_index, target_utterance)
        if latent.ndim != 2 or latent.shape[1] != self._reference_frame_counts[reference_index]:
            raise ValueError("Reference latent shape changed after utterance-store validation.")
        return np.ascontiguousarray(latent[:, start : start + crop_frames], dtype=np.float32)

    def _dynamic_item(self, index: int) -> dict[str, Any]:
        target = dict(self.dataset[index])
        reference_index, reference_language = self._reference_index(index)
        reference = self.dataset[reference_index]
        speaker_latents = self._crop_reference(
            reference,
            reference_index,
            target[UTTERANCE_ID_COLUMN],
        )
        target["speaker_latents"] = speaker_latents
        target["speaker_language"] = reference_language
        target["reference_utterance_id"] = reference[UTTERANCE_ID_COLUMN]
        target["reference_audio_sha256"] = reference[AUDIO_SHA256_COLUMN]
        target["reference_segment_lengths"] = [int(speaker_latents.shape[1])]
        target[SPEAKER_NUM_FRAMES_COLUMN] = int(speaker_latents.shape[1])
        return target

    def __getitem__(self, index: int | str):
        if isinstance(index, str):
            if index == SPEAKER_NUM_FRAMES_COLUMN:
                return list(self.get_length_metadata(index))
            return self.dataset[index]
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("DynamicReferenceDataset indices must be integers or column names.")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self)
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        return self._dynamic_item(normalized)


__all__ = [
    "AUDIO_SHA256_COLUMN",
    "CONDITIONING_NUM_TOKENS_COLUMN",
    "DynamicReferenceDataset",
    "SPEAKER_ID_COLUMN",
    "SPEAKER_NUM_FRAMES_COLUMN",
    "UTTERANCE_ID_COLUMN",
    "attach_utterance_metadata",
    "canonical_audio_sha256",
    "namespace_speaker_id",
    "stable_utterance_id",
    "validate_utterance_store",
]
