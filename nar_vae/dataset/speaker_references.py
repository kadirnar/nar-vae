"""Safe same-speaker reference selection for voice-cloning datasets."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from nar_vae.languages import normalize_language


def build_speaker_index(dataset, speaker_id_column: str) -> dict[Any, list[int]]:
    """Index dataset rows by speaker without selecting target utterances as references."""
    column_names = set(getattr(dataset, "column_names", ()))
    if column_names and speaker_id_column not in column_names:
        raise ValueError(f"Speaker ID column not found: {speaker_id_column!r}")

    speaker_ids = dataset[speaker_id_column]
    index: dict[Any, list[int]] = defaultdict(list)
    for row_index, speaker_id in enumerate(speaker_ids):
        if speaker_id is not None:
            index[speaker_id].append(row_index)
    if not index:
        raise ValueError(f"Speaker ID column {speaker_id_column!r} contains no usable values.")
    return dict(index)


def select_reference_indices(
    speaker_index: Mapping[Any, Sequence[int]],
    *,
    speaker_id: Any,
    target_index: int,
    maximum_utterances: int,
    seed: int,
    session_ids: Mapping[int, Any] | Sequence[Any] | None = None,
    target_session_id: Any = None,
    require_different_session: bool = False,
) -> list[int]:
    """Choose deterministic same-speaker peers, always excluding the target row.

    Session filtering is opt-in and is useful for preventing a cloning model from
    relying on recording-channel identity instead of speaker identity.
    """
    if maximum_utterances <= 0:
        raise ValueError("maximum_utterances must be positive")
    if require_different_session and session_ids is None:
        raise ValueError("Cross-session references require session_ids.")
    if require_different_session and target_session_id in (None, ""):
        raise ValueError("Cross-session references require a target_session_id.")

    candidates = [index for index in speaker_index.get(speaker_id, ()) if index != target_index]
    if require_different_session:
        candidates = [
            index
            for index in candidates
            if session_ids[index] not in (None, "") and session_ids[index] != target_session_id
        ]
    random.Random(seed + target_index).shuffle(candidates)
    return candidates[:maximum_utterances]


def validate_zero_shot_splits(
    datasets: Mapping[str, Any],
    *,
    speaker_id_column: str = "speaker_id",
    utterance_id_column: str | None = None,
    language_column: str = "language",
    session_id_column: str | None = None,
    require_cross_session_references: bool = False,
    required_splits: Sequence[str] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Validate a speaker-disjoint raw-data contract for zero-shot cloning.

    Every speaker needs at least two utterances so a target can use a different
    same-speaker reference. When cross-session references are required, every
    speaker also needs at least two non-empty recording/session identifiers.
    """
    if not datasets:
        raise ValueError("Zero-shot datasets cannot be empty.")
    missing_splits = set(required_splits) - set(datasets)
    if missing_splits:
        raise ValueError(f"Zero-shot datasets are missing splits: {sorted(missing_splits)}")
    if require_cross_session_references and session_id_column is None:
        raise ValueError("Cross-session references require session_id_column.")

    speakers_by_split: dict[str, set[Any]] = {}
    languages: set[str] = set()
    utterance_locations: dict[Any, tuple[str, int]] = {}
    total_utterances = 0

    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"Zero-shot split {split!r} is empty.")
        column_names = set(getattr(dataset, "column_names", ()) or dataset[0].keys())
        required_columns = {speaker_id_column, language_column}
        if utterance_id_column is not None:
            required_columns.add(utterance_id_column)
        if session_id_column is not None:
            required_columns.add(session_id_column)
        missing_columns = required_columns - column_names
        if missing_columns:
            raise ValueError(
                f"Zero-shot split {split!r} is missing columns: {sorted(missing_columns)}"
            )

        speaker_counts: dict[Any, int] = defaultdict(int)
        speaker_sessions: dict[Any, set[Any]] = defaultdict(set)
        for index in range(len(dataset)):
            row = dataset[index]
            speaker_id = row.get(speaker_id_column)
            if speaker_id in (None, ""):
                raise ValueError(f"Split {split!r} row {index} has no speaker ID.")
            speaker_counts[speaker_id] += 1
            languages.add(normalize_language(row.get(language_column)))

            if utterance_id_column is not None:
                utterance_id = row.get(utterance_id_column)
                if utterance_id in (None, ""):
                    raise ValueError(f"Split {split!r} row {index} has no utterance ID.")
                if utterance_id in utterance_locations:
                    previous = utterance_locations[utterance_id]
                    raise ValueError(
                        f"Utterance ID {utterance_id!r} is duplicated at "
                        f"{previous} and {(split, index)}."
                    )
                utterance_locations[utterance_id] = (split, index)

            if session_id_column is not None:
                session_id = row.get(session_id_column)
                if session_id not in (None, ""):
                    speaker_sessions[speaker_id].add(session_id)

        insufficient = sorted(
            (str(speaker_id) for speaker_id, count in speaker_counts.items() if count < 2)
        )
        if insufficient:
            raise ValueError(
                f"Split {split!r} speakers need at least two utterances: {insufficient}"
            )
        if require_cross_session_references:
            single_session = sorted(
                str(speaker_id)
                for speaker_id in speaker_counts
                if len(speaker_sessions[speaker_id]) < 2
            )
            if single_session:
                raise ValueError(
                    f"Split {split!r} speakers need at least two sessions: {single_session}"
                )

        speakers_by_split[split] = set(speaker_counts)
        total_utterances += len(dataset)

    split_names = list(speakers_by_split)
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            overlap = speakers_by_split[left_split] & speakers_by_split[right_split]
            if overlap:
                raise ValueError(
                    f"Speakers overlap between {left_split!r} and {right_split!r}: "
                    f"{sorted(map(str, overlap))}"
                )

    return {
        "splits": tuple(split_names),
        "utterances": total_utterances,
        "speakers": sum(len(speakers) for speakers in speakers_by_split.values()),
        "languages": tuple(sorted(languages)),
        "cross_session_references": require_cross_session_references,
    }


def collect_reference_audio(
    dataset,
    indices: Sequence[int],
    *,
    audio_column: str,
) -> list[tuple[np.ndarray, int]]:
    """Load valid audio payloads for selected reference rows."""
    references = []
    for index in indices:
        audio = dataset[index].get(audio_column)
        if not isinstance(audio, Mapping) or "array" not in audio:
            continue
        samples = np.asarray(audio["array"])
        sample_rate = int(audio.get("sampling_rate", 0))
        if samples.size and sample_rate > 0:
            references.append((samples, sample_rate))
    return references


def collect_reference_audio_with_language(
    dataset,
    indices: Sequence[int],
    *,
    audio_column: str,
    language_column: str | None,
    fallback_language: str,
) -> tuple[list[tuple[np.ndarray, int]], str]:
    """Load a single-language reference set and return its independent source code."""
    selected_language = None
    references = []
    for index in indices:
        row = dataset[index]
        row_language = normalize_language(
            row.get(language_column, fallback_language) if language_column else fallback_language
        )
        if selected_language is None:
            selected_language = row_language
        if row_language != selected_language:
            continue
        audio = row.get(audio_column)
        if not isinstance(audio, Mapping) or "array" not in audio:
            continue
        samples = np.asarray(audio["array"])
        sample_rate = int(audio.get("sampling_rate", 0))
        if samples.size and sample_rate > 0:
            references.append((samples, sample_rate))
    return references, selected_language or normalize_language(fallback_language)


__all__ = [
    "build_speaker_index",
    "collect_reference_audio",
    "collect_reference_audio_with_language",
    "select_reference_indices",
    "validate_zero_shot_splits",
]
