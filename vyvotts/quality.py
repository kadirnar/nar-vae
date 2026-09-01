"""Objective audio and transcript quality checks for generated speech."""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torchaudio

from vyvotts.benchmarking import file_metadata
from vyvotts.languages import DEFAULT_LANGUAGE, LANGUAGE_BY_CODE, LanguagePair, normalize_language


def normalize_transcript(text: str) -> list[str]:
    """Normalize Unicode text into words for a transparent WER calculation."""
    normalized = unicodedata.normalize("NFKC", text).upper()
    words = "".join(
        character if character.isalnum() or character == "'" else " " for character in normalized
    )
    return [word for word in words.split() if word]


def word_error_count(reference: list[str], hypothesis: list[str]) -> int:
    """Return Levenshtein distance over word tokens."""
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference_text: str, hypothesis_text: str) -> tuple[int, float]:
    """Return word-error count and rate for two strings."""
    reference = normalize_transcript(reference_text)
    if not reference:
        raise ValueError("reference text must contain at least one word.")
    errors = word_error_count(reference, normalize_transcript(hypothesis_text))
    return errors, errors / len(reference)


def normalize_characters(text: str) -> list[str]:
    """Normalize Unicode text into alphanumeric characters for transparent CER."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [character for character in normalized if character.isalnum()]


def character_error_rate(reference_text: str, hypothesis_text: str) -> tuple[int, float]:
    """Return character-error count and rate for scripts without word boundaries."""
    reference = normalize_characters(reference_text)
    if not reference:
        raise ValueError("reference text must contain at least one alphanumeric character.")
    errors = word_error_count(reference, normalize_characters(hypothesis_text))
    return errors, errors / len(reference)


def transcript_error_rate(
    reference_text: str,
    hypothesis_text: str,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, float | int | str]:
    """Score a transcript with WER or CER according to its registered script."""
    code = normalize_language(language)
    script = LANGUAGE_BY_CODE[code].script
    if script in {"Hans", "Hant", "Jpan", "Kore"}:
        errors, rate = character_error_rate(reference_text, hypothesis_text)
        metric = "cer"
    else:
        errors, rate = word_error_rate(reference_text, hypothesis_text)
        metric = "wer"
    return {
        "language": code,
        "metric": metric,
        "errors": errors,
        "error_rate": rate,
    }


def speaker_similarity(
    reference_embedding: torch.Tensor,
    synthesized_embedding: torch.Tensor,
) -> float:
    """Return cosine similarity for two fixed-size speaker embeddings."""
    reference = torch.as_tensor(reference_embedding, dtype=torch.float32).flatten()
    synthesized = torch.as_tensor(synthesized_embedding, dtype=torch.float32).flatten()
    if reference.numel() == 0 or synthesized.numel() == 0:
        raise ValueError("speaker embeddings must not be empty.")
    if reference.shape != synthesized.shape:
        raise ValueError(
            "speaker embeddings must have the same flattened shape; "
            f"received {tuple(reference.shape)} and {tuple(synthesized.shape)}."
        )
    if not torch.isfinite(reference).all() or not torch.isfinite(synthesized).all():
        raise ValueError("speaker embeddings must contain only finite values.")
    if reference.norm() == 0 or synthesized.norm() == 0:
        raise ValueError("speaker embeddings must have non-zero magnitude.")
    return float(torch.nn.functional.cosine_similarity(reference, synthesized, dim=0))


def cross_lingual_quality_report(
    reference_text: str,
    hypothesis_text: str,
    reference_speaker_embedding: torch.Tensor,
    synthesized_speaker_embedding: torch.Tensor,
    *,
    target_language: str,
    reference_language: str,
    asr_model: str | None = None,
    speaker_verification_model: str | None = None,
    maximum_error_rate: float | None = None,
    minimum_speaker_similarity: float | None = None,
) -> dict[str, Any]:
    """Combine reproducible intelligibility and source-speaker preservation records.

    Thresholds are optional because they must come from checkpoint- and pair-specific
    baseline distributions rather than library defaults.
    """
    if maximum_error_rate is not None and not 0 <= maximum_error_rate <= 1:
        raise ValueError("maximum_error_rate must be between 0 and 1.")
    if minimum_speaker_similarity is not None and not -1 <= minimum_speaker_similarity <= 1:
        raise ValueError("minimum_speaker_similarity must be between -1 and 1.")
    pair = LanguagePair.resolve(
        target_language,
        reference_language,
        has_reference=True,
    )
    intelligibility = transcript_error_rate(
        reference_text,
        hypothesis_text,
        language=pair.target,
    )
    similarity = speaker_similarity(
        reference_speaker_embedding,
        synthesized_speaker_embedding,
    )
    checks = {
        "intelligibility": (
            intelligibility["error_rate"] <= maximum_error_rate
            if maximum_error_rate is not None
            else None
        ),
        "speaker_similarity": (
            similarity >= minimum_speaker_similarity
            if minimum_speaker_similarity is not None
            else None
        ),
    }
    thresholds_complete = all(value is not None for value in checks.values())
    return {
        "language_pair": {
            "target": pair.target,
            "reference": pair.reference,
            "cross_lingual": pair.is_cross_lingual,
        },
        "models": {
            "asr": asr_model,
            "speaker_verification": speaker_verification_model,
        },
        "intelligibility": intelligibility,
        "speaker_similarity": similarity,
        "thresholds": {
            "maximum_error_rate": maximum_error_rate,
            "minimum_speaker_similarity": minimum_speaker_similarity,
        },
        "checks": checks,
        "passed": all(checks.values()) if thresholds_complete else None,
    }


def audio_metrics(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    """Return basic signal checks that distinguish valid audio from silence/clipping."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError("waveform must have shape [samples] or [channels, samples].")

    mono = waveform.float().mean(dim=0)
    absolute = mono.abs()
    return {
        "samples": int(mono.numel()),
        "sample_rate": int(sample_rate),
        "duration_s": mono.numel() / sample_rate,
        "finite": bool(torch.isfinite(mono).all()),
        "peak": float(absolute.max()),
        "rms": float(mono.square().mean().sqrt()),
        "clipping_ratio": float((absolute >= 0.999).float().mean()),
        "near_silence_ratio": float((absolute < 1e-3).float().mean()),
    }


@lru_cache(maxsize=4)
def _cached_evaluator_artifact(path: str) -> dict[str, str | int]:
    """Hash a locally materialized evaluator weight file once per process."""
    return file_metadata(Path(path), label=Path(path).name)


def _greedy_wav2vec_transcript(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    device: str,
) -> tuple[str, str, dict[str, Any]]:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    mono = waveform.float().mean(dim=0, keepdim=True)
    if sample_rate != bundle.sample_rate:
        mono = torchaudio.functional.resample(mono, sample_rate, bundle.sample_rate)
    with torch.inference_mode():
        emissions, _ = model(mono.to(device))

    labels = bundle.get_labels()
    token_ids = emissions[0].argmax(dim=-1).tolist()
    decoded: list[str] = []
    previous = None
    for token_id in token_ids:
        if token_id != previous and token_id != 0:
            decoded.append(labels[token_id])
        previous = token_id
    checkpoint_filename = getattr(bundle, "_path", None)
    checkpoint_artifact = None
    if checkpoint_filename:
        candidate = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_filename
        if candidate.is_file():
            checkpoint_artifact = _cached_evaluator_artifact(str(candidate.resolve()))
    evaluator = {
        "id": "torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H",
        "implementation": "torchaudio",
        "implementation_version": getattr(torchaudio, "__version__", None),
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_artifact": checkpoint_artifact,
        "revision": checkpoint_artifact["sha256"] if checkpoint_artifact is not None else None,
        "revision_kind": "checkpoint_sha256" if checkpoint_artifact is not None else None,
    }
    return (
        "".join(decoded).replace("|", " ").strip(),
        "WAV2VEC2_ASR_BASE_960H",
        evaluator,
    )


def evaluate_audio_file(
    audio_path: str | Path,
    expected_text: str,
    *,
    device: str | None = None,
    maximum_wer: float | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Transcribe a WAV; return a pass/fail result only with an explicit WER gate."""
    if maximum_wer is not None and not 0 <= maximum_wer <= 1:
        raise ValueError("maximum_wer must be between 0 and 1.")
    language = normalize_language(language)
    if language != DEFAULT_LANGUAGE:
        raise ValueError(
            "The bundled WAV2VEC2_ASR_BASE_960H evaluator is English-only. "
            "Use transcript_error_rate with a language-appropriate ASR transcript."
        )
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = Path(audio_path)
    waveform, sample_rate = torchaudio.load(str(path))
    metrics = audio_metrics(waveform, sample_rate)
    transcript, asr_model, asr_evaluator = _greedy_wav2vec_transcript(
        waveform,
        sample_rate,
        device=selected_device,
    )
    errors, wer = word_error_rate(expected_text, transcript)
    minimum_duration = max(1.0, len(normalize_transcript(expected_text)) * 0.15)
    checks = {
        "finite": metrics["finite"],
        "duration": metrics["duration_s"] >= minimum_duration,
        "not_silent": metrics["rms"] >= 1e-3 and metrics["near_silence_ratio"] < 0.98,
        "not_clipped": metrics["clipping_ratio"] <= 0.01,
        "intelligible": wer <= maximum_wer if maximum_wer is not None else None,
    }
    return {
        "audio_path": str(path),
        "audio_artifact": file_metadata(path, label=str(path)),
        "expected_text": expected_text,
        "language": language,
        "transcript": transcript,
        "asr_model": asr_model,
        "asr_evaluator": asr_evaluator,
        "word_errors": errors,
        "word_error_rate": wer,
        "maximum_word_error_rate": maximum_wer,
        "audio": metrics,
        "checks": checks,
        "passed": all(value is True for value in checks.values())
        if maximum_wer is not None
        else None,
    }


__all__ = [
    "audio_metrics",
    "character_error_rate",
    "cross_lingual_quality_report",
    "evaluate_audio_file",
    "normalize_characters",
    "normalize_transcript",
    "speaker_similarity",
    "transcript_error_rate",
    "word_error_count",
    "word_error_rate",
]
