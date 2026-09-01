"""Dataset APIs with training-only preparation modules loaded lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from nar_vae.tokenization import (
    END_OF_AI,
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    PAD_TOKEN,
    START_OF_AI,
    START_OF_HUMAN,
    START_OF_SPEECH,
    START_OF_TEXT,
    TOKENIZER_LENGTH,
)

from .data_collator import (
    FlowMatchingDataCollator,
    SimpleTTSCollator,
    create_data_collator,
)

_LAZY_EXPORTS = {
    "DatasetPreparer": ("nar_vae.dataset.prepare", "DatasetPreparer"),
    "prepare_dataset": ("nar_vae.dataset.prepare", "prepare_dataset"),
    "EmiliaPreparer": ("nar_vae.dataset.emilia_prepare", "EmiliaPreparer"),
    "prepare_emilia_dataset": (
        "nar_vae.dataset.emilia_prepare",
        "prepare_emilia_dataset",
    ),
    "merge_parts": ("nar_vae.dataset.emilia_prepare", "merge_parts"),
    "prepare_finetune_dataset": (
        "nar_vae.dataset.finetune_prepare",
        "prepare_finetune_dataset",
    ),
    "validate_zero_shot_splits": (
        "nar_vae.dataset.speaker_references",
        "validate_zero_shot_splits",
    ),
    "FrameBudgetBatchSampler": (
        "nar_vae.dataset.sampling",
        "FrameBudgetBatchSampler",
    ),
    "LATENT_NUM_FRAMES_COLUMN": (
        "nar_vae.dataset.sampling",
        "LATENT_NUM_FRAMES_COLUMN",
    ),
    "read_dataset_frame_lengths": (
        "nar_vae.dataset.sampling",
        "read_dataset_frame_lengths",
    ),
    "REPRESENTATION_CONTRACT_COLUMN": (
        "nar_vae.dataset.representation",
        "REPRESENTATION_CONTRACT_COLUMN",
    ),
    "REPRESENTATION_CONTRACT_VERSION": (
        "nar_vae.dataset.representation",
        "REPRESENTATION_CONTRACT_VERSION",
    ),
    "RepresentationContract": (
        "nar_vae.dataset.representation",
        "RepresentationContract",
    ),
    "attach_representation_contract": (
        "nar_vae.dataset.representation",
        "attach_representation_contract",
    ),
    "build_representation_contract": (
        "nar_vae.dataset.representation",
        "build_representation_contract",
    ),
    "PREPARED_DATASET_MANIFEST_FILENAME": (
        "nar_vae.dataset.identity",
        "PREPARED_DATASET_MANIFEST_FILENAME",
    ),
    "DatasetIdentityError": (
        "nar_vae.dataset.identity",
        "DatasetIdentityError",
    ),
    "resolve_hub_dataset_identity": (
        "nar_vae.dataset.identity",
        "resolve_hub_dataset_identity",
    ),
    "resolve_local_prepared_dataset_identity": (
        "nar_vae.dataset.identity",
        "resolve_local_prepared_dataset_identity",
    ),
    "write_prepared_dataset_manifest": (
        "nar_vae.dataset.identity",
        "write_prepared_dataset_manifest",
    ),
}

__all__ = [
    "DatasetPreparer",
    "prepare_dataset",
    "EmiliaPreparer",
    "prepare_emilia_dataset",
    "merge_parts",
    "prepare_finetune_dataset",
    "validate_zero_shot_splits",
    "FrameBudgetBatchSampler",
    "LATENT_NUM_FRAMES_COLUMN",
    "read_dataset_frame_lengths",
    "REPRESENTATION_CONTRACT_COLUMN",
    "REPRESENTATION_CONTRACT_VERSION",
    "RepresentationContract",
    "attach_representation_contract",
    "build_representation_contract",
    "PREPARED_DATASET_MANIFEST_FILENAME",
    "DatasetIdentityError",
    "resolve_hub_dataset_identity",
    "resolve_local_prepared_dataset_identity",
    "write_prepared_dataset_manifest",
    "FlowMatchingDataCollator",
    "SimpleTTSCollator",
    "create_data_collator",
    "TOKENIZER_LENGTH",
    "START_OF_TEXT",
    "END_OF_TEXT",
    "START_OF_SPEECH",
    "END_OF_SPEECH",
    "START_OF_HUMAN",
    "END_OF_HUMAN",
    "START_OF_AI",
    "END_OF_AI",
    "PAD_TOKEN",
]


def __getattr__(name: str) -> Any:
    """Import dataset preparation dependencies only for preparation calls."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
