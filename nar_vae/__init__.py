"""Lightweight public API for NAR-VAE."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._version import __version__

__author__ = "NAR-VAE Team"

_LAZY_EXPORTS = {
    "AdaLN": ("nar_vae.modules", "AdaLN"),
    "AdaLNZero": ("nar_vae.modules", "AdaLNZero"),
    "DACVAE": ("nar_vae.dacvae", "DACVAE"),
    "DACVAE_BACKENDS": ("nar_vae.dacvae", "DACVAE_BACKENDS"),
    "HubDACVAESource": ("nar_vae.dacvae", "HubDACVAESource"),
    "EchoDiT": ("nar_vae.models", "EchoDiT"),
    "EchoDurationPredictor": ("nar_vae.models", "EchoDurationPredictor"),
    "FlowMatchingDataCollator": ("nar_vae.dataset.data_collator", "FlowMatchingDataCollator"),
    "FlowMatchingEchoDiT": ("nar_vae.models", "FlowMatchingEchoDiT"),
    "FlowMatchingLoss": ("nar_vae.losses", "FlowMatchingLoss"),
    "FlowMatchingTTSInference": ("nar_vae.inference", "FlowMatchingTTSInference"),
    "FlowGRPOConfig": ("nar_vae.post_training", "FlowGRPOConfig"),
    "FlowGRPOTrainer": ("nar_vae.post_training", "FlowGRPOTrainer"),
    "GRPOStageConfig": ("nar_vae.post_training", "GRPOStageConfig"),
    "DEFAULT_GRPO_CONFIG_PATH": ("nar_vae.post_training", "DEFAULT_GRPO_CONFIG_PATH"),
    "GenerationConfig": ("nar_vae.configuration", "GenerationConfig"),
    "CheckpointProvenance": ("nar_vae.checkpoint", "CheckpointProvenance"),
    "CrossLingualUnsupportedError": (
        "nar_vae.languages",
        "CrossLingualUnsupportedError",
    ),
    "Language": ("nar_vae.languages", "Language"),
    "LanguagePair": ("nar_vae.languages", "LanguagePair"),
    "HubCheckpointSource": ("nar_vae.checkpoint", "HubCheckpointSource"),
    "LearnedDurationUnsupportedError": (
        "nar_vae.inference",
        "LearnedDurationUnsupportedError",
    ),
    "MultilingualUnsupportedError": (
        "nar_vae.languages",
        "MultilingualUnsupportedError",
    ),
    "ModelPreset": ("nar_vae.model_presets", "ModelPreset"),
    "ODESolver": ("nar_vae.solvers", "ODESolver"),
    "RealtimeTTSInference": ("nar_vae.inference_realtime", "RealtimeTTSInference"),
    "RotaryPositionalEncoding": ("nar_vae.modules", "RotaryPositionalEncoding"),
    "SimpleTTSCollator": ("nar_vae.dataset.data_collator", "SimpleTTSCollator"),
    "TimestepEmbedding": ("nar_vae.modules", "TimestepEmbedding"),
    "VoiceCloningUnsupportedError": (
        "nar_vae.inference",
        "VoiceCloningUnsupportedError",
    ),
    "create_data_collator": ("nar_vae.dataset.data_collator", "create_data_collator"),
    "create_flow_matching_echodit": ("nar_vae.models", "create_flow_matching_echodit"),
    "bind_reward_evaluator_manifest": (
        "nar_vae.post_training",
        "bind_reward_evaluator_manifest",
    ),
    "get_model_preset": ("nar_vae.model_presets", "get_model_preset"),
    "grpo_post_train": ("nar_vae.post_training", "grpo_post_train"),
    "list_model_presets": ("nar_vae.model_presets", "list_model_presets"),
    "normalize_language": ("nar_vae.languages", "normalize_language"),
    "load_dacvae": ("nar_vae.dacvae", "load_dacvae"),
}

__all__ = [
    "__author__",
    "__version__",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str) -> Any:
    """Import optional/heavy public objects only when first requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
