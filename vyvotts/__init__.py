"""Compatibility API for the former ``vyvotts`` package name.

New applications should import :mod:`nar_vae`. This namespace remains lazy and supported while
existing users migrate.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ._version import __version__

__author__ = "NAR-VAE Team"

_LAZY_EXPORTS = {
    "AdaLN": ("vyvotts.modules", "AdaLN"),
    "AdaLNZero": ("vyvotts.modules", "AdaLNZero"),
    "DACVAE": ("vyvotts.dacvae", "DACVAE"),
    "DACVAE_BACKENDS": ("vyvotts.dacvae", "DACVAE_BACKENDS"),
    "HubDACVAESource": ("vyvotts.dacvae", "HubDACVAESource"),
    "EchoDiT": ("vyvotts.models", "EchoDiT"),
    "EchoDurationPredictor": ("vyvotts.models", "EchoDurationPredictor"),
    "FlowMatchingDataCollator": ("vyvotts.dataset.data_collator", "FlowMatchingDataCollator"),
    "FlowMatchingEchoDiT": ("vyvotts.models", "FlowMatchingEchoDiT"),
    "FlowMatchingLoss": ("vyvotts.losses", "FlowMatchingLoss"),
    "FlowMatchingTTSInference": ("vyvotts.inference", "FlowMatchingTTSInference"),
    "FlowGRPOConfig": ("vyvotts.post_training", "FlowGRPOConfig"),
    "FlowGRPOTrainer": ("vyvotts.post_training", "FlowGRPOTrainer"),
    "GRPOStageConfig": ("vyvotts.post_training", "GRPOStageConfig"),
    "DEFAULT_GRPO_CONFIG_PATH": ("vyvotts.post_training", "DEFAULT_GRPO_CONFIG_PATH"),
    "GenerationConfig": ("vyvotts.configuration", "GenerationConfig"),
    "CheckpointProvenance": ("vyvotts.checkpoint", "CheckpointProvenance"),
    "CrossLingualUnsupportedError": (
        "vyvotts.languages",
        "CrossLingualUnsupportedError",
    ),
    "Language": ("vyvotts.languages", "Language"),
    "LanguagePair": ("vyvotts.languages", "LanguagePair"),
    "HubCheckpointSource": ("vyvotts.checkpoint", "HubCheckpointSource"),
    "LearnedDurationUnsupportedError": (
        "vyvotts.inference",
        "LearnedDurationUnsupportedError",
    ),
    "MultilingualUnsupportedError": (
        "vyvotts.languages",
        "MultilingualUnsupportedError",
    ),
    "ModelPreset": ("vyvotts.model_presets", "ModelPreset"),
    "ODESolver": ("vyvotts.solvers", "ODESolver"),
    "RealtimeTTSInference": ("vyvotts.inference_realtime", "RealtimeTTSInference"),
    "RotaryPositionalEncoding": ("vyvotts.modules", "RotaryPositionalEncoding"),
    "SimpleTTSCollator": ("vyvotts.dataset.data_collator", "SimpleTTSCollator"),
    "TimestepEmbedding": ("vyvotts.modules", "TimestepEmbedding"),
    "VoiceCloningUnsupportedError": (
        "vyvotts.inference",
        "VoiceCloningUnsupportedError",
    ),
    "create_data_collator": ("vyvotts.dataset.data_collator", "create_data_collator"),
    "create_flow_matching_echodit": ("vyvotts.models", "create_flow_matching_echodit"),
    "bind_reward_evaluator_manifest": (
        "vyvotts.post_training",
        "bind_reward_evaluator_manifest",
    ),
    "get_model_preset": ("vyvotts.model_presets", "get_model_preset"),
    "grpo_post_train": ("vyvotts.post_training", "grpo_post_train"),
    "list_model_presets": ("vyvotts.model_presets", "list_model_presets"),
    "normalize_language": ("vyvotts.languages", "normalize_language"),
    "load_dacvae": ("vyvotts.dacvae", "load_dacvae"),
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
