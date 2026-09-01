"""Validated EchoDiT architecture families for training and inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

ARCHITECTURE_FIELDS = (
    "model_size",
    "num_layers",
    "num_heads",
    "intermediate_size",
    "text_model_size",
    "text_num_layers",
    "text_num_heads",
    "text_intermediate_size",
    "speaker_model_size",
    "speaker_num_layers",
    "speaker_num_heads",
    "speaker_intermediate_size",
    "timestep_embed_size",
    "adaln_rank",
)


@dataclass(frozen=True)
class ModelPreset:
    """One internally consistent EchoDiT/text/speaker architecture."""

    name: str
    description: str
    model_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    text_model_size: int
    text_num_layers: int
    text_num_heads: int
    text_intermediate_size: int
    speaker_model_size: int
    speaker_num_layers: int
    speaker_num_heads: int
    speaker_intermediate_size: int
    timestep_embed_size: int
    adaln_rank: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Model preset name must not be empty.")
        for field_name in ARCHITECTURE_FIELDS:
            minimum = 0 if field_name == "text_num_layers" else 1
            if getattr(self, field_name) < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{field_name} must be {qualifier} in preset {self.name!r}.")
        for prefix in ("", "text_", "speaker_"):
            width = getattr(self, f"{prefix}model_size")
            heads = getattr(self, f"{prefix}num_heads")
            if width % heads:
                raise ValueError(
                    f"{prefix}model_size must be divisible by {prefix}num_heads "
                    f"in preset {self.name!r}."
                )
            if (width // heads) % 2:
                raise ValueError(
                    f"{prefix} attention head width must be even for rotary embeddings "
                    f"in preset {self.name!r}."
                )
        if self.adaln_rank > self.model_size:
            raise ValueError(f"adaln_rank cannot exceed model_size in preset {self.name!r}.")

    def model_kwargs(self) -> dict[str, int]:
        """Return constructor arguments shared by scratch, fine-tune, and inference paths."""
        return {field_name: getattr(self, field_name) for field_name in ARCHITECTURE_FIELDS}


@lru_cache(maxsize=1)
def load_model_presets() -> Mapping[str, ModelPreset]:
    """Load every packaged model-size preset and validate its dimensions."""
    resource = resources.files("nar_vae.configs").joinpath("model_presets.toml")
    with resource.open("rb") as handle:
        raw = tomllib.load(handle)
    presets = {name: ModelPreset(name=name, **values) for name, values in raw["presets"].items()}
    if not presets:
        raise ValueError("At least one model preset must be packaged.")
    return MappingProxyType(presets)


def list_model_presets() -> tuple[str, ...]:
    """Return stable model preset names from smallest to largest."""
    return tuple(load_model_presets())


def get_model_preset(name: str) -> ModelPreset:
    """Return a validated model preset with an actionable error for typos."""
    try:
        return load_model_presets()[name]
    except KeyError as exc:
        choices = ", ".join(list_model_presets())
        raise ValueError(f"Unknown model preset {name!r}. Expected one of: {choices}.") from exc


def resolve_model_architecture(config: Mapping[str, Any]) -> ModelPreset:
    """Resolve a named preset or validate a fully explicit custom architecture.

    Explicit architecture fields alongside ``model_preset`` must match it. This prevents a run
    configuration from claiming one preset name while silently training different tensor shapes.
    """
    preset_name = config.get("model_preset")
    if preset_name is not None:
        if not isinstance(preset_name, str):
            raise ValueError("model_preset must be a string.")
        preset = get_model_preset(preset_name)
        frozen_features = config.get("text_conditioning_mode") == "frozen_features"
        conflicts = {
            field_name: (config[field_name], getattr(preset, field_name))
            for field_name in ARCHITECTURE_FIELDS
            if field_name in config
            and config[field_name] != getattr(preset, field_name)
            and not (
                frozen_features and field_name == "text_num_layers" and config[field_name] == 0
            )
        }
        if conflicts:
            details = ", ".join(
                f"{name}={actual!r} (preset {expected!r})"
                for name, (actual, expected) in conflicts.items()
            )
            raise ValueError(
                f"Configuration conflicts with model_preset {preset_name!r}: {details}."
            )
        if frozen_features:
            # Frozen contextual states replace the scratch text Transformer.
            # Retain the preset's adapter width while making the absent layers
            # explicit in parameter counts, manifests, and checkpoint topology.
            return ModelPreset(
                **{
                    **preset.__dict__,
                    "description": f"{preset.description} Frozen text-feature adapter.",
                    "text_num_layers": 0,
                }
            )
        return preset

    missing = [field_name for field_name in ARCHITECTURE_FIELDS if field_name not in config]
    if missing:
        raise ValueError(
            "A custom architecture must define every model field. "
            f"Missing: {missing}. Alternatively set model_preset."
        )
    return ModelPreset(
        name="custom",
        description="Explicit custom architecture",
        **{field_name: config[field_name] for field_name in ARCHITECTURE_FIELDS},
    )


__all__ = [
    "ARCHITECTURE_FIELDS",
    "ModelPreset",
    "get_model_preset",
    "list_model_presets",
    "load_model_presets",
    "resolve_model_architecture",
]
