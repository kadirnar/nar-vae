"""DACVAE backend API with the bundled codec imported only when requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .loader import (
    DACVAE_BACKENDS,
    DEFAULT_DACVAE_BACKEND,
    DEFAULT_DACVAE_FILENAME,
    FAST_DACVAE_REVISION,
    FAST_DACVAE_SOURCE_URL,
    DACVAEBackend,
    DACVAESourceDescription,
    HubDACVAESource,
    describe_dacvae_source,
    get_dacvae_class,
    is_fast_dacvae_available,
    load_dacvae,
    normalize_dacvae_source,
    resolve_dacvae_backend,
)

__version__ = "1.0.0"
__model_version__ = "latest"

__all__ = [
    "DACVAE_BACKENDS",
    "DEFAULT_DACVAE_BACKEND",
    "DEFAULT_DACVAE_FILENAME",
    "DACVAE",
    "DACVAEBackend",
    "DACVAESourceDescription",
    "FAST_DACVAE_REVISION",
    "FAST_DACVAE_SOURCE_URL",
    "HubDACVAESource",
    "describe_dacvae_source",
    "get_dacvae_class",
    "is_fast_dacvae_available",
    "load_dacvae",
    "model",
    "nn",
    "normalize_dacvae_source",
    "resolve_dacvae_backend",
]


def __getattr__(name: str) -> Any:
    """Load the audiotools-backed codec implementation only when it is used."""
    if name == "DACVAE":
        value = getattr(import_module("nar_vae.dacvae.model"), name)
    elif name in {"model", "nn"}:
        value = import_module(f"nar_vae.dacvae.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
