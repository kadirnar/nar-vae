"""Canonical NAR-VAE public API.

The implementation currently remains in :mod:`vyvotts` so existing checkpoints and imports keep
working during the package transition. New code should import :mod:`nar_vae`.
"""

from __future__ import annotations

import sys
from importlib import import_module, util
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

import vyvotts as _implementation

__author__ = "NAR-VAE Team"
__version__ = _implementation.__version__
__all__ = _implementation.__all__


class _ImplementationAliasLoader(Loader):
    """Return one existing implementation module for its canonical public name."""

    def __init__(self, public_name: str, implementation_name: str) -> None:
        self.public_name = public_name
        self.implementation_name = implementation_name
        self._implementation_metadata: dict[str, object] = {}

    def create_module(self, spec: ModuleSpec):
        del spec
        module = import_module(self.implementation_name)
        for name in (
            "__name__",
            "__spec__",
            "__loader__",
            "__package__",
            "__file__",
            "__cached__",
            "__path__",
        ):
            if hasattr(module, name):
                self._implementation_metadata[name] = getattr(module, name)
        return module

    def exec_module(self, module) -> None:
        # Import machinery initializes the returned object with the alias spec.
        # Restore the implementation metadata so reload, resources, pickling,
        # and introspection continue to see one internally consistent module.
        for name, value in self._implementation_metadata.items():
            setattr(module, name, value)
        sys.modules[self.public_name] = module


class _ImplementationAliasFinder(MetaPathFinder):
    """Lazily map ``nar_vae.*`` imports to the single ``vyvotts.*`` module graph."""

    nar_vae_alias_finder = True

    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        prefix = f"{__name__}."
        if not fullname.startswith(prefix):
            return None
        implementation_name = f"vyvotts.{fullname.removeprefix(prefix)}"
        try:
            implementation_spec = util.find_spec(implementation_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            return None
        if implementation_spec is None:
            return None
        loader = _ImplementationAliasLoader(fullname, implementation_name)
        return util.spec_from_loader(
            fullname,
            loader,
            origin=implementation_spec.origin,
            is_package=implementation_spec.submodule_search_locations is not None,
        )


# A copied ``__path__`` makes Python execute each implementation file a second
# time as ``nar_vae.<module>``. The finder instead aliases submodules lazily,
# preserving optional dependencies while sharing classes, locks, and caches.
if not any(getattr(finder, "nar_vae_alias_finder", False) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ImplementationAliasFinder())


def __getattr__(name: str):
    """Resolve public objects lazily through the compatibility implementation."""
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(__all__)
