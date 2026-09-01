"""Backend selection and loading helpers for DACVAE models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module, util
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Literal

DACVAEBackend = Literal["auto", "bundled", "fast"]
DACVAE_BACKENDS = ("auto", "bundled", "fast")
DEFAULT_DACVAE_BACKEND: DACVAEBackend = "bundled"
DEFAULT_DACVAE_FILENAME = "weights.pth"
_HUB_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

FAST_DACVAE_SOURCE_URL = "https://github.com/kadirnar/fast-dacvae.git"
FAST_DACVAE_REVISION = "406f2e5c803927ef18cc9bbe38d715e5417459b9"

_FAST_DACVAE_INSTALL_HINT = (
    "Install the pinned optional backend directly with "
    "`pip install 'fast-dacvae @ "
    f"git+{FAST_DACVAE_SOURCE_URL}@{FAST_DACVAE_REVISION}'`."
)
_FAST_DACVAE_MODEL_KWARGS = {
    "codebook_dim",
    "codebook_size",
    "decoder_dim",
    "decoder_rates",
    "encoder_dim",
    "encoder_rates",
    "latent_dim",
    "n_codebooks",
    "quantizer_dropout",
    "sample_rate",
}


@dataclass(frozen=True)
class HubDACVAESource:
    """Explicit immutable Hub source for one DACVAE weight artifact.

    User-facing APIs also accept a plain Hugging Face repository ID together
    with ``dacvae_revision``. This type is the normalized representation used
    internally and remains available for advanced callers.
    """

    repo_id: str
    revision: str
    filename: str = DEFAULT_DACVAE_FILENAME

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repo_id, str)
            or self.repo_id != self.repo_id.strip()
            or self.repo_id.count("/") != 1
            or not all(self.repo_id.split("/"))
        ):
            raise ValueError("repo_id must use the non-empty 'owner/name' Hub format.")
        if not isinstance(self.revision, str) or not _HUB_COMMIT_PATTERN.fullmatch(self.revision):
            raise ValueError("revision must be an explicit 40-character Hub commit hash.")
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be an explicit non-empty repository-relative path.")
        path = Path(self.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("filename must be an explicit repository-relative path.")


@dataclass(frozen=True)
class DACVAESourceDescription:
    """Stable source identity stored in model and dataset contracts."""

    identifier: str
    revision: str | None
    filename: str | None


def _looks_like_hub_repo_id(source: str) -> bool:
    """Return whether a plain string uses Hugging Face's ``owner/name`` shape."""
    return (
        source == source.strip()
        and not source.startswith((".", "~", os.sep))
        and source.count("/") == 1
        and all(source.split("/"))
    )


def normalize_dacvae_source(
    source: str | os.PathLike[str] | HubDACVAESource,
    *,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
) -> str | os.PathLike[str] | HubDACVAESource:
    """Normalize a local source or a commit-pinned Hugging Face repository ID.

    A plain ``owner/name`` string is a Hub ID. It must be paired with a full
    commit hash so training data and inference never depend on a mutable branch.
    Use a :class:`~pathlib.Path` or an explicit ``./`` prefix for a local path
    that happens to have the same two-component shape.
    """
    if isinstance(source, HubDACVAESource):
        if dacvae_revision is not None or dacvae_filename is not None:
            raise ValueError(
                "Do not combine HubDACVAESource with dacvae_revision or dacvae_filename."
            )
        return source

    if isinstance(source, str) and _looks_like_hub_repo_id(source):
        if dacvae_revision is None:
            raise ValueError(
                "A Hugging Face-shaped dacvae_model cannot be used as an unpinned local "
                "source; set dacvae_revision to a full 40-character commit hash."
            )
        return HubDACVAESource(
            repo_id=source,
            revision=dacvae_revision,
            filename=(DEFAULT_DACVAE_FILENAME if dacvae_filename is None else dacvae_filename),
        )

    if dacvae_revision is not None or dacvae_filename is not None:
        raise ValueError(
            "dacvae_revision and dacvae_filename are only valid with a plain "
            "'owner/name' Hugging Face ID."
        )
    return source


def describe_dacvae_source(
    source: str | os.PathLike[str] | HubDACVAESource,
) -> DACVAESourceDescription:
    """Describe a codec source without accessing the filesystem or network."""
    if isinstance(source, HubDACVAESource):
        return DACVAESourceDescription(source.repo_id, source.revision.lower(), source.filename)
    return DACVAESourceDescription(os.fspath(source), None, None)


def _commit_from_hub_cache_path(path: Path) -> str | None:
    for candidate in (path, path.resolve()):
        parts = candidate.parts
        for index, part in enumerate(parts[:-1]):
            if part == "snapshots" and _HUB_COMMIT_PATTERN.fullmatch(parts[index + 1]):
                return parts[index + 1]
    return None


def _resolve_dacvae_artifact(
    source: str | os.PathLike[str] | HubDACVAESource,
) -> tuple[Path, DACVAESourceDescription]:
    """Resolve one local artifact or explicitly pinned Hub artifact."""
    description = describe_dacvae_source(source)
    if isinstance(source, HubDACVAESource):
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=source.repo_id,
                filename=source.filename,
                revision=source.revision,
            )
        )
        resolved_commit = _commit_from_hub_cache_path(path)
        if resolved_commit is not None and resolved_commit.lower() != source.revision.lower():
            raise RuntimeError(
                "Hugging Face Hub resolved a different DACVAE commit than requested: "
                f"{resolved_commit!r} != {source.revision!r}."
            )
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded DACVAE artifact is missing: {path}.")
        return path, description

    path = Path(os.fspath(source)).expanduser()
    if path.is_dir():
        path = path / "weights.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"DACVAE checkpoint not found: {path}. Use a plain 'owner/name' Hugging Face ID "
            "with dacvae_revision for a remote codec."
        )
    return path, description


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_fast_dacvae_distribution() -> Any | None:
    try:
        return distribution("fast-dacvae")
    except PackageNotFoundError:
        return None


def _validate_fast_dacvae_provenance(installed_distribution: Any) -> None:
    """Require the reviewed immutable VCS source recorded by PEP 610."""
    try:
        direct_url_text = installed_distribution.read_text("direct_url.json")
    except (FileNotFoundError, OSError) as exc:
        raise ImportError(
            "The fast-dacvae installation has no readable PEP 610 direct_url.json provenance. "
            f"{_FAST_DACVAE_INSTALL_HINT}"
        ) from exc
    if not direct_url_text:
        raise ImportError(
            "The fast-dacvae installation has no PEP 610 direct_url.json provenance. "
            "PyPI, local-directory, and unpinned installs are not accepted. "
            f"{_FAST_DACVAE_INSTALL_HINT}"
        )

    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ImportError(
            "The fast-dacvae PEP 610 direct_url.json provenance is malformed. "
            f"{_FAST_DACVAE_INSTALL_HINT}"
        ) from exc
    if not isinstance(direct_url, Mapping):
        raise ImportError(
            "The fast-dacvae PEP 610 direct_url.json provenance must be an object. "
            f"{_FAST_DACVAE_INSTALL_HINT}"
        )

    source_url = direct_url.get("url")
    vcs_info = direct_url.get("vcs_info")
    vcs = vcs_info.get("vcs") if isinstance(vcs_info, Mapping) else None
    commit_id = vcs_info.get("commit_id") if isinstance(vcs_info, Mapping) else None
    if source_url != FAST_DACVAE_SOURCE_URL:
        raise ImportError(
            "The fast-dacvae installation does not come from the reviewed source URL: "
            f"{source_url!r} != {FAST_DACVAE_SOURCE_URL!r}. {_FAST_DACVAE_INSTALL_HINT}"
        )
    if vcs != "git" or not isinstance(commit_id, str):
        raise ImportError(
            "The fast-dacvae installation does not record immutable Git commit provenance. "
            f"{_FAST_DACVAE_INSTALL_HINT}"
        )
    if commit_id.lower() != FAST_DACVAE_REVISION:
        raise ImportError(
            "The fast-dacvae installation is not the reviewed commit: "
            f"{commit_id!r} != {FAST_DACVAE_REVISION!r}. {_FAST_DACVAE_INSTALL_HINT}"
        )


def _require_fast_dacvae_distribution() -> Any:
    installed_distribution = _installed_fast_dacvae_distribution()
    if installed_distribution is None:
        raise ImportError(
            f"The fast-dacvae distribution is not installed. {_FAST_DACVAE_INSTALL_HINT}"
        )
    if util.find_spec("dacvae") is None:
        raise ImportError(
            "The fast-dacvae distribution is installed but its 'dacvae' module is missing. "
            f"{_FAST_DACVAE_INSTALL_HINT}"
        )
    _validate_fast_dacvae_provenance(installed_distribution)
    return installed_distribution


def is_fast_dacvae_available() -> bool:
    """Return whether the exact reviewed external backend is installed and importable."""
    try:
        _require_fast_dacvae_distribution()
    except ImportError:
        return False
    return True


def resolve_dacvae_backend(
    backend: str = DEFAULT_DACVAE_BACKEND,
) -> Literal["bundled", "fast"]:
    """Resolve a user-facing backend name to a concrete implementation."""
    normalized_backend = backend.strip().lower().replace("_", "-")
    if normalized_backend not in DACVAE_BACKENDS:
        choices = ", ".join(DACVAE_BACKENDS)
        raise ValueError(f"Unknown DACVAE backend {backend!r}. Expected one of: {choices}.")

    if normalized_backend == "auto":
        if _installed_fast_dacvae_distribution() is None:
            return "bundled"
        _require_fast_dacvae_distribution()
        return "fast"
    if normalized_backend == "fast":
        _require_fast_dacvae_distribution()
    return normalized_backend


def get_dacvae_class(backend: str = DEFAULT_DACVAE_BACKEND) -> type[Any]:
    """Return the DACVAE class provided by the selected backend."""
    resolved_backend = resolve_dacvae_backend(backend)
    module_name = "dacvae" if resolved_backend == "fast" else "nar_vae.dacvae.model"

    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if resolved_backend == "fast":
            missing_package = exc.name or "an optional dependency"
            raise ImportError(
                "The fast-dacvae backend could not be imported because "
                f"{missing_package!r} is missing. {_FAST_DACVAE_INSTALL_HINT}"
            ) from exc
        raise

    if resolved_backend == "fast" and not hasattr(module, "optimize_dacvae"):
        raise ImportError(
            "The imported 'dacvae' module is not provided by fast-dacvae. "
            "The official dacvae package uses the same import name; uninstall "
            "the conflicting distribution before selecting backend='fast'."
        )

    dacvae_class = getattr(module, "DACVAE", None)
    if dacvae_class is None:
        raise ImportError(f"The {resolved_backend!r} backend does not export a DACVAE class.")
    return dacvae_class


def _resolve_fast_checkpoint(model_name_or_path: str | os.PathLike[str]) -> Path:
    source = os.fspath(model_name_or_path)
    checkpoint_path = Path(source).expanduser()

    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "weights.pth"
    if checkpoint_path.is_file():
        return checkpoint_path

    raise FileNotFoundError(
        f"DACVAE checkpoint not found: {checkpoint_path}. Remote loading requires "
        "HubDACVAESource at the load_dacvae() boundary."
    )


def _get_dacvae_latent_size(model: Any) -> int:
    """Read the latent width shared by the bundled and fast implementations."""
    try:
        return int(model.quantizer.out_proj.in_channels)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("The loaded DACVAE does not expose its latent width.") from exc


def _load_fast_dacvae(
    dacvae_class: type[Any],
    model_name_or_path: str | os.PathLike[str],
) -> Any:
    """Load fast-dacvae without relying on its unsafe default architecture."""
    import torch

    checkpoint_path = _resolve_fast_checkpoint(model_name_or_path)
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("state_dict"), Mapping):
        raise ValueError("Expected a DACVAE checkpoint containing a 'state_dict' mapping.")

    metadata = artifact.get("metadata")
    model_kwargs = metadata.get("kwargs") if isinstance(metadata, Mapping) else None
    if not isinstance(model_kwargs, Mapping):
        raise ValueError(
            "DACVAE checkpoint metadata.kwargs is missing. Refusing to guess "
            "an architecture for the fast backend."
        )

    supplied_keys = set(model_kwargs)
    missing_keys = _FAST_DACVAE_MODEL_KWARGS - supplied_keys
    unknown_keys = supplied_keys - _FAST_DACVAE_MODEL_KWARGS
    if missing_keys or unknown_keys:
        raise ValueError(
            "Unsupported DACVAE checkpoint metadata: "
            f"missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}."
        )

    normalized_kwargs = dict(model_kwargs)
    normalized_kwargs["encoder_rates"] = list(normalized_kwargs["encoder_rates"])
    normalized_kwargs["decoder_rates"] = list(normalized_kwargs["decoder_rates"])

    model = dacvae_class(**normalized_kwargs)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.metadata = dict(metadata)

    expected_hop_length = math.prod(normalized_kwargs["encoder_rates"])
    if model.hop_length != expected_hop_length:
        raise RuntimeError(
            "Loaded DACVAE has an unexpected hop length: "
            f"{model.hop_length} != {expected_hop_length}."
        )

    latent_width = _get_dacvae_latent_size(model)
    if latent_width != normalized_kwargs["codebook_dim"]:
        raise RuntimeError(
            "Loaded DACVAE has an unexpected latent width: "
            f"{latent_width} != {normalized_kwargs['codebook_dim']}."
        )

    return model


def load_dacvae(
    model_name_or_path: str | os.PathLike[str] | HubDACVAESource,
    *,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
    backend: str = DEFAULT_DACVAE_BACKEND,
    device: Any | None = None,
    freeze: bool = False,
    verbose: bool = True,
    expected_latent_size: int | None = None,
    expected_sha256: str | None = None,
) -> Any:
    """Load and prepare a DACVAE model.

    ``model_name_or_path`` may be a local artifact, an advanced
    :class:`HubDACVAESource`, or a plain Hugging Face ``owner/name`` ID paired
    with a full ``dacvae_revision``. Both codec backends expose the ``load``,
    ``encode`` and ``decode`` methods used by NAR-VAE.
    """
    resolved_backend = resolve_dacvae_backend(backend)
    source = normalize_dacvae_source(
        model_name_or_path,
        dacvae_revision=dacvae_revision,
        dacvae_filename=dacvae_filename,
    )
    checkpoint_path, source_description = _resolve_dacvae_artifact(source)
    artifact_sha256 = _file_sha256(checkpoint_path)
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256.")
        if artifact_sha256 != expected_sha256:
            raise ValueError(
                "DACVAE artifact does not match the checkpoint representation SHA-256."
            )
    if verbose:
        print(f"Loading DACVAE: {source_description.identifier} (backend={resolved_backend})")

    dacvae_class = get_dacvae_class(resolved_backend)
    if resolved_backend == "fast":
        model = _load_fast_dacvae(dacvae_class, checkpoint_path)
    else:
        model = dacvae_class.load(os.fspath(checkpoint_path))

    if expected_latent_size is not None:
        actual_latent_size = _get_dacvae_latent_size(model)
        if actual_latent_size != expected_latent_size:
            raise ValueError(
                "DACVAE latent width does not match the EchoDiT configuration: "
                f"{actual_latent_size} != {expected_latent_size}."
            )

    if device is not None:
        model = model.to(device)
    model.eval()

    if freeze:
        model.requires_grad_(False)

    model.nar_vae_backend = resolved_backend
    model.nar_vae_codec_identifier = source_description.identifier
    model.nar_vae_codec_revision = source_description.revision
    model.nar_vae_codec_filename = source_description.filename
    model.nar_vae_codec_path = os.fspath(checkpoint_path)
    model.nar_vae_codec_sha256 = artifact_sha256
    return model


__all__ = [
    "DACVAE_BACKENDS",
    "DEFAULT_DACVAE_BACKEND",
    "DEFAULT_DACVAE_FILENAME",
    "DACVAEBackend",
    "DACVAESourceDescription",
    "FAST_DACVAE_REVISION",
    "FAST_DACVAE_SOURCE_URL",
    "HubDACVAESource",
    "describe_dacvae_source",
    "get_dacvae_class",
    "is_fast_dacvae_available",
    "load_dacvae",
    "normalize_dacvae_source",
    "resolve_dacvae_backend",
]
