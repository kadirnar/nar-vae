"""Versioned representation metadata for newly prepared acoustic rows."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from nar_vae.dacvae.loader import HubDACVAESource, describe_dacvae_source
from nar_vae.dacvae_encoding import DACVAE_POSTERIOR_SAMPLING_POLICY
from nar_vae.tokenization import (
    TEXT_FRONTEND_NAME as ACTIVE_TEXT_FRONTEND_NAME,
)
from nar_vae.tokenization import (
    TEXT_FRONTEND_VERSION as ACTIVE_TEXT_FRONTEND_VERSION,
)

REPRESENTATION_CONTRACT_COLUMN = "representation_contract"
REPRESENTATION_CONTRACT_VERSION = 3
PREPARED_ROW_VERSION_COLUMN = "prepared_row_version"
PREPARED_ROW_VERSION = 2
TEXT_FRONTEND_NAME = ACTIVE_TEXT_FRONTEND_NAME
TEXT_FRONTEND_VERSION = ACTIVE_TEXT_FRONTEND_VERSION
LEGACY_TEXT_FRONTENDS = frozenset({("nar_vae.encode_tts_text/cl100k_base", 1)})


class RepresentationContractError(ValueError):
    """Raised when prepared data does not match its declared representation."""


@dataclass(frozen=True)
class RepresentationContract:
    """Stable metadata required to interpret a prepared acoustic row."""

    contract_version: int
    text_frontend_name: str
    text_frontend_version: int
    codec_source: str
    codec_backend: str
    codec_revision: str | None
    codec_filename: str | None
    codec_sha256: str
    codec_encoding_policy: str
    sample_rate: int
    hop_length: int
    latent_width: int

    def to_dict(self) -> dict[str, int | str | None]:
        """Return an independent, Arrow-friendly row value."""
        return asdict(self)


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise RepresentationContractError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RepresentationContractError(f"{name} must be a positive integer.") from exc
    if result <= 0 or result != value:
        raise RepresentationContractError(f"{name} must be a positive integer.")
    return result


def _codec_latent_width(codec: Any) -> int:
    try:
        width = codec.quantizer.out_proj.in_channels
    except AttributeError as exc:
        raise RepresentationContractError(
            "The loaded codec does not expose quantizer.out_proj.in_channels."
        ) from exc
    return _positive_integer(width, name="codec latent_width")


def build_representation_contract(
    codec: Any,
    *,
    codec_source: str | os.PathLike[str] | HubDACVAESource,
    text_frontend_name: str = TEXT_FRONTEND_NAME,
    text_frontend_version: int = TEXT_FRONTEND_VERSION,
) -> RepresentationContract:
    """Build a contract from the codec instance actually used for preparation."""
    source = describe_dacvae_source(codec_source)
    if not source.identifier.strip():
        raise RepresentationContractError("codec_source must be non-empty.")
    if (
        not isinstance(text_frontend_name, str)
        or not text_frontend_name.strip()
        or text_frontend_name != text_frontend_name.strip()
    ):
        raise RepresentationContractError("text_frontend_name must be a normalized string.")
    text_frontend_version = _positive_integer(
        text_frontend_version,
        name="text_frontend_version",
    )

    backend = getattr(codec, "nar_vae_backend", None)
    if not isinstance(backend, str) or not backend.strip():
        raise RepresentationContractError("The loaded codec does not expose its resolved backend.")
    resolved_source = (
        getattr(codec, "nar_vae_codec_identifier", None),
        getattr(codec, "nar_vae_codec_revision", None),
        getattr(codec, "nar_vae_codec_filename", None),
    )
    declared_source = (source.identifier, source.revision, source.filename)
    if resolved_source != declared_source:
        raise RepresentationContractError(
            "The declared codec source does not match the artifact actually loaded."
        )
    artifact_sha256 = getattr(codec, "nar_vae_codec_sha256", None)
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise RepresentationContractError(
            "The loaded codec does not expose a lowercase artifact SHA-256."
        )

    return RepresentationContract(
        contract_version=REPRESENTATION_CONTRACT_VERSION,
        text_frontend_name=text_frontend_name,
        text_frontend_version=text_frontend_version,
        codec_source=source.identifier,
        codec_backend=backend.strip(),
        codec_revision=source.revision,
        codec_filename=source.filename,
        codec_sha256=artifact_sha256,
        codec_encoding_policy=DACVAE_POSTERIOR_SAMPLING_POLICY,
        sample_rate=_positive_integer(getattr(codec, "sample_rate", None), name="sample_rate"),
        hop_length=_positive_integer(getattr(codec, "hop_length", None), name="hop_length"),
        latent_width=_codec_latent_width(codec),
    )


def validate_latent_representation(
    latents: Any,
    contract: RepresentationContract,
    *,
    field_name: str = "latents",
) -> None:
    """Validate a generated ``[latent_width, frames]`` array against its contract."""
    shape = getattr(latents, "shape", None)
    if shape is None:
        raise RepresentationContractError(f"{field_name} must expose a two-dimensional shape.")
    if len(shape) != 2:
        raise RepresentationContractError(
            f"{field_name} must have shape [latent_width, frames], got {tuple(shape)}."
        )
    width, frames = shape
    if int(width) != contract.latent_width:
        raise RepresentationContractError(
            f"{field_name} latent width {int(width)} does not match the representation contract "
            f"({contract.latent_width})."
        )
    if int(frames) <= 0:
        raise RepresentationContractError(f"{field_name} must contain at least one frame.")


def attach_representation_contract(
    row: dict[str, Any],
    contract: RepresentationContract,
) -> dict[str, Any]:
    """Validate acoustic tensors and attach a fresh per-row contract mapping."""
    if "latents" not in row:
        raise RepresentationContractError("A prepared row must contain latents.")
    validate_latent_representation(row["latents"], contract)
    if "speaker_latents" in row:
        validate_latent_representation(
            row["speaker_latents"],
            contract,
            field_name="speaker_latents",
        )
    row[PREPARED_ROW_VERSION_COLUMN] = PREPARED_ROW_VERSION
    row[REPRESENTATION_CONTRACT_COLUMN] = contract.to_dict()
    return row


def is_supported_text_frontend(
    name: Any,
    version: Any,
    *,
    allow_legacy: bool = False,
) -> bool:
    """Return whether a row frontend can be interpreted by the selected loader.

    Legacy cl100k rows remain identifiable but are never silently mixed with
    v2 hybrid rows. Callers must opt in and synthesize their missing parallel
    masks explicitly.
    """
    identity = (name, version)
    if identity == (TEXT_FRONTEND_NAME, TEXT_FRONTEND_VERSION):
        return True
    return allow_legacy and identity in LEGACY_TEXT_FRONTENDS


__all__ = [
    "REPRESENTATION_CONTRACT_COLUMN",
    "REPRESENTATION_CONTRACT_VERSION",
    "PREPARED_ROW_VERSION",
    "PREPARED_ROW_VERSION_COLUMN",
    "TEXT_FRONTEND_NAME",
    "TEXT_FRONTEND_VERSION",
    "LEGACY_TEXT_FRONTENDS",
    "RepresentationContract",
    "RepresentationContractError",
    "attach_representation_contract",
    "build_representation_contract",
    "is_supported_text_frontend",
    "validate_latent_representation",
]
