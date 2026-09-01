"""Immutable content identities for prepared NAR-VAE training datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PREPARED_DATASET_MANIFEST_FILENAME = "nar_vae_dataset_manifest.json"
PREPARED_DATASET_MANIFEST_SCHEMA_VERSION = 1
TRAINING_DATASET_IDENTITY_SCHEMA_VERSION = 1
DATASET_IDENTITY_LIBRARY = "nar-vae"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_HUB_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_IDENTITY_FIELDS = {
    "schema_version",
    "library",
    "kind",
    "source",
    "revision",
    "split",
    "fingerprint",
    "num_rows",
    "columns",
    "content_sha256",
}


class DatasetIdentityError(ValueError):
    """Raised when prepared data does not match its immutable identity."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_metadata(dataset: Any) -> tuple[int, list[str], str | None]:
    try:
        num_rows = len(dataset)
    except (TypeError, AttributeError) as exc:
        raise DatasetIdentityError("A training dataset must expose a finite row count.") from exc
    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 0:
        raise DatasetIdentityError("A training dataset must expose a non-negative row count.")
    raw_columns = getattr(dataset, "column_names", None)
    if not isinstance(raw_columns, (list, tuple)) or not all(
        isinstance(column, str) and column for column in raw_columns
    ):
        raise DatasetIdentityError("A training dataset must expose non-empty string column names.")
    columns = list(raw_columns)
    if len(columns) != len(set(columns)):
        raise DatasetIdentityError("Training dataset column names must be unique.")
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint.strip()):
        raise DatasetIdentityError("Dataset fingerprint must be a non-empty string or null.")
    return num_rows, columns, fingerprint


def _relative_artifact(path: Path, root: Path) -> str:
    if path.is_symlink():
        raise DatasetIdentityError(f"Prepared dataset artifacts cannot be symlinks: {path}.")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise DatasetIdentityError(f"Prepared dataset artifact is outside {root}: {path}.") from exc
    if path.resolve().parent != root and root not in path.resolve().parents:
        raise DatasetIdentityError(f"Prepared dataset artifact resolves outside {root}: {path}.")
    return relative.as_posix()


def _prepared_artifacts(directory: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    manifest_path = directory / PREPARED_DATASET_MANIFEST_FILENAME
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DatasetIdentityError(f"Prepared dataset artifacts cannot be symlinks: {path}.")
        if not path.is_file() or path == manifest_path:
            continue
        relative = _relative_artifact(path, directory)
        artifacts[relative] = path
    if not artifacts:
        raise DatasetIdentityError(f"Prepared dataset has no saved artifacts: {directory}.")
    return artifacts


def _persisted_dataset_fingerprint(directory: Path) -> str | None:
    """Read the content fingerprint written by ``Dataset.save_to_disk``.

    ``load_from_disk`` derives a new runtime fingerprint, so that value cannot
    be compared directly with the source Dataset that produced the artifacts.
    The saved state is part of the hashed inventory and is the stable local
    fingerprint.
    """
    state_path = directory / "state.json"
    if not state_path.is_file() or state_path.is_symlink():
        raise DatasetIdentityError(
            f"Prepared dataset is missing a regular state.json: {directory}."
        )
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIdentityError(f"Could not read prepared dataset state: {state_path}.") from exc
    if not isinstance(state, Mapping):
        raise DatasetIdentityError("Prepared dataset state.json must be a JSON object.")
    fingerprint = state.get("_fingerprint")
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint.strip()):
        raise DatasetIdentityError("Prepared dataset state has an invalid fingerprint.")
    return fingerprint


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_prepared_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetIdentityError("Prepared dataset manifest must be a JSON object.")
    manifest = dict(value)
    expected = {
        "schema_version",
        "library",
        "format",
        "num_rows",
        "columns",
        "fingerprint",
        "artifact_sha256",
    }
    if set(manifest) != expected:
        raise DatasetIdentityError(
            "Prepared dataset manifest has incomplete or unknown root fields."
        )
    if manifest["schema_version"] != PREPARED_DATASET_MANIFEST_SCHEMA_VERSION:
        raise DatasetIdentityError("Unsupported prepared-dataset manifest schema.")
    if manifest["library"] != DATASET_IDENTITY_LIBRARY:
        raise DatasetIdentityError("Prepared dataset manifest was not produced by NAR-VAE.")
    if manifest["format"] != "huggingface-save-to-disk":
        raise DatasetIdentityError("Prepared dataset manifest has an unsupported format.")
    num_rows = manifest["num_rows"]
    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 0:
        raise DatasetIdentityError("Prepared dataset manifest has an invalid num_rows.")
    columns = manifest["columns"]
    if not isinstance(columns, list) or not all(
        isinstance(column, str) and column for column in columns
    ):
        raise DatasetIdentityError("Prepared dataset manifest has invalid columns.")
    if len(columns) != len(set(columns)):
        raise DatasetIdentityError("Prepared dataset manifest columns must be unique.")
    fingerprint = manifest["fingerprint"]
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint.strip()):
        raise DatasetIdentityError("Prepared dataset manifest has an invalid fingerprint.")
    artifact_hashes = manifest["artifact_sha256"]
    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise DatasetIdentityError("Prepared dataset manifest has no artifact inventory.")
    normalized_hashes: dict[str, str] = {}
    for relative, checksum in artifact_hashes.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise DatasetIdentityError("Prepared dataset manifest has an unsafe artifact path.")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise DatasetIdentityError("Prepared dataset manifest has an invalid artifact SHA-256.")
        normalized_relative = Path(relative).as_posix()
        if normalized_relative in normalized_hashes:
            raise DatasetIdentityError("Prepared dataset manifest repeats an artifact path.")
        normalized_hashes[normalized_relative] = checksum
    manifest["artifact_sha256"] = normalized_hashes
    return manifest


def write_prepared_dataset_manifest(
    dataset: Any,
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash the saved Arrow/JSON artifact inventory beside a prepared dataset.

    ``output_dir`` must be the output of ``Dataset.save_to_disk``. Raw source
    audio is deliberately outside this boundary and is never recursively read.
    """
    directory = Path(output_dir).expanduser().resolve()
    if not directory.is_dir():
        raise DatasetIdentityError(f"Prepared dataset directory is missing: {directory}.")
    num_rows, columns, _ = _dataset_metadata(dataset)
    artifacts = _prepared_artifacts(directory)
    fingerprint = _persisted_dataset_fingerprint(directory)
    manifest = {
        "schema_version": PREPARED_DATASET_MANIFEST_SCHEMA_VERSION,
        "library": DATASET_IDENTITY_LIBRARY,
        "format": "huggingface-save-to-disk",
        "num_rows": num_rows,
        "columns": columns,
        "fingerprint": fingerprint,
        "artifact_sha256": {relative: _file_sha256(path) for relative, path in artifacts.items()},
    }
    manifest = _validate_prepared_manifest(manifest)
    _atomic_write_json(directory / PREPARED_DATASET_MANIFEST_FILENAME, manifest)
    return manifest


def _load_prepared_manifest(directory: Path) -> dict[str, Any]:
    path = directory / PREPARED_DATASET_MANIFEST_FILENAME
    if not path.is_file():
        raise DatasetIdentityError(
            "Local training requires an immutable prepared-dataset manifest; missing: "
            f"{path}. Re-run dataset preparation with this NAR-VAE version."
        )
    if path.is_symlink():
        raise DatasetIdentityError(f"Prepared dataset manifest cannot be a symlink: {path}.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIdentityError(f"Could not read prepared dataset manifest: {path}.") from exc
    return _validate_prepared_manifest(value)


def _identity(
    *,
    kind: str,
    source: str,
    revision: str | None,
    split: str | None,
    fingerprint: str | None,
    num_rows: int,
    columns: list[str],
    content: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": TRAINING_DATASET_IDENTITY_SCHEMA_VERSION,
        "library": DATASET_IDENTITY_LIBRARY,
        "kind": kind,
        "source": source,
        "revision": revision,
        "split": split,
        "fingerprint": fingerprint,
        "num_rows": num_rows,
        "columns": columns,
        "content_sha256": _canonical_sha256(content),
    }
    return validate_training_dataset_identity(identity)


def resolve_local_prepared_dataset_identity(
    dataset: Any,
    dataset_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate every prepared artifact and return its compact training identity."""
    directory = Path(dataset_dir).expanduser().resolve()
    manifest = _load_prepared_manifest(directory)
    artifacts = _prepared_artifacts(directory)
    expected_hashes = manifest["artifact_sha256"]
    if set(artifacts) != set(expected_hashes):
        raise DatasetIdentityError("Prepared dataset artifact inventory has changed.")
    for relative, path in artifacts.items():
        if _file_sha256(path) != expected_hashes[relative]:
            raise DatasetIdentityError(
                f"Prepared dataset artifact {relative} does not match its SHA-256."
            )
    num_rows, columns, _ = _dataset_metadata(dataset)
    if num_rows != manifest["num_rows"]:
        raise DatasetIdentityError("Prepared dataset row count does not match its manifest.")
    if columns != manifest["columns"]:
        raise DatasetIdentityError("Prepared dataset columns do not match its manifest.")
    fingerprint = _persisted_dataset_fingerprint(directory)
    if fingerprint != manifest["fingerprint"]:
        raise DatasetIdentityError(
            "Prepared dataset state fingerprint does not match its manifest."
        )
    return _identity(
        kind="local-prepared",
        source=str(directory),
        revision=None,
        split=None,
        fingerprint=fingerprint,
        num_rows=num_rows,
        columns=columns,
        content=manifest,
    )


def resolve_hub_dataset_identity(
    dataset: Any,
    *,
    repo_id: str,
    revision: str,
    split: str,
    snapshot_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Bind a commit-contained prepared Hub snapshot by its exact artifact bytes."""
    if not isinstance(repo_id, str) or repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise DatasetIdentityError("Hub dataset repo_id must use 'owner/name' format.")
    if not isinstance(revision, str) or not _HUB_COMMIT.fullmatch(revision):
        raise DatasetIdentityError("Hub dataset revision must be a full 40-character commit SHA.")
    if not isinstance(split, str) or not split.strip():
        raise DatasetIdentityError("Hub dataset split must be non-empty.")
    prepared_identity = resolve_local_prepared_dataset_identity(dataset, snapshot_dir)
    num_rows = prepared_identity["num_rows"]
    columns = prepared_identity["columns"]
    fingerprint = prepared_identity["fingerprint"]
    core = {
        "repo_id": repo_id,
        "revision": revision.lower(),
        "split": split,
        "fingerprint": fingerprint,
        "num_rows": num_rows,
        "columns": columns,
        "prepared_content_sha256": prepared_identity["content_sha256"],
    }
    return _identity(
        kind="hub",
        source=repo_id,
        revision=revision.lower(),
        split=split,
        fingerprint=fingerprint,
        num_rows=num_rows,
        columns=columns,
        content=core,
    )


def validate_training_dataset_identity(value: Any) -> dict[str, Any]:
    """Schema-validate one compact identity before hashing it into a run."""
    if not isinstance(value, Mapping):
        raise DatasetIdentityError("Training dataset identity must be an object.")
    identity = dict(value)
    if set(identity) != _IDENTITY_FIELDS:
        raise DatasetIdentityError("Training dataset identity has incomplete or unknown fields.")
    if identity["schema_version"] != TRAINING_DATASET_IDENTITY_SCHEMA_VERSION:
        raise DatasetIdentityError("Unsupported training-dataset identity schema.")
    if identity["library"] != DATASET_IDENTITY_LIBRARY:
        raise DatasetIdentityError("Training dataset identity was not produced by NAR-VAE.")
    if identity["kind"] not in {"local-prepared", "hub"}:
        raise DatasetIdentityError("Training dataset identity has an unsupported kind.")
    if not isinstance(identity["source"], str) or not identity["source"].strip():
        raise DatasetIdentityError("Training dataset identity source must be non-empty.")
    if identity["kind"] == "hub":
        if not isinstance(identity["revision"], str) or not _HUB_COMMIT.fullmatch(
            identity["revision"]
        ):
            raise DatasetIdentityError("Hub training identity requires a full commit revision.")
        if not isinstance(identity["split"], str) or not identity["split"].strip():
            raise DatasetIdentityError("Hub training identity requires a split.")
    elif identity["revision"] is not None or identity["split"] is not None:
        raise DatasetIdentityError("Local prepared-data identity cannot declare revision or split.")
    fingerprint = identity["fingerprint"]
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint.strip()):
        raise DatasetIdentityError("Training dataset fingerprint must be non-empty or null.")
    num_rows = identity["num_rows"]
    if isinstance(num_rows, bool) or not isinstance(num_rows, int) or num_rows < 0:
        raise DatasetIdentityError("Training dataset identity has an invalid num_rows.")
    columns = identity["columns"]
    if not isinstance(columns, list) or not all(
        isinstance(column, str) and column for column in columns
    ):
        raise DatasetIdentityError("Training dataset identity has invalid columns.")
    if len(columns) != len(set(columns)):
        raise DatasetIdentityError("Training dataset identity columns must be unique.")
    if not isinstance(identity["content_sha256"], str) or not _SHA256.fullmatch(
        identity["content_sha256"]
    ):
        raise DatasetIdentityError("Training dataset identity has an invalid content SHA-256.")
    return identity


def training_dataset_identity_sha256(value: Mapping[str, Any]) -> str:
    """Return the canonical digest stored in the immutable run manifest."""
    return _canonical_sha256(validate_training_dataset_identity(value))


__all__ = [
    "DATASET_IDENTITY_LIBRARY",
    "PREPARED_DATASET_MANIFEST_FILENAME",
    "PREPARED_DATASET_MANIFEST_SCHEMA_VERSION",
    "TRAINING_DATASET_IDENTITY_SCHEMA_VERSION",
    "DatasetIdentityError",
    "resolve_hub_dataset_identity",
    "resolve_local_prepared_dataset_identity",
    "training_dataset_identity_sha256",
    "validate_training_dataset_identity",
    "write_prepared_dataset_manifest",
]
