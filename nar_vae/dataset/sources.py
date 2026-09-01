"""Reproducible local and Hugging Face dataset source handling."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_DOWNLOAD_WORKERS = 8
MAX_DATASET_DOWNLOAD_WORKERS = 32
_HUB_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class DatasetSource:
    """A validated dataset source and its bounded loading policy."""

    location: str
    revision: str | None
    download_workers: int
    is_local: bool

    def load_dataset_kwargs(self) -> dict[str, int | str]:
        """Return kwargs safe to pass to ``datasets.load_dataset``."""
        kwargs: dict[str, int | str] = {"num_proc": self.download_workers}
        if not self.is_local:
            if self.revision is None:
                raise RuntimeError("A remote DatasetSource must carry a pinned revision.")
            kwargs["revision"] = self.revision
        return kwargs


def _validate_download_workers(download_workers: int) -> int:
    if (
        isinstance(download_workers, bool)
        or not isinstance(download_workers, int)
        or not 1 <= download_workers <= MAX_DATASET_DOWNLOAD_WORKERS
    ):
        raise ValueError(
            "dataset_download_workers must be an integer between "
            f"1 and {MAX_DATASET_DOWNLOAD_WORKERS}."
        )
    return download_workers


def _looks_like_local_path(location: str, path: Path) -> bool:
    return path.exists() or path.is_absolute() or location.startswith(("./", "../", "~/"))


def resolve_dataset_source(
    location: str | os.PathLike[str],
    *,
    revision: str | None,
    download_workers: int = DEFAULT_DATASET_DOWNLOAD_WORKERS,
) -> DatasetSource:
    """Resolve a local path or a revision-pinned remote dataset.

    Existing paths and explicit filesystem-looking paths never fall back to the
    Hub. Remote sources require a non-empty revision other than the mutable
    ``main`` branch.
    """
    if not isinstance(location, (str, os.PathLike)):
        raise TypeError("dataset location must be a string or filesystem path.")
    explicitly_pathlike = isinstance(location, os.PathLike)
    source = os.fspath(location).strip()
    if not source:
        raise ValueError("dataset location must be non-empty.")

    workers = _validate_download_workers(download_workers)
    path = Path(source).expanduser()
    if explicitly_pathlike or _looks_like_local_path(source, path):
        if not path.exists():
            raise FileNotFoundError(f"Local dataset source does not exist: {path}")
        return DatasetSource(
            location=str(path),
            revision=None,
            download_workers=workers,
            is_local=True,
        )

    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(
            "Remote dataset preparation requires a full 40-character dataset_revision commit."
        )
    resolved_revision = revision.strip()
    if not _HUB_COMMIT_PATTERN.fullmatch(resolved_revision):
        raise ValueError(
            "dataset_revision must be a full 40-character Hub commit SHA; mutable branches and "
            "tags are not reproducible dataset sources."
        )

    return DatasetSource(
        location=source,
        revision=resolved_revision,
        download_workers=workers,
        is_local=False,
    )


__all__ = [
    "DEFAULT_DATASET_DOWNLOAD_WORKERS",
    "MAX_DATASET_DOWNLOAD_WORKERS",
    "DatasetSource",
    "resolve_dataset_source",
]
