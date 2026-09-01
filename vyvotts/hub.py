"""Hugging Face Hub download helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import get_token, snapshot_download

DEFAULT_IGNORE_PATTERNS = ("*.md", ".gitattributes")
_HUB_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


def resolve_revision(repo_id: str, revision: str | None) -> str:
    """Require an immutable Hub commit instead of resolving an implicit branch."""
    if not repo_id.strip():
        raise ValueError("repo_id must be non-empty.")
    if revision is None or not _HUB_COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(f"{repo_id!r} requires an explicit 40-character Hub commit revision.")
    return revision


def download_snapshot(
    *,
    repo_id: str,
    repo_type: str,
    local_dir: str | os.PathLike[str],
    revision: str | None = None,
    ignore_patterns: Sequence[str] | None = None,
    allow_patterns: Sequence[str] | None = None,
    max_workers: int = 8,
) -> Path:
    """Download a commit-pinned repository snapshot using the configured Hub token."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    resolved_revision = resolve_revision(repo_id, revision)
    token = os.environ.get("HF_TOKEN") or get_token()
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=resolved_revision,
        local_dir=os.fspath(local_dir),
        ignore_patterns=list(
            DEFAULT_IGNORE_PATTERNS if ignore_patterns is None else ignore_patterns
        ),
        allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
        max_workers=max_workers,
        token=token,
    )
    return Path(snapshot_path)


__all__ = [
    "DEFAULT_IGNORE_PATTERNS",
    "download_snapshot",
    "resolve_revision",
]
