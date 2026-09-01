"""Reusable benchmark statistics and environment capture."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty series."""
    if not values:
        raise ValueError("Cannot calculate a percentile for an empty timing series.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int]:
    """Return stable descriptive statistics for one timing series."""
    if not values:
        raise ValueError("Cannot summarize an empty timing series.")
    return {
        "count": len(values),
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "p95_s": percentile(values, 0.95),
        "min_s": min(values),
        "max_s": max(values),
    }


def command_output(command: list[str]) -> str | None:
    """Run a read-only environment command and return stripped stdout."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def sha256(path: Path) -> str:
    """Hash a local benchmark artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_metadata(path: Path, *, label: str | None = None) -> dict[str, str | int]:
    """Bind a benchmark record to the exact bytes of a local artifact."""
    return {
        "path": label or path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def package_source_hashes(package_root: Path) -> dict[str, str]:
    """Hash every packaged source/config file that can affect a benchmark."""
    repository_root = package_root.parent
    paths = {
        *package_root.rglob("*.py"),
        *package_root.rglob("*.toml"),
        *package_root.rglob("*.yaml"),
    }
    return {
        str(path.relative_to(repository_root)): sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }


def environment() -> dict[str, Any]:
    """Capture the software and accelerator facts needed to interpret timings."""
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "nar_vae_use_fa3": os.environ.get("NAR_VAE_USE_FA3"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu": properties.name,
                "gpu_memory_bytes": properties.total_memory,
                "driver": command_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ]
                ),
                "cudnn": torch.backends.cudnn.version(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            }
        )
    return result


__all__ = [
    "command_output",
    "environment",
    "file_metadata",
    "package_source_hashes",
    "percentile",
    "sha256",
    "summarize",
]
