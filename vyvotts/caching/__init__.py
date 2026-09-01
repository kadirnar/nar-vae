"""Optional inference caching backends used by NAR-VAE."""

from .cache_dit import (
    CacheDiTPoisonedError,
    CacheDiTRequestActiveError,
    CacheDiTSession,
    CacheDiTStats,
    CacheDiTUnavailableError,
    assert_cache_dit_healthy,
)
from .scm import SCMContext, VelocityPredictor, create_scm_context

__all__ = [
    "CacheDiTPoisonedError",
    "CacheDiTRequestActiveError",
    "CacheDiTSession",
    "CacheDiTStats",
    "CacheDiTUnavailableError",
    "SCMContext",
    "VelocityPredictor",
    "assert_cache_dit_healthy",
    "create_scm_context",
]
