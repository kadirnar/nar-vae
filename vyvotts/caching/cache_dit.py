"""Real Cache-DiT DBCache integration for the EchoDiT backbone."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from threading import Lock
from types import ModuleType
from typing import Any

import torch.nn as nn

from vyvotts.configuration import CACHE_DIT_MIN_STEPS

_SESSION_LOCK = Lock()


class CacheDiTUnavailableError(RuntimeError):
    """Raised when the turbo profile cannot initialize Cache-DiT."""


class CacheDiTPoisonedError(RuntimeError):
    """Raised when failed backend cleanup makes a model unsafe to reuse."""


class CacheDiTRequestActiveError(RuntimeError):
    """Raised when teardown races with an in-flight persistent request."""


_POISON_ATTRIBUTE = "_nar_vae_cache_dit_poison_reason"


def assert_cache_dit_healthy(model: nn.Module) -> None:
    """Reject a model whose Cache-DiT hooks could not be removed cleanly."""
    reason = getattr(model, _POISON_ATTRIBUTE, None)
    if reason is not None:
        raise CacheDiTPoisonedError(
            "Cache-DiT cleanup previously failed and this model may still contain backend "
            "hooks. Reconstruct the inference runtime before any cached or uncached use. "
            f"Cleanup error: {reason}"
        )


def _poison_model(model: nn.Module, error: BaseException) -> None:
    setattr(model, _POISON_ATTRIBUTE, f"{type(error).__name__}: {error}")


@dataclass(frozen=True)
class CacheDiTStats:
    """Small, stable view of Cache-DiT's request statistics."""

    version: str | None = None
    cached_steps: int = 0
    executed_steps: int = 0
    block_count: int = 0
    computed_blocks_per_cached_step: int = 0

    @property
    def cache_ratio(self) -> float:
        return self.cached_steps / self.executed_steps if self.executed_steps else 0.0

    @property
    def baseline_block_calls(self) -> int:
        """Block calls without DBCache for the same solver-step count."""
        return self.executed_steps * self.block_count

    @property
    def estimated_block_calls(self) -> int:
        """Main DiT block calls after applying the observed cache hits."""
        skipped_per_hit = max(self.block_count - self.computed_blocks_per_cached_step, 0)
        return self.baseline_block_calls - self.cached_steps * skipped_per_hit

    @property
    def block_work_reduction(self) -> float:
        """Estimated fraction of main DiT block calls avoided by DBCache."""
        baseline = self.baseline_block_calls
        return 1.0 - self.estimated_block_calls / baseline if baseline else 0.0


def _load_cache_dit() -> ModuleType:
    try:
        module = import_module("cache_dit")
    except ModuleNotFoundError as exc:
        if exc.name != "cache_dit":
            raise
        raise CacheDiTUnavailableError(
            "The turbo profile requires Cache-DiT. Install it with `pip install 'nar-vae[turbo]'`."
        ) from exc

    required = (
        "BlockAdapter",
        "DBCacheConfig",
        "DMDCalibratorConfig",
        "ForwardPattern",
        "disable_cache",
        "enable_cache",
        "summary",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        names = ", ".join(missing)
        raise CacheDiTUnavailableError(
            f"The installed Cache-DiT version is missing required APIs: {names}. "
            "Install NAR-VAE turbo support with `pip install 'nar-vae[turbo]'`."
        )
    return module


def _cache_dit_version(module: ModuleType) -> str | None:
    try:
        return distribution_version("cache-dit")
    except PackageNotFoundError:
        return getattr(module, "__version__", None)


def _turbo_step_mask(num_steps: int) -> list[int]:
    """Use isolated cache hits after DMD has six computed states to learn from."""
    if num_steps < CACHE_DIT_MIN_STEPS:
        raise ValueError(f"Cache-DiT turbo mode requires at least {CACHE_DIT_MIN_STEPS} steps.")
    mask = [1] * num_steps
    for step in range(6, num_steps - 1, 3):
        mask[step] = 0
    return mask


def _echo_dit_backbone(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "dit", None)
    blocks = getattr(backbone, "blocks", None)
    if backbone is None or not isinstance(blocks, nn.ModuleList) or len(blocks) < 2:
        raise TypeError(
            "Cache-DiT turbo mode requires a FlowMatchingEchoDiT model with at least two blocks."
        )
    return backbone


def _cache_configs(
    api: ModuleType,
    *,
    num_steps: int,
    block_count: int,
) -> tuple[Any, Any, int]:
    """Build the fixed turbo policy shared by transient and compiled sessions."""
    first_blocks = min(8, max(1, block_count // 3))
    cache_config = api.DBCacheConfig(
        Fn_compute_blocks=first_blocks,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.08,
        max_warmup_steps=0,
        max_cached_steps=-1,
        max_continuous_cached_steps=1,
        enable_separate_cfg=False,
        num_inference_steps=num_steps,
        steps_computation_mask=_turbo_step_mask(num_steps),
        steps_computation_policy="static",
    )
    calibrator_config = api.DMDCalibratorConfig(
        dmd_history=6,
        dmd_rank=0,
        dmd_ridge=1e-8,
    )
    return cache_config, calibrator_config, first_blocks


class CacheDiTSession:
    """Enable DBCache for one request, capture stats, then restore the model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        num_steps: int,
    ) -> None:
        if num_steps < CACHE_DIT_MIN_STEPS:
            raise ValueError(f"Cache-DiT turbo mode requires at least {CACHE_DIT_MIN_STEPS} steps.")
        self.model = model
        self.num_steps = num_steps
        self.stats = CacheDiTStats()
        self._api: ModuleType | None = None
        self._adapter: Any | None = None
        self._backbone: nn.Module | None = None
        self._enabled = False
        self._owns_lock = False
        self._request_lock = Lock()
        self._block_count = 0

    def __enter__(self) -> "CacheDiTSession":
        assert_cache_dit_healthy(self.model)
        if not _SESSION_LOCK.acquire(blocking=False):
            raise RuntimeError(
                "Cache-DiT turbo mode does not support concurrent or nested sessions."
            )
        self._owns_lock = True
        try:
            api = _load_cache_dit()
            backbone = _echo_dit_backbone(self.model)
            block_count = len(backbone.blocks)
            cache_config, calibrator_config, first_blocks = _cache_configs(
                api,
                num_steps=self.num_steps,
                block_count=block_count,
            )

            adapter = api.BlockAdapter(
                pipe=None,
                transformer=backbone,
                blocks=backbone.blocks,
                forward_pattern=api.ForwardPattern.Pattern_3,
                check_forward_pattern=True,
                check_num_outputs=True,
                has_separate_cfg=False,
            )

            self._api = api
            self._adapter = adapter
            self._backbone = backbone
            self._block_count = block_count
            self.stats = CacheDiTStats(
                version=_cache_dit_version(api),
                block_count=block_count,
                computed_blocks_per_cached_step=first_blocks,
            )
            api.enable_cache(
                adapter,
                cache_config=cache_config,
                calibrator_config=calibrator_config,
            )
            self._enabled = True
        except BaseException as setup_error:
            if self._api is not None and self._adapter is not None:
                try:
                    self._api.disable_cache(self._adapter)
                except BaseException as cleanup_error:
                    _poison_model(self.model, cleanup_error)
                    warnings.warn(
                        "Cache-DiT cleanup also failed after setup raised "
                        f"{type(setup_error).__name__}: {cleanup_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self._reset_runtime_state()
            raise
        return self

    @property
    def backbone(self) -> nn.Module:
        """Return the hooked EchoDiT backbone while the session is active."""
        if not self._enabled or self._backbone is None:
            raise RuntimeError("Cache-DiT session is not active.")
        return self._backbone

    @property
    def api(self) -> ModuleType:
        """Return the loaded Cache-DiT module while the session is active."""
        if not self._enabled or self._api is None:
            raise RuntimeError("Cache-DiT session is not active.")
        return self._api

    def refresh(self, num_steps: int) -> None:
        """Reset a persistent cache context for one new request."""
        if num_steps < CACHE_DIT_MIN_STEPS:
            raise ValueError(f"Cache-DiT turbo mode requires at least {CACHE_DIT_MIN_STEPS} steps.")
        api = self.api
        backbone = self.backbone
        refresh_context = getattr(api, "refresh_context", None)
        if not callable(refresh_context):
            raise CacheDiTUnavailableError(
                "The installed Cache-DiT version does not support persistent compiled sessions. "
                "Install NAR-VAE turbo support with `pip install 'nar-vae[turbo]'`."
            )
        cache_config, calibrator_config, first_blocks = _cache_configs(
            api,
            num_steps=num_steps,
            block_count=self._block_count,
        )
        # Clear the previous request's counters before touching the backend. If
        # refresh_context fails partway through, no stale successful statistics
        # remain observable on this session.
        self.stats = CacheDiTStats(
            version=_cache_dit_version(api),
            block_count=self._block_count,
            computed_blocks_per_cached_step=first_blocks,
        )
        refresh_context(
            backbone,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
            verbose=False,
        )
        self.num_steps = num_steps

    def request(self, num_steps: int) -> "_PersistentCacheDiTRequest":
        """Create a non-overlapping request context without removing cache hooks."""
        if not self._enabled:
            raise RuntimeError("Cache-DiT session is not active.")
        return _PersistentCacheDiTRequest(self, num_steps)

    def close(self) -> None:
        """Disable cache hooks and release a persistent session; safe to call twice."""
        # Checking ``locked()`` and tearing down separately permits a request
        # to enter between the two operations. Owning the same lock used by
        # request contexts makes the decision and teardown one atomic region.
        if not self._request_lock.acquire(blocking=False):
            raise CacheDiTRequestActiveError("Cannot close Cache-DiT while a request is running.")
        try:
            self._close_locked(None, None, None)
        finally:
            self._request_lock.release()

    def _close_locked(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Tear down hooks while the caller owns ``_request_lock``."""
        if self._enabled:
            self.__exit__(exc_type, exc, traceback)

    def _reset_runtime_state(self) -> None:
        self._enabled = False
        self._adapter = None
        self._backbone = None
        self._api = None
        if self._owns_lock:
            _SESSION_LOCK.release()
            self._owns_lock = False

    def collect_stats(self) -> CacheDiTStats:
        if not self._enabled or self._api is None or self._backbone is None:
            return self.stats

        summaries: list[Any] = self._api.summary(self._backbone, logging=False)
        cached_steps = max(
            (int(getattr(summary, "accumulated_cached_steps", 0)) for summary in summaries),
            default=0,
        )
        executed_steps = max(
            (
                int(
                    getattr(summary, "accumulated_transformer_executed_steps", 0)
                    or getattr(summary, "accumulated_executed_steps", 0)
                )
                for summary in summaries
            ),
            default=0,
        )
        self.stats = CacheDiTStats(
            version=self.stats.version,
            cached_steps=cached_steps,
            executed_steps=executed_steps,
            block_count=self.stats.block_count,
            computed_blocks_per_cached_step=self.stats.computed_blocks_per_cached_step,
        )
        return self.stats

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc, traceback
        if (
            not self._enabled
            or self._api is None
            or self._adapter is None
            or self._backbone is None
        ):
            self._reset_runtime_state()
            return
        try:
            try:
                self.collect_stats()
            except Exception as stats_error:
                warnings.warn(
                    f"Cache-DiT statistics could not be collected: {stats_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        finally:
            try:
                self._api.disable_cache(self._adapter)
            except BaseException as cleanup_error:
                _poison_model(self.model, cleanup_error)
                if exc_type is None:
                    raise
                warnings.warn(
                    f"Cache-DiT cleanup failed after inference raised an error: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            finally:
                self._reset_runtime_state()


class _PersistentCacheDiTRequest:
    """Refresh and collect one request while keeping compiled hooks installed."""

    def __init__(self, session: CacheDiTSession, num_steps: int) -> None:
        self.session = session
        self.num_steps = num_steps
        self._owns_lock = False

    def __enter__(self) -> CacheDiTSession:
        if not self.session._request_lock.acquire(blocking=False):
            raise RuntimeError("Compiled Cache-DiT does not support concurrent requests.")
        self._owns_lock = True
        try:
            self.session.refresh(self.num_steps)
        except BaseException as error:
            self._invalidate_session(type(error), error, error.__traceback__)
            raise
        return self.session

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is not None:
            # A partially completed compiled request may leave backend-owned
            # residuals in an unknown state. Disable the persistent hooks and
            # require explicit runtime reconstruction before another request.
            self._invalidate_session(exc_type, exc, traceback)
            return
        try:
            try:
                self.session.collect_stats()
            except Exception as stats_error:
                warnings.warn(
                    f"Cache-DiT statistics could not be collected: {stats_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        finally:
            self._release_lock()

    def _release_lock(self) -> None:
        if self._owns_lock:
            self.session._request_lock.release()
            self._owns_lock = False

    def _invalidate_session(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Best-effort cleanup under the request lock without hiding the original error."""
        try:
            self.session._close_locked(exc_type, exc, traceback)
        except BaseException as cleanup_error:
            warnings.warn(
                f"Cache-DiT cleanup failed while invalidating a request: {cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            self._release_lock()
