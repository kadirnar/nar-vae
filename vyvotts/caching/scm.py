"""Legacy whole-step computation masking for the Euler solver."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class VelocityPredictor:
    """Reuse or linearly extrapolate velocity on a masked solver step."""

    def __init__(self, order: int = 1):
        if order not in (0, 1):
            raise ValueError("predictor order must be 0 or 1.")
        self.order = order
        self.v_prev: torch.Tensor | None = None
        self.v_last: torch.Tensor | None = None

    def reset(self) -> None:
        self.v_prev = None
        self.v_last = None

    def update(self, velocity: torch.Tensor) -> None:
        self.v_prev = self.v_last
        self.v_last = velocity

    def predict(self) -> torch.Tensor:
        if self.v_last is None:
            raise RuntimeError("A velocity must be computed before a solver step can be cached.")
        if self.order == 0 or self.v_prev is None:
            return self.v_last
        return 2 * self.v_last - self.v_prev


@dataclass
class SCMContext:
    """Track a Cache-DiT step mask used by the legacy solver-level cache."""

    mask: list[int]
    predictor: VelocityPredictor
    total_steps: int = 0
    cached_steps: int = 0

    def reset(self) -> None:
        self.predictor.reset()
        self.total_steps = 0
        self.cached_steps = 0

    def is_compute_step(self, step_idx: int) -> bool:
        return step_idx >= len(self.mask) or self.mask[step_idx] == 1

    @property
    def cache_ratio(self) -> float:
        return self.cached_steps / self.total_steps if self.total_steps else 0.0


def create_scm_context(
    mask_policy: str = "ultra",
    num_steps: int = 50,
    predictor_order: int = 1,
) -> SCMContext:
    """Create the legacy whole-step cache from Cache-DiT's public mask API."""
    try:
        from cache_dit import steps_mask
    except ModuleNotFoundError as exc:
        if exc.name != "cache_dit":
            raise
        raise RuntimeError(
            "Solver-level caching requires Cache-DiT. Install it with "
            "`pip install 'nar-vae[turbo]'`."
        ) from exc

    mask = steps_mask(mask_policy=mask_policy, total_steps=num_steps)
    return SCMContext(mask=mask, predictor=VelocityPredictor(order=predictor_order))
