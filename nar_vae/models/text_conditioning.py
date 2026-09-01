"""Versioned text-state inputs for the acoustic diffusion model.

The frozen-feature mode deliberately accepts already computed, token-aligned
hidden states.  A pretrained text backbone belongs to dataset preparation and
inference, not to the acoustic checkpoint, so this module never imports or
registers a Hugging Face model.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn as nn

SCRATCH_TOKEN_TEXT_CONDITIONING = "scratch_tokens"
FROZEN_FEATURE_TEXT_CONDITIONING = "frozen_features"

TEXT_CONDITIONING_VERSION = 1
FROZEN_FEATURE_ADAPTER_VERSION = 1

TEXT_CONDITIONING_VERSION_KEY = "text_conditioning_version"
TEXT_CONDITIONING_MODE_KEY = "text_conditioning_mode_metadata"
TEXT_CONDITIONING_FEATURE_SIZE_KEY = "text_conditioning_feature_size_metadata"
TEXT_CONDITIONING_ADAPTER_VERSION_KEY = "text_conditioning_adapter_version_metadata"

_MODE_CODES = {
    SCRATCH_TOKEN_TEXT_CONDITIONING: 0,
    FROZEN_FEATURE_TEXT_CONDITIONING: 1,
}


@dataclass(frozen=True)
class TextConditioningMetadata:
    """Exact acoustic-checkpoint topology for one text-conditioning mode."""

    mode: str
    mode_code: int
    feature_size: int
    adapter_version: int


def resolve_text_conditioning_metadata(
    mode: str,
    conditioning_feature_size: int | None,
) -> TextConditioningMetadata:
    """Validate a mode and return scalar values suitable for state-dict buffers."""
    if not isinstance(mode, str) or mode not in _MODE_CODES:
        choices = ", ".join(sorted(_MODE_CODES))
        raise ValueError(f"text_conditioning_mode must be one of: {choices}.")

    if mode == SCRATCH_TOKEN_TEXT_CONDITIONING:
        if conditioning_feature_size not in (None, 0):
            raise ValueError(
                "conditioning_feature_size is only valid with frozen_features text conditioning."
            )
        return TextConditioningMetadata(
            mode=mode,
            mode_code=_MODE_CODES[mode],
            feature_size=0,
            adapter_version=0,
        )

    if (
        isinstance(conditioning_feature_size, bool)
        or not isinstance(conditioning_feature_size, Integral)
        or conditioning_feature_size <= 0
    ):
        raise ValueError(
            "frozen_features text conditioning requires a positive conditioning_feature_size."
        )
    return TextConditioningMetadata(
        mode=mode,
        mode_code=_MODE_CODES[mode],
        feature_size=int(conditioning_feature_size),
        adapter_version=FROZEN_FEATURE_ADAPTER_VERSION,
    )


class FrozenTextFeatureAdapter(nn.Module):
    """Project immutable contextual features into EchoDiT's text-state width.

    ``features`` must already share its token axis with ``conditioning_ids``,
    ``token_language_ids`` and ``alignment_mask``.  Detaching the input makes the
    frozen-backbone boundary explicit even when a caller accidentally supplies a
    tensor that requires gradients.
    """

    def __init__(
        self,
        *,
        feature_size: int,
        model_size: int,
        num_languages: int = 0,
    ) -> None:
        super().__init__()
        for name, value in (("feature_size", feature_size), ("model_size", model_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(num_languages, bool) or not isinstance(num_languages, Integral):
            raise ValueError("num_languages must be a non-negative integer.")
        if num_languages < 0:
            raise ValueError("num_languages must be a non-negative integer.")

        self.feature_size = int(feature_size)
        self.model_size = int(model_size)
        self.feature_projection = nn.Linear(self.feature_size, self.model_size, bias=True)
        self.language_embedding = (
            nn.Embedding(int(num_languages) + 1, self.model_size, padding_idx=0)
            if num_languages > 0
            else None
        )

        # EchoDiT toggles this attribute on every active conditioner.  There are
        # no internal Transformer blocks to checkpoint in frozen-feature mode.
        self.gradient_checkpointing = False
        self._gradient_checkpointing_kwargs: dict = {}

    def forward(
        self,
        features: torch.Tensor,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_size:
            raise ValueError(
                f"conditioning_features must have shape [batch, token, {self.feature_size}]."
            )
        if not torch.is_floating_point(features):
            raise TypeError("conditioning_features must use a floating-point dtype.")

        projection = self.feature_projection
        state = projection(
            features.detach().to(device=projection.weight.device, dtype=projection.weight.dtype)
        )
        if self.language_embedding is None:
            if language_ids is not None:
                raise ValueError(
                    "token language IDs require a language-conditioned frozen-feature adapter."
                )
            return state

        if language_ids is None:
            raise ValueError(
                "token language IDs are required by this multilingual frozen-feature adapter."
            )
        if language_ids.ndim == 1:
            if language_ids.shape[0] != features.shape[0]:
                raise ValueError("language_ids must have shape [batch] or [batch, token].")
            language_state = self.language_embedding(
                language_ids.to(device=state.device, dtype=torch.long)
            )[:, None, :]
        elif language_ids.ndim == 2:
            if tuple(language_ids.shape) != tuple(features.shape[:2]):
                raise ValueError("language_ids must have shape [batch] or [batch, token].")
            language_state = self.language_embedding(
                language_ids.to(device=state.device, dtype=torch.long)
            )
        else:
            raise ValueError("language_ids must have shape [batch] or [batch, token].")
        return state + language_state.to(dtype=state.dtype)


__all__ = [
    "FROZEN_FEATURE_ADAPTER_VERSION",
    "FROZEN_FEATURE_TEXT_CONDITIONING",
    "FrozenTextFeatureAdapter",
    "SCRATCH_TOKEN_TEXT_CONDITIONING",
    "TEXT_CONDITIONING_ADAPTER_VERSION_KEY",
    "TEXT_CONDITIONING_FEATURE_SIZE_KEY",
    "TEXT_CONDITIONING_MODE_KEY",
    "TEXT_CONDITIONING_VERSION",
    "TEXT_CONDITIONING_VERSION_KEY",
    "TextConditioningMetadata",
    "resolve_text_conditioning_metadata",
]
