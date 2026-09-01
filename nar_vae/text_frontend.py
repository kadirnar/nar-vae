"""Versioned frozen multilingual text-feature providers.

The pretrained model is intentionally not a child of the acoustic model.  It therefore cannot
enter an acoustic checkpoint or optimizer by accident.  Prepared datasets may cache the returned
contextual states; inference may run the same provider online from the exact pinned revision.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch

_HUB_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
FROZEN_TEXT_FRONTEND_VERSION = 2


@dataclass(frozen=True)
class FrozenTextFrontendSpec:
    """Immutable contract for one external contextual text provider."""

    model_id: str
    revision: str
    hidden_size: int
    max_length: int
    hidden_layer: int = -1
    input_mode: str = "raw_text"
    feature_dtype: str = "float16"
    license: str = ""
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = False
    contract_version: int = FROZEN_TEXT_FRONTEND_VERSION

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("Frozen text model_id must be non-empty.")
        if not _HUB_COMMIT.fullmatch(self.revision):
            raise ValueError("Frozen text revision must be a full 40-character commit SHA.")
        tokenizer_revision = self.tokenizer_revision or self.revision
        if not _HUB_COMMIT.fullmatch(tokenizer_revision):
            raise ValueError("Tokenizer revision must be a full 40-character commit SHA.")
        if self.hidden_size <= 0 or self.max_length <= 0:
            raise ValueError("Frozen text hidden_size and max_length must be positive.")
        if self.input_mode not in {"raw_text", "phonemes"}:
            raise ValueError("Frozen text input_mode must be 'raw_text' or 'phonemes'.")
        if self.feature_dtype not in {"float16", "float32"}:
            raise ValueError("Unsupported frozen text feature_dtype.")
        if not self.license.strip():
            raise ValueError("Frozen text provider license must be declared.")
        if self.trust_remote_code:
            raise ValueError("Frozen text providers require trust_remote_code=False.")
        if self.contract_version != FROZEN_TEXT_FRONTEND_VERSION:
            raise ValueError("Unsupported frozen text frontend contract version.")

    @property
    def resolved_tokenizer_id(self) -> str:
        return self.tokenizer_id or self.model_id

    @property
    def resolved_tokenizer_revision(self) -> str:
        return self.tokenizer_revision or self.revision

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.feature_dtype]

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def contract_name(self) -> str:
        return f"nar_vae.frozen_hf_features/{self.model_id}@{self.revision}#{self.fingerprint}"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrozenTextFrontendSpec":
        if config.get("text_encoder_type", "scratch") != "frozen_features":
            raise ValueError("The configuration does not select a frozen-feature frontend.")
        return cls(
            model_id=config["text_frontend_model"],
            revision=config["text_frontend_revision"],
            hidden_size=int(config["frozen_text_input_size"]),
            max_length=int(config["text_frontend_max_length"]),
            hidden_layer=int(config.get("text_frontend_layer", -1)),
            input_mode=config["text_frontend_input_mode"],
            feature_dtype=config["text_frontend_dtype"],
            license=config["text_frontend_license"],
            tokenizer_id=config.get("text_frontend_tokenizer_model"),
            tokenizer_revision=config.get("text_frontend_tokenizer_revision"),
            trust_remote_code=False,
        )


@dataclass(frozen=True)
class FrozenTextBatch:
    """Token IDs, validity mask, and contextual states sharing the same token axis."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    features: torch.Tensor


class FrozenTextFrontend:
    """No-grad Hugging Face frontend kept outside the trainable acoustic module."""

    def __init__(
        self,
        spec: FrozenTextFrontendSpec,
        *,
        device: torch.device | str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - dependency failure
            raise RuntimeError("transformers is required for a frozen text frontend.") from exc

        self.spec = spec
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.resolved_tokenizer_id,
            revision=spec.resolved_tokenizer_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.assert_frozen()

    def assert_frozen(self) -> None:
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("The external text provider must remain in eval mode and frozen.")

    @property
    def num_parameters(self) -> int:
        """Return external parameters, which are intentionally outside acoustic counts."""
        return sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        inputs_are_phonemes: bool = False,
    ) -> FrozenTextBatch:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Frozen text input must contain non-empty strings.")
        if self.spec.input_mode == "phonemes" and not inputs_are_phonemes:
            raise ValueError(
                "This provider expects already phonemized strings. Pass a separately versioned "
                "normalizer/G2P result and set inputs_are_phonemes=True; raw text is not guessed."
            )
        if self.spec.input_mode == "raw_text" and inputs_are_phonemes:
            raise ValueError("A raw-text provider cannot be marked as phoneme input.")

        self.model.eval()
        self.assert_frozen()
        # NFC is deliberately the only built-in normalization. In particular, Turkish casing,
        # punctuation, numbers, and apostrophes are preserved for the pinned tokenizer. The
        # frontend contract version makes this preprocessing rule part of the cache fingerprint.
        normalized_texts = [unicodedata.normalize("NFC", text.strip()) for text in texts]
        encoded = self.tokenizer(
            normalized_texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].to(dtype=torch.bool)
        if input_ids.shape[1] > self.spec.max_length:
            raise ValueError(
                f"Frozen text input has {input_ids.shape[1]} tokens, exceeding the pinned "
                f"maximum of {self.spec.max_length}; inputs are never silently truncated."
            )
        model_inputs = {
            name: value.to(self.device)
            for name, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        if self.spec.hidden_layer == -1:
            output = self.model(**model_inputs)
            features = output.last_hidden_state
        else:
            output = self.model(**model_inputs, output_hidden_states=True)
            features = output.hidden_states[self.spec.hidden_layer]
        if features.ndim != 3 or features.shape[-1] != self.spec.hidden_size:
            raise RuntimeError(
                "Frozen provider hidden width does not match the versioned frontend contract: "
                f"{tuple(features.shape)} versus {self.spec.hidden_size}."
            )
        features = features.masked_fill(~attention_mask.to(self.device)[:, :, None], 0.0)
        return FrozenTextBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            features=features.to(dtype=self.spec.torch_dtype).cpu(),
        )


__all__ = [
    "FROZEN_TEXT_FRONTEND_VERSION",
    "FrozenTextBatch",
    "FrozenTextFrontend",
    "FrozenTextFrontendSpec",
]
