from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from nar_vae.languages import DEFAULT_LANGUAGE, language_from_id, language_id
from nar_vae.tokenization import PAD_TOKEN, TOTAL_VOCAB_SIZE, token_receives_alignment

_LEGACY_CL100K_LENGTH = 100277


def _is_legacy_frontend_sample(sample: dict[str, Any]) -> bool:
    contract = sample.get("representation_contract")
    if isinstance(contract, dict):
        return contract.get("text_frontend_name") == "nar_vae.encode_tts_text/cl100k_base"
    return any(int(token_id) >= TOTAL_VOCAB_SIZE for token_id in sample["conditioning_ids"])


def _parallel_text_fields(
    sample: dict[str, Any],
    token_ids: torch.Tensor,
    *,
    target_language_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve v2 masks or conservatively adapt one explicitly legacy row."""
    raw_languages = sample.get("token_language_ids")
    raw_alignment = sample.get("alignment_mask")
    if (raw_languages is None) != (raw_alignment is None):
        raise ValueError(
            "token_language_ids and alignment_mask must be present together or both omitted"
        )
    if raw_languages is not None:
        token_languages = torch.as_tensor(raw_languages, dtype=torch.long)
        alignment = torch.as_tensor(raw_alignment, dtype=torch.bool)
        if tuple(token_languages.shape) != tuple(token_ids.shape):
            raise ValueError("token_language_ids must match conditioning_ids length")
        if tuple(alignment.shape) != tuple(token_ids.shape):
            raise ValueError("alignment_mask must match conditioning_ids length")
        if not bool(alignment.any()):
            raise ValueError("alignment_mask must contain at least one speech-content token")
        return token_languages, alignment

    if _is_legacy_frontend_sample(sample):
        # cl100k text IDs are below 100277; all old control/emotion IDs are
        # above it. Punctuation embedded inside a BPE token cannot be separated,
        # which is why legacy adaptation must remain explicit and conservative.
        alignment = token_ids < _LEGACY_CL100K_LENGTH
    else:
        alignment = torch.tensor(
            [token_receives_alignment(int(token_id)) for token_id in token_ids],
            dtype=torch.bool,
        )
    token_languages = torch.where(
        alignment,
        torch.full_like(token_ids, target_language_id),
        torch.zeros_like(token_ids),
    )
    return token_languages, alignment


def _language_id_from_sample(
    sample: dict[str, Any],
    *,
    code_key: str,
    id_key: str,
    default: str,
) -> int:
    """Resolve either human-readable or precomputed language metadata."""
    declared_id = sample.get(id_key)
    if code_key not in sample and declared_id is not None:
        declared_id = int(declared_id)
        language_from_id(declared_id)
        return declared_id
    declared_code = sample.get(code_key, default)
    resolved_id = language_id(declared_code)
    if declared_id is None:
        return resolved_id
    declared_id = int(declared_id)
    language_from_id(declared_id)
    if declared_id != resolved_id:
        raise ValueError(f"Conflicting {code_key}={declared_code!r} and {id_key}={declared_id}.")
    return declared_id


class FlowMatchingDataCollator:
    """Data collator for TTS flow-matching batches.

    Handles:
    - continuous acoustic latents and text conditioning IDs
    - Variable-length sequences with padding

    Args:
        pad_token: Token ID for padding
        latent_pad_value: Value for padding latents (default: 0.0)
        speaker_patch_size: Number of adjacent speaker frames per encoder patch
    """

    def __init__(
        self,
        pad_token: int = PAD_TOKEN,
        latent_pad_value: float = 0.0,
        speaker_patch_size: int = 4,
    ):
        if speaker_patch_size <= 0:
            raise ValueError("speaker_patch_size must be greater than zero")
        self.pad_token = pad_token
        self.latent_pad_value = latent_pad_value
        self.speaker_patch_size = speaker_patch_size

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate batch of features.

        Args:
            features: List of dataset samples

        Returns:
            Batched dictionary with appropriate padding
        """
        if not features:
            raise ValueError("A TTS batch must contain at least one feature.")
        missing_latents = [
            index for index, feature in enumerate(features) if "latents" not in feature
        ]
        if missing_latents:
            raise ValueError(
                "NAR-VAE training batches require acoustic latent rows. "
                f"Rows without latents: {missing_latents}."
            )
        return self._collate_tts(features)

    def _collate_tts(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate TTS samples with continuous latents and speaker reference.

        Args:
            samples: List of TTS samples with latents, conditioning_ids, speaker_latents

        Returns:
            Dictionary with padded tensors
        """
        # Extract latents (numpy arrays) and convert to tensors
        latents_list = []
        for s in samples:
            latent = s["latents"]
            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent).float()
            elif not isinstance(latent, torch.Tensor):
                latent = torch.tensor(latent, dtype=torch.float32)
            latents_list.append(latent)  # [D, T]

        # Pad latents along time dimension
        # First transpose to [T, D] for pad_sequence, then back to [D, T]
        latents_transposed = [latent.transpose(0, 1) for latent in latents_list]  # [T, D]
        latents_padded = pad_sequence(
            latents_transposed, batch_first=True, padding_value=self.latent_pad_value
        )  # [B, T_max, D]
        latents_padded = latents_padded.transpose(1, 2)  # [B, D, T_max]

        # Create latent mask (1 for valid, 0 for padding)
        batch_size = len(latents_list)
        max_len = latents_padded.shape[2]
        latent_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, latent in enumerate(latents_list):
            latent_mask[i, : latent.shape[1]] = True

        # Extract and pad conditioning IDs
        conditioning_ids = [torch.tensor(s["conditioning_ids"], dtype=torch.long) for s in samples]
        if any(ids.ndim != 1 or ids.numel() == 0 for ids in conditioning_ids):
            raise ValueError("conditioning_ids must be non-empty one-dimensional sequences")
        conditioning_ids_padded = pad_sequence(
            conditioning_ids, batch_first=True, padding_value=self.pad_token
        )

        # Frozen contextual features are cached outside the acoustic model.  They
        # must use exactly the same token axis as every parallel text field so MAS
        # never aligns one frontend's tokens to another frontend's states.
        samples_with_features = ["conditioning_features" in sample for sample in samples]
        if any(samples_with_features) and not all(samples_with_features):
            raise ValueError(
                "conditioning_features must be present in every sample or omitted from the "
                "entire batch"
            )
        conditioning_features_padded = None
        if all(samples_with_features):
            dtype_metadata = [sample.get("conditioning_feature_dtype") for sample in samples]
            if any(value is None for value in dtype_metadata):
                raise ValueError(
                    "Frozen conditioning_features require conditioning_feature_dtype in every "
                    "sample."
                )
            if len(set(dtype_metadata)) != 1:
                raise ValueError(
                    "conditioning_feature_dtype must be identical across a frozen batch."
                )
            declared_dtype_name = dtype_metadata[0]
            try:
                declared_dtype = {
                    "float16": torch.float16,
                    "float32": torch.float32,
                }[declared_dtype_name]
            except KeyError as exc:
                raise ValueError("conditioning_feature_dtype must be float16 or float32.") from exc
            feature_rows = []
            feature_width = None
            for sample, token_ids in zip(samples, conditioning_ids):
                features = torch.as_tensor(
                    sample["conditioning_features"],
                    dtype=declared_dtype,
                )
                if features.ndim != 2 or features.shape[0] != token_ids.numel():
                    raise ValueError(
                        "conditioning_features must have shape [conditioning_tokens, features]"
                    )
                if features.shape[1] <= 0:
                    raise ValueError("conditioning_features must have a positive feature width")
                if not torch.is_floating_point(features):
                    raise TypeError("conditioning_features must use a floating-point dtype")
                if not bool(torch.isfinite(features).all()):
                    raise ValueError("conditioning_features must contain only finite values")
                if feature_width is None:
                    feature_width = int(features.shape[1])
                elif features.shape[1] != feature_width:
                    raise ValueError(
                        "conditioning_features must use one shared feature width per batch"
                    )
                feature_rows.append(features)
            conditioning_features_padded = pad_sequence(
                feature_rows,
                batch_first=True,
                padding_value=0.0,
            )
        elif any("conditioning_feature_dtype" in sample for sample in samples):
            raise ValueError(
                "conditioning_feature_dtype cannot be present without conditioning_features."
            )

        target_language_ids = [
            _language_id_from_sample(
                sample,
                code_key="language",
                id_key="language_id",
                default=DEFAULT_LANGUAGE,
            )
            for sample in samples
        ]

        # Sequence length, rather than token value, defines validity. This also
        # keeps explicitly adapted legacy cl100k token ID zero distinguishable
        # from the compact v2 padding ID zero.
        conditioning_mask = torch.zeros_like(conditioning_ids_padded, dtype=torch.bool)
        token_language_values = []
        alignment_values = []
        for sample, ids, target_language_id in zip(
            samples,
            conditioning_ids,
            target_language_ids,
        ):
            conditioning_mask[len(token_language_values), : ids.numel()] = True
            token_languages, alignment = _parallel_text_fields(
                sample,
                ids,
                target_language_id=target_language_id,
            )
            token_language_values.append(token_languages)
            alignment_values.append(alignment)

        result = {
            "latents": latents_padded,
            "latent_mask": latent_mask,
            "conditioning_ids": conditioning_ids_padded,
            "conditioning_mask": conditioning_mask,
            "language_ids": torch.tensor(target_language_ids, dtype=torch.long),
            "token_language_ids": pad_sequence(
                token_language_values,
                batch_first=True,
                padding_value=0,
            ),
            "alignment_mask": pad_sequence(
                alignment_values,
                batch_first=True,
                padding_value=False,
            ),
        }
        if conditioning_features_padded is not None:
            result["conditioning_features"] = conditioning_features_padded

        # Speaker conditioning must be present consistently across the batch.
        samples_with_speaker = ["speaker_latents" in sample for sample in samples]
        if any(samples_with_speaker) and not all(samples_with_speaker):
            raise ValueError(
                "speaker_latents must be present in every sample or omitted from the entire batch"
            )

        if all(samples_with_speaker):
            speaker_latents_list = []
            speaker_frame_lengths = []
            for s in samples:
                speaker_latent = s["speaker_latents"]
                if isinstance(speaker_latent, np.ndarray):
                    speaker_latent = torch.from_numpy(speaker_latent).float()
                elif not isinstance(speaker_latent, torch.Tensor):
                    speaker_latent = torch.tensor(speaker_latent, dtype=torch.float32)
                if speaker_latent.ndim != 2:
                    raise ValueError("speaker_latents must have shape [channels, frames]")
                if speaker_latent.shape[1] == 0:
                    raise ValueError("speaker_latents must contain at least one frame")
                if not torch.isfinite(speaker_latent).all():
                    raise ValueError("speaker_latents must contain only finite values")
                speaker_frame_lengths.append(speaker_latent.shape[1])
                remainder = speaker_latent.shape[1] % self.speaker_patch_size
                if remainder:
                    speaker_latent = F.pad(
                        speaker_latent,
                        (0, self.speaker_patch_size - remainder),
                        value=self.latent_pad_value,
                    )
                speaker_latents_list.append(speaker_latent)  # [D, T_speaker]

            # Pad speaker latents along time dimension
            speaker_transposed = [
                latent.transpose(0, 1) for latent in speaker_latents_list
            ]  # [T, D]
            speaker_padded = pad_sequence(
                speaker_transposed, batch_first=True, padding_value=self.latent_pad_value
            )  # [B, T_max_speaker, D]
            speaker_padded = speaker_padded.transpose(1, 2)  # [B, D, T_max_speaker]

            # Create the patch-level mask consumed by the speaker encoder.
            max_speaker_patches = speaker_padded.shape[2] // self.speaker_patch_size
            speaker_mask = torch.zeros(batch_size, max_speaker_patches, dtype=torch.bool)
            for i, frame_length in enumerate(speaker_frame_lengths):
                valid_patches = (frame_length + self.speaker_patch_size - 1) // (
                    self.speaker_patch_size
                )
                speaker_mask[i, :valid_patches] = True

            result["speaker_latents"] = speaker_padded
            result["speaker_mask"] = speaker_mask
            result["speaker_language_ids"] = torch.tensor(
                [
                    _language_id_from_sample(
                        sample,
                        code_key="speaker_language",
                        id_key="speaker_language_id",
                        default=language_from_id(target_language_id).code,
                    )
                    for sample, target_language_id in zip(samples, target_language_ids)
                ],
                dtype=torch.long,
            )

            if any("reference_segment_lengths" in sample for sample in samples):
                if not all("reference_segment_lengths" in sample for sample in samples):
                    raise ValueError(
                        "reference_segment_lengths must be present for every speaker reference"
                    )
                segment_lengths = []
                for sample, frame_length in zip(samples, speaker_frame_lengths):
                    lengths = torch.as_tensor(
                        sample["reference_segment_lengths"],
                        dtype=torch.long,
                    )
                    if lengths.ndim != 1 or lengths.numel() == 0 or bool((lengths <= 0).any()):
                        raise ValueError(
                            "reference_segment_lengths must contain positive frame counts"
                        )
                    if int(lengths.sum()) != frame_length:
                        raise ValueError(
                            "reference_segment_lengths must sum to the speaker latent frames"
                        )
                    segment_lengths.append(lengths)
                result["speaker_segment_lengths"] = pad_sequence(
                    segment_lengths,
                    batch_first=True,
                    padding_value=0,
                )
                result["speaker_segment_mask"] = result["speaker_segment_lengths"] > 0

        return result


def create_data_collator(
    pad_token: int = PAD_TOKEN,
    speaker_patch_size: int = 4,
) -> FlowMatchingDataCollator:
    """
    Factory function to create data collator.

    Args:
        pad_token: Padding token ID
        speaker_patch_size: Number of adjacent speaker frames per encoder patch

    Returns:
        FlowMatchingDataCollator instance
    """
    return FlowMatchingDataCollator(
        pad_token=pad_token,
        speaker_patch_size=speaker_patch_size,
    )


class SimpleTTSCollator:
    """
    Simple collator for TTS-only data.

    Args:
        pad_token: Padding token ID
    """

    def __init__(
        self,
        pad_token: int = PAD_TOKEN,
        speaker_patch_size: int = 4,
    ):
        if speaker_patch_size <= 0:
            raise ValueError("speaker_patch_size must be greater than zero")
        self.pad_token = pad_token
        self.speaker_patch_size = speaker_patch_size

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate TTS features."""
        collator = FlowMatchingDataCollator(
            pad_token=self.pad_token,
            speaker_patch_size=self.speaker_patch_size,
        )
        return collator(features)
