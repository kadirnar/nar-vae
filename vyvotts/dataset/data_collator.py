from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from vyvotts.languages import DEFAULT_LANGUAGE, language_from_id, language_id
from vyvotts.tokenization import PAD_TOKEN


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
                "NAR-VAE accepts TTS latent rows only; text-QA/OPT mixing support was removed. "
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
        conditioning_ids_padded = pad_sequence(
            conditioning_ids, batch_first=True, padding_value=self.pad_token
        )

        # Create attention mask for conditioning
        conditioning_mask = (conditioning_ids_padded != self.pad_token).long()

        target_language_ids = [
            _language_id_from_sample(
                sample,
                code_key="language",
                id_key="language_id",
                default=DEFAULT_LANGUAGE,
            )
            for sample in samples
        ]
        result = {
            "latents": latents_padded,
            "latent_mask": latent_mask,
            "conditioning_ids": conditioning_ids_padded,
            "conditioning_mask": conditioning_mask,
            "language_ids": torch.tensor(target_language_ids, dtype=torch.long),
        }

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
