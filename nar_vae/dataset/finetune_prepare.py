import hashlib
import json
import os
import shutil
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from tqdm import tqdm

from nar_vae.dacvae import HubDACVAESource, load_dacvae, normalize_dacvae_source
from nar_vae.dacvae_encoding import (
    DACVAEEncodingError,
    derive_dacvae_posterior_seed,
    encode_dacvae_posterior_seeded,
)
from nar_vae.dataset.identity import write_prepared_dataset_manifest
from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_COLUMN,
    RepresentationContractError,
    attach_representation_contract,
    build_representation_contract,
)
from nar_vae.dataset.sampling import LATENT_NUM_FRAMES_COLUMN, infer_latent_num_frames
from nar_vae.dataset.sources import (
    DEFAULT_DATASET_DOWNLOAD_WORKERS,
    resolve_dataset_source,
)
from nar_vae.distributed import (
    cleanup_distributed as cleanup_process_group,
)
from nar_vae.distributed import initialize_distributed, shard_indices
from nar_vae.frozen_text_provider import (
    FROZEN_TEXT_REPRESENTATION_NAME,
    FROZEN_TEXT_REPRESENTATION_VERSION,
    FrozenTextConditioning,
    FrozenTextProvider,
    resolve_frozen_text_provider,
)
from nar_vae.languages import DEFAULT_LANGUAGE, normalize_language
from nar_vae.tokenization import TextConditioning, encode_tts_conditioning, encode_tts_text
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS

from .utterance_store import (
    AUDIO_SHA256_COLUMN,
    CONDITIONING_NUM_TOKENS_COLUMN,
    SPEAKER_NUM_FRAMES_COLUMN,
    attach_utterance_metadata,
    canonical_audio_sha256,
)

# Suppress audio decoding warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning)

GRPO_PROMPT_ID_COLUMN = "utterance_id"


def _content_bound_utterance_id(row: dict) -> str:
    """Derive a stable prompt ID from the prepared target rather than mutable row order."""
    latents = np.ascontiguousarray(row["latents"])
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "conditioning_ids": row["conditioning_ids"],
                "language": row.get("language"),
                "text": row.get("text"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    digest.update(str(latents.dtype).encode("ascii"))
    digest.update(json.dumps(list(latents.shape), separators=(",", ":")).encode("ascii"))
    digest.update(latents.view(np.uint8).tobytes())
    return f"narvae-{digest.hexdigest()}"


def _validate_unique_prompt_ids(rows) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        value = row.get(GRPO_PROMPT_ID_COLUMN)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Prepared row {index} has an invalid immutable utterance_id.")
        if value in seen:
            raise ValueError(f"Prepared utterance_id values must be unique; duplicate {value!r}.")
        seen.add(value)


def setup_distributed():
    """Select this torchrun process's GPU before loading DACVAE."""
    process = initialize_distributed()
    return process.local_rank, process.world_size, process.is_distributed


def cleanup_distributed():
    """Release the data workers' process group after shard merging."""
    cleanup_process_group(barrier=False)


class DataPreparer:
    """Prepares dataset for EchoDiT fine-tuning."""

    def __init__(
        self,
        dacvae_model: str | os.PathLike[str] | HubDACVAESource | None = None,
        device: str = "cuda",
        dacvae_backend: str = "bundled",
        max_reference_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
        language: str = DEFAULT_LANGUAGE,
        dacvae_revision: str | None = None,
        dacvae_filename: str | None = None,
        dacvae_sha256: str | None = None,
        frozen_text_provider: FrozenTextProvider | None = None,
    ):
        self.device = device
        if max_reference_seconds <= 0:
            raise ValueError("max_reference_seconds must be positive")
        if dacvae_model is None:
            raise ValueError(
                "dacvae_model is required; use a local path or a pinned Hugging Face ID."
            )

        codec_source = normalize_dacvae_source(
            dacvae_model,
            dacvae_revision=dacvae_revision,
            dacvae_filename=dacvae_filename,
        )
        self.dacvae = load_dacvae(
            codec_source,
            backend=dacvae_backend,
            device=device,
            freeze=True,
            expected_sha256=dacvae_sha256,
        )
        self.representation_contract = build_representation_contract(
            self.dacvae,
            codec_source=codec_source,
            **(
                {
                    "text_frontend_name": FROZEN_TEXT_REPRESENTATION_NAME,
                    "text_frontend_version": FROZEN_TEXT_REPRESENTATION_VERSION,
                }
                if frozen_text_provider is not None
                else {}
            ),
        )

        self.sample_rate = self.dacvae.sample_rate  # 48000
        self.max_reference_samples = int(max_reference_seconds * self.sample_rate)
        self.language = normalize_language(language)
        self.frozen_text_provider = frozen_text_provider

    @torch.no_grad()
    def extract_latents(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract DACVAE latents from audio."""
        # Convert to tensor
        waveform = torch.from_numpy(audio).float()

        # Ensure mono
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        seed = derive_dacvae_posterior_seed(
            waveform,
            codec_sha256=self.representation_contract.codec_sha256,
        )
        # Add batch dimension and move to device
        waveform = waveform.unsqueeze(0).to(self.device)  # [1, 1, T]

        # Encode with DACVAE
        latents = encode_dacvae_posterior_seeded(
            self.dacvae,
            waveform,
            seed=seed,
        )  # [1, D, T_latent]

        return latents.squeeze(0).cpu().numpy()  # [D, T_latent]

    def tokenize_text(self, text: str, language: str | None = None) -> list:
        """Tokenize text and add special tokens for TTS."""
        if getattr(self, "frozen_text_provider", None) is not None:
            raise RuntimeError(
                "Frozen-feature preparation must use encode_text_conditioning(), not the "
                "compact scratch tokenizer."
            )
        return encode_tts_text(text, language=language or self.language)

    def encode_text_conditioning(
        self,
        text: str,
        *,
        language: str | None = None,
        normalized_text: str | None = None,
        phonemes: str | list[str] | None = None,
        language_spans: list[dict] | None = None,
    ) -> TextConditioning | FrozenTextConditioning:
        provider = getattr(self, "frozen_text_provider", None)
        if provider is not None:
            return provider.encode(
                text,
                normalized_text=normalized_text,
                phonemes=phonemes,
                language=language or self.language,
                language_spans=language_spans,
            )
        return encode_tts_conditioning(
            text,
            normalized_text=normalized_text,
            phonemes=phonemes,
            language=language or self.language,
            language_spans=language_spans,
        )

    @torch.no_grad()
    def extract_speaker_latents(
        self,
        references: list[tuple[np.ndarray, int]],
    ) -> np.ndarray | None:
        """Encode bounded same-speaker utterances as the voice reference."""
        waveforms = []
        remaining = self.max_reference_samples
        for audio, sample_rate in references:
            waveform = torch.as_tensor(np.asarray(audio)).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.ndim != 2 or waveform.shape[-1] == 0 or sample_rate <= 0:
                continue
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    self.sample_rate,
                )
            waveform = waveform[..., :remaining]
            if waveform.shape[-1]:
                waveforms.append(waveform)
                remaining -= waveform.shape[-1]
            if remaining <= 0:
                break
        if not waveforms:
            return None
        waveform = torch.cat(waveforms, dim=-1)
        seed = derive_dacvae_posterior_seed(
            waveform,
            codec_sha256=self.representation_contract.codec_sha256,
        )
        waveform = waveform.unsqueeze(0).to(self.device)
        return (
            encode_dacvae_posterior_seeded(self.dacvae, waveform, seed=seed)
            .squeeze(0)
            .cpu()
            .numpy()
        )

    def process_example(
        self,
        example: dict,
        speaker_references: list[tuple[np.ndarray, int]] | None = None,
        audio_column: str = "audio",
        language: str | None = None,
        speaker_language: str | None = None,
        dataset_namespace: str = "dataset",
        speaker_id=None,
        utterance_id=None,
        normalized_text: str | None = None,
        phonemes: str | list[str] | None = None,
        language_spans: list[dict] | None = None,
    ) -> dict:
        """Process a single example."""
        target_language = normalize_language(language or self.language)
        provider = getattr(self, "frozen_text_provider", None)
        if (
            provider is not None
            and provider.spec.frozen_text_frontend == "phonemes"
            and phonemes is None
            and not language_spans
        ):
            raise ValueError(
                "Frozen phoneme preparation requires canonical phonemes or "
                "phoneme-bearing language_spans for every row."
            )
        del speaker_language
        if speaker_references is not None:
            raise ValueError(
                "Static speaker references are not part of prepared-row v2; use "
                "DynamicReferenceDataset over cached utterance latents."
            )
        try:
            # Get audio
            audio_data = example.get(audio_column, {})
            if isinstance(audio_data, dict):
                audio = np.array(audio_data.get("array", []))
                sr = audio_data.get("sampling_rate", self.sample_rate)
            else:
                return None

            if len(audio) == 0:
                return None

            # Get text
            text = (
                example.get("text", "")
                or example.get("transcript", "")
                or example.get("sentence", "")
            )
            if not text and normalized_text is None and phonemes is None and language_spans is None:
                return None

            # Extract latents
            latents = self.extract_latents(audio, sr)

            # Tokenize text
            conditioning = self.encode_text_conditioning(
                text,
                language=target_language,
                normalized_text=normalized_text,
                phonemes=phonemes,
                language_spans=language_spans,
            )
            audio_hash = canonical_audio_sha256(audio, sr)

            result = {
                "latents": latents,
                LATENT_NUM_FRAMES_COLUMN: infer_latent_num_frames(latents),
                "conditioning_ids": (
                    conditioning.conditioning_ids.tolist()
                    if isinstance(conditioning.conditioning_ids, torch.Tensor)
                    else conditioning.conditioning_ids
                ),
                "token_language_ids": (
                    conditioning.token_language_ids.tolist()
                    if isinstance(conditioning.token_language_ids, torch.Tensor)
                    else conditioning.token_language_ids
                ),
                "alignment_mask": (
                    conditioning.alignment_mask.tolist()
                    if isinstance(conditioning.alignment_mask, torch.Tensor)
                    else conditioning.alignment_mask
                ),
                CONDITIONING_NUM_TOKENS_COLUMN: len(conditioning.conditioning_ids),
                SPEAKER_NUM_FRAMES_COLUMN: infer_latent_num_frames(latents),
                AUDIO_SHA256_COLUMN: audio_hash,
                "text": text,
                "language": target_language,
            }
            if isinstance(conditioning, FrozenTextConditioning):
                result.update(conditioning.to_cache_row())
            if speaker_id is not None:
                attach_utterance_metadata(
                    result,
                    dataset_namespace=dataset_namespace,
                    speaker_id=speaker_id,
                    utterance_id=utterance_id,
                    audio_sha256=audio_hash,
                    text=text,
                )
            else:
                result[GRPO_PROMPT_ID_COLUMN] = _content_bound_utterance_id(result)
            return attach_representation_contract(result, self.representation_contract)
        except (RepresentationContractError, DACVAEEncodingError):
            raise
        except Exception as e:
            if provider is not None:
                raise
            print(f"Error processing example: {e}")
            return None


def prepare_finetune_dataset(
    dataset_name: str,
    output_dir: str = "finetune_data",
    *,
    split: str = "train",
    max_samples: int | None = None,
    dacvae_model: str | os.PathLike[str] | HubDACVAESource | None = None,
    dacvae_backend: str = "bundled",
    speaker_id_column: str | None = None,
    audio_column: str = "audio",
    max_reference_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
    max_reference_utterances: int = 3,
    reference_seed: int = 1234,
    language: str = DEFAULT_LANGUAGE,
    language_column: str | None = None,
    utterance_id_column: str | None = None,
    normalized_text_column: str | None = None,
    phonemes_column: str | None = None,
    language_spans_column: str | None = None,
    dataset_namespace: str | None = None,
    dataset_revision: str | None = None,
    dataset_download_workers: int = DEFAULT_DATASET_DOWNLOAD_WORKERS,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
    dacvae_sha256: str | None = None,
    frozen_text_provider: FrozenTextProvider | None = None,
    frozen_text_config: Mapping[str, Any] | None = None,
) -> None:
    """Prepare a Hugging Face dataset for EchoDiT fine-tuning.

    Distributed execution is enabled automatically when ``LOCAL_RANK`` is set.
    Speaker-conditioned rows use a different utterance from the same speaker.
    """

    source = resolve_dataset_source(
        dataset_name,
        revision=dataset_revision,
        download_workers=dataset_download_workers,
    )

    # Setup distributed
    local_rank, world_size, is_distributed = setup_distributed()
    rank = dist.get_rank() if is_distributed else 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    frozen_text_provider = resolve_frozen_text_provider(
        provider=frozen_text_provider,
        config=frozen_text_config,
        device=device,
    )

    is_main = rank == 0

    if is_main:
        print("=" * 80)
        print("EchoDiT Fine-tuning Data Preparation")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Output: {output_dir}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print("=" * 80)

    # Load dataset
    if is_main:
        print(f"\nLoading dataset: {dataset_name}")

    ds = load_dataset(
        source.location,
        split=split,
        **source.load_dataset_kwargs(),
    )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    del max_reference_utterances, reference_seed
    dataset_namespace = dataset_namespace or dataset_name
    for optional_column in (
        speaker_id_column,
        normalized_text_column,
        phonemes_column,
        language_spans_column,
    ):
        if optional_column is not None and optional_column not in ds.column_names:
            raise ValueError(f"Dataset column not found: {optional_column!r}")
    if (
        frozen_text_provider is not None
        and frozen_text_provider.spec.frozen_text_frontend == "phonemes"
        and phonemes_column is None
        and language_spans_column is None
    ):
        raise ValueError(
            "Frozen phoneme dataset preparation requires phonemes_column or "
            "language_spans_column; raw transcript fallback is forbidden."
        )

    if utterance_id_column is not None:
        if utterance_id_column not in ds.column_names:
            raise ValueError(f"Utterance ID column not found: {utterance_id_column!r}")
        source_ids = ds[utterance_id_column]
        if any(not isinstance(value, str) or not value.strip() for value in source_ids):
            raise ValueError("The configured utterance ID column must contain non-empty strings.")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("The configured utterance ID column must be globally unique.")

    if is_main:
        print(f"Dataset size: {len(ds)} samples")

    # Initialize preparer
    preparer = DataPreparer(
        dacvae_model=dacvae_model,
        dacvae_backend=dacvae_backend,
        device=device,
        max_reference_seconds=max_reference_seconds,
        language=language,
        dacvae_revision=dacvae_revision,
        dacvae_filename=dacvae_filename,
        dacvae_sha256=dacvae_sha256,
        frozen_text_provider=frozen_text_provider,
    )

    # Split dataset for distributed processing
    if is_distributed:
        indices = list(shard_indices(len(ds), rank=rank, world_size=world_size))
        ds_shard = ds.select(indices)
        if is_main:
            print(f"Processing {len(ds_shard)} samples on each GPU")
    else:
        ds_shard = ds

    # Process examples
    processed_examples = []
    skipped_count = 0

    # Use index-based iteration to handle corrupted audio files
    iterator = tqdm(range(len(ds_shard)), desc=f"Rank {rank}", disable=not is_main)
    for i in iterator:
        try:
            example = ds_shard[i]
            target_language = normalize_language(
                example.get(language_column, language) if language_column else language
            )
            result = preparer.process_example(
                example,
                audio_column=audio_column,
                language=target_language,
                dataset_namespace=dataset_namespace,
                speaker_id=(
                    example.get(speaker_id_column) if speaker_id_column is not None else None
                ),
                utterance_id=(
                    example.get(utterance_id_column) if utterance_id_column is not None else None
                ),
                normalized_text=(
                    example.get(normalized_text_column)
                    if normalized_text_column is not None
                    else None
                ),
                phonemes=(example.get(phonemes_column) if phonemes_column is not None else None),
                language_spans=(
                    example.get(language_spans_column)
                    if language_spans_column is not None
                    else None
                ),
            )
            if result is not None:
                processed_examples.append(result)
            else:
                skipped_count += 1
        except (RepresentationContractError, DACVAEEncodingError):
            raise
        except Exception as e:
            if frozen_text_provider is not None:
                raise RuntimeError(f"Frozen text preparation failed for dataset row {i}.") from e
            # Skip corrupted/unreadable audio files
            skipped_count += 1
            if skipped_count <= 10:  # Only print first 10 errors
                print(f"\n[Rank {rank}] Skipping sample {i}: {type(e).__name__}: {str(e)[:100]}")
            elif skipped_count == 11:
                print(f"\n[GPU {local_rank}] Suppressing further error messages...")
            continue

    if is_main:
        print(f"\nSkipped {skipped_count} corrupted/invalid samples")

    print(f"\n[Rank {rank}] Processed {len(processed_examples)} examples, skipped {skipped_count}")

    # Gather results from all GPUs
    if is_distributed:
        # Save shard to temp file
        shard_path = f"{output_dir}_shard_{rank}"
        shard_ds = Dataset.from_list(processed_examples)
        shard_ds.save_to_disk(shard_path)

        dist.barrier()

        # Main process combines all shards
        if is_main:
            print("\nCombining shards from all GPUs...")
            all_datasets = []
            for rank in range(world_size):
                shard_path = f"{output_dir}_shard_{rank}"
                shard_ds = load_from_disk(shard_path)
                all_datasets.append(shard_ds)
                print(f"  Loaded shard {rank}: {len(shard_ds)} samples")

            # Concatenate
            final_ds = concatenate_datasets(all_datasets)
            _validate_unique_prompt_ids(final_ds)
            print(f"\nTotal samples: {len(final_ds)}")

            # Save final dataset
            print(f"Saving to {output_dir}...")
            final_ds.save_to_disk(output_dir)
            write_prepared_dataset_manifest(final_ds, output_dir)

            # Cleanup temp files
            for rank in range(world_size):
                shard_path = f"{output_dir}_shard_{rank}"
                if os.path.exists(shard_path):
                    shutil.rmtree(shard_path)

            print(f"\nDataset saved to: {output_dir}")
    else:
        # Single GPU - save directly
        final_ds = Dataset.from_list(processed_examples)
        _validate_unique_prompt_ids(final_ds)
        print(f"\nTotal samples: {len(final_ds)}")
        print(f"Saving to {output_dir}...")
        final_ds.save_to_disk(output_dir)
        write_prepared_dataset_manifest(final_ds, output_dir)
        print(f"Dataset saved to: {output_dir}")

    # Cleanup
    cleanup_distributed()

    if is_main:
        print("\n" + "=" * 80)
        print("Data preparation complete!")
        print("=" * 80)
        print("\nThe dataset is ready for EchoDiTFineTuner.")
        print(f"Rows include {REPRESENTATION_CONTRACT_COLUMN} metadata.")
