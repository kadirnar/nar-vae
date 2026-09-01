import os
import shutil
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchaudio.transforms as T
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
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
from nar_vae.tokenization import (
    PAD_TOKEN,
    START_OF_TEXT,
    TOKENIZER_LENGTH,
    TextConditioning,
    encode_tts_conditioning,
    encode_tts_text,
)
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS

from .utterance_store import (
    AUDIO_SHA256_COLUMN,
    CONDITIONING_NUM_TOKENS_COLUMN,
    SPEAKER_NUM_FRAMES_COLUMN,
    attach_utterance_metadata,
    canonical_audio_sha256,
    stable_utterance_id,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_distributed():
    """Select this torchrun process's GPU before loading DACVAE."""
    process = initialize_distributed()
    return process.local_rank, process.world_size, process.is_distributed


def cleanup_distributed():
    """Synchronize all data workers and release their process group."""
    cleanup_process_group()


class DatasetPreparer:
    """Prepares standard text/audio datasets for NAR-VAE training."""

    def __init__(
        self,
        dacvae_model: str | os.PathLike[str] | HubDACVAESource,
        device: str,
        max_speaker_ref_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
        dacvae_backend: str = "bundled",
        language: str = DEFAULT_LANGUAGE,
        dacvae_revision: str | None = None,
        dacvae_filename: str | None = None,
        dacvae_sha256: str | None = None,
        frozen_text_provider: FrozenTextProvider | None = None,
    ):
        self.device = device
        self.max_speaker_ref_seconds = max_speaker_ref_seconds

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
        self.sample_rate = self.dacvae.sample_rate
        self.max_speaker_ref_samples = int(max_speaker_ref_seconds * self.sample_rate)
        self.language = normalize_language(language)
        self.frozen_text_provider = frozen_text_provider

    @torch.no_grad()
    def extract_latents(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract DACVAE latents from audio."""
        waveform = torch.from_numpy(np.array(audio)).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        seed = derive_dacvae_posterior_seed(
            waveform,
            codec_sha256=self.representation_contract.codec_sha256,
        )
        waveform = waveform.unsqueeze(0).to(self.device)
        latents = encode_dacvae_posterior_seeded(self.dacvae, waveform, seed=seed)

        return latents.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_speaker_latents(self, audio_list: list) -> np.ndarray:
        """
        Extract speaker reference latents from multiple audio samples.

        Concatenates audio samples up to max_speaker_ref_seconds and extracts latents.

        Args:
            audio_list: List of (audio_array, sample_rate) tuples

        Returns:
            Speaker reference latents [D, T_speaker]
        """
        concatenated_audio = []
        total_samples = 0

        for audio, sr in audio_list:
            waveform = torch.from_numpy(np.array(audio)).float()

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)

            # Check if adding this would exceed max duration
            remaining = self.max_speaker_ref_samples - total_samples
            if remaining <= 0:
                break

            if waveform.shape[1] > remaining:
                waveform = waveform[:, :remaining]

            concatenated_audio.append(waveform)
            total_samples += waveform.shape[1]

        if not concatenated_audio:
            return None

        # Concatenate all audio
        speaker_audio = torch.cat(concatenated_audio, dim=1)
        seed = derive_dacvae_posterior_seed(
            speaker_audio,
            codec_sha256=self.representation_contract.codec_sha256,
        )
        speaker_audio = speaker_audio.unsqueeze(0).to(self.device)

        latents = encode_dacvae_posterior_seeded(self.dacvae, speaker_audio, seed=seed)
        return latents.squeeze(0).cpu().numpy()

    def tokenize_text(self, text: str, language: str | None = None) -> list:
        """
        Tokenize text with special tokens and emotion tags.

        Emotion tags like <laugh>, <sigh>, <cry> are converted to special tokens
        instead of being encoded as regular text. This allows the model to learn
        to produce the corresponding sounds during fine-tuning.

        Example:
            "Hello <laugh> how are you?" ->
            [START_OF_HUMAN, ..hello tokens.., LAUGH_TOKEN, ..how are you tokens.., END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]
        """
        if getattr(self, "frozen_text_provider", None) is not None:
            raise RuntimeError(
                "Frozen-feature preparation requires encode_text_conditioning() with the "
                "provider's canonical input, not compact tokenize_text()."
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

    def process_sample(
        self,
        sample: dict,
        speaker_ref_audio: list = None,
        *,
        language: str | None = None,
        speaker_language: str | None = None,
        dataset_namespace: str = "dataset",
        speaker_id=None,
        utterance_id=None,
        normalized_text: str | None = None,
        phonemes: str | list[str] | None = None,
        language_spans: list[dict] | None = None,
    ) -> dict:
        """
        Process a single sample with standard text/audio format.

        Expected format:
        - sample["audio"] = {"array": ndarray, "sampling_rate": int}
        - sample["text"] or sample["transcript"] or sample["sentence"] = str

        Args:
            sample: Dataset sample
            speaker_ref_audio: List of (audio_array, sample_rate) tuples for speaker reference
        """
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
        if speaker_ref_audio is not None:
            raise ValueError(
                "Prepared-row v2 stores one latent per utterance. Use speaker_id metadata "
                "and DynamicReferenceDataset instead of static speaker_ref_audio."
            )
        try:
            audio_data = sample.get("audio")
            if audio_data is None or not isinstance(audio_data, dict):
                return None

            if "array" not in audio_data:
                return None

            audio = np.array(audio_data["array"])
            sr = audio_data.get("sampling_rate", self.sample_rate)

            if len(audio) == 0:
                return None

            text = sample.get("text") or sample.get("transcript") or sample.get("sentence") or ""

            if not text and normalized_text is None and phonemes is None and language_spans is None:
                return None

            latents = self.extract_latents(audio, sr)
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
                result["utterance_id"] = stable_utterance_id(
                    dataset_namespace,
                    source_id=utterance_id,
                    audio_sha256=audio_hash,
                    language=target_language,
                    text=text,
                )
            return attach_representation_contract(result, self.representation_contract)
        except (RepresentationContractError, DACVAEEncodingError):
            raise
        except Exception:
            if provider is not None:
                raise
            return None


def prepare_dataset(
    dataset_name: str,
    output_dir: str,
    split: str,
    dacvae_model: str | os.PathLike[str] | HubDACVAESource,
    max_samples: int,
    push_to_hub: str,
    use_speaker_id: bool = False,
    speaker_id_column: str = "speaker_id",
    max_speaker_ref_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
    dacvae_backend: str = "bundled",
    language: str = DEFAULT_LANGUAGE,
    language_column: str | None = None,
    utterance_id_column: str | None = None,
    normalized_text_column: str | None = None,
    phonemes_column: str | None = None,
    language_spans_column: str | None = None,
    dataset_namespace: str | None = None,
    max_reference_utterances: int = 5,
    reference_seed: int = 1234,
    dataset_revision: str | None = None,
    dataset_download_workers: int = DEFAULT_DATASET_DOWNLOAD_WORKERS,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
    dacvae_sha256: str | None = None,
    frozen_text_provider: FrozenTextProvider | None = None,
    frozen_text_config: Mapping[str, Any] | None = None,
):
    """Prepare a HuggingFace dataset with distributed processing support."""
    source = resolve_dataset_source(
        dataset_name,
        revision=dataset_revision,
        download_workers=dataset_download_workers,
    )
    local_rank, world_size, is_distributed = setup_distributed()
    rank = dist.get_rank() if is_distributed else 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0
    frozen_text_provider = resolve_frozen_text_provider(
        provider=frozen_text_provider,
        config=frozen_text_config,
        device=device,
    )

    if is_main:
        print("=" * 80)
        print("NAR-VAE Dataset Preparation")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Output: {output_dir}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Speaker metadata for reference pairing: {use_speaker_id}")
        print("=" * 80)

        if not source.is_local:
            snapshot_download(
                repo_id=source.location,
                repo_type="dataset",
                revision=source.revision,
                max_workers=source.download_workers,
            )

    if is_distributed:
        dist.barrier()

    ds = load_dataset(
        source.location,
        split=split,
        **source.load_dataset_kwargs(),
    )

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    del max_reference_utterances, reference_seed
    dataset_namespace = dataset_namespace or dataset_name
    required_metadata_columns = [speaker_id_column] if use_speaker_id else []
    required_metadata_columns.extend(
        column
        for column in (
            utterance_id_column,
            normalized_text_column,
            phonemes_column,
            language_spans_column,
        )
        if column is not None
    )
    for column in required_metadata_columns:
        if column not in ds.column_names:
            raise ValueError(f"Dataset column not found: {column!r}")
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

    if is_main:
        print(f"Dataset size: {len(ds)} samples")

    preparer = DatasetPreparer(
        dacvae_model=dacvae_model,
        device=device,
        max_speaker_ref_seconds=max_speaker_ref_seconds,
        dacvae_backend=dacvae_backend,
        language=language,
        dacvae_revision=dacvae_revision,
        dacvae_filename=dacvae_filename,
        dacvae_sha256=dacvae_sha256,
        frozen_text_provider=frozen_text_provider,
    )

    if is_main:
        print(f"DACVAE sample rate: {preparer.sample_rate}")

    if is_distributed:
        indices = list(shard_indices(len(ds), rank=rank, world_size=world_size))
        ds_shard = ds.select(indices)
    else:
        ds_shard = ds

    processed_examples = []
    skipped_count = 0

    iterator = tqdm(range(len(ds_shard)), desc=f"Rank {rank}", disable=not is_main)
    for i in iterator:
        try:
            example = ds_shard[i]
            target_language = normalize_language(
                example.get(language_column, language) if language_column else language
            )

            result = preparer.process_sample(
                example,
                language=target_language,
                dataset_namespace=dataset_namespace,
                speaker_id=(example.get(speaker_id_column) if use_speaker_id else None),
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
        except Exception as exc:
            if frozen_text_provider is not None:
                raise RuntimeError(f"Frozen text preparation failed for dataset row {i}.") from exc
            skipped_count += 1
            continue

    print(f"[Rank {rank}] Processed {len(processed_examples)}, skipped {skipped_count}")

    if is_distributed:
        shard_path = f"{output_dir}_shard_{rank}"
        shard_ds = Dataset.from_list(processed_examples)
        shard_ds.save_to_disk(shard_path)

        dist.barrier()

        if is_main:
            print("\nCombining shards from all GPUs...")
            all_datasets = []
            for rank in range(world_size):
                shard_path = f"{output_dir}_shard_{rank}"
                shard_ds = load_from_disk(shard_path)
                all_datasets.append(shard_ds)

            final_ds = concatenate_datasets(all_datasets)
            print(f"Total samples: {len(final_ds)}")

            os.makedirs(output_dir, exist_ok=True)
            final_ds.save_to_disk(output_dir)
            write_prepared_dataset_manifest(final_ds, output_dir)

            for rank in range(world_size):
                shutil.rmtree(f"{output_dir}_shard_{rank}", ignore_errors=True)

            if push_to_hub:
                print(f"Pushing to {push_to_hub}...")
                final_ds.push_to_hub(push_to_hub, private=True)
    else:
        final_ds = Dataset.from_list(processed_examples)
        print(f"Total samples: {len(final_ds)}")

        os.makedirs(output_dir, exist_ok=True)
        final_ds.save_to_disk(output_dir)
        write_prepared_dataset_manifest(final_ds, output_dir)

        if push_to_hub:
            print(f"Pushing to {push_to_hub}...")
            final_ds.push_to_hub(push_to_hub, private=True)

    if is_main:
        print(f"\nDataset saved to: {output_dir}")
        print("\nDataset columns:")
        print("  - latents: DACVAE encoded audio")
        print("  - latent_num_frames: Persisted DACVAE frame count for efficient batching")
        print("  - conditioning_ids: Tokenized text with special tokens")
        print(f"  - {REPRESENTATION_CONTRACT_COLUMN}: Versioned text/codec representation")
        if use_speaker_id:
            print("  - speaker_id: namespaced pairing/split metadata (never a model input)")
        print("\nTokenizer info:")
        if frozen_text_provider is not None:
            print(
                "  Tokenizer: pinned frozen provider "
                f"({frozen_text_provider.spec.frozen_text_tokenizer_id})"
            )
            print(f"  Vocab size: {frozen_text_provider.vocab_size}")
            print(f"  Padding token: {frozen_text_provider.pad_token_id}")
            print(f"  Provider contract SHA-256: {frozen_text_provider.spec.contract_sha256}")
        else:
            print("  Tokenizer: compact hybrid IPA/grapheme/UTF-8")
            print(f"  Vocab size: {TOKENIZER_LENGTH}")
            print(f"  Padding token: {PAD_TOKEN}; start token: {START_OF_TEXT}")

    cleanup_distributed()
