import os
import shutil
import warnings
from collections import defaultdict

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torchaudio.transforms as T
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from tqdm import tqdm

from nar_vae.dacvae import HubDACVAESource, load_dacvae
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
from nar_vae.dataset.speaker_references import (
    collect_reference_audio_with_language,
    select_reference_indices,
)
from nar_vae.distributed import (
    cleanup_distributed as cleanup_process_group,
)
from nar_vae.distributed import initialize_distributed, shard_indices
from nar_vae.languages import DEFAULT_LANGUAGE, normalize_language
from nar_vae.tokenization import (
    PAD_TOKEN,
    START_OF_TEXT,
    TOKENIZER_LENGTH,
    encode_tts_text,
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
        max_speaker_ref_seconds: int = 120,
        dacvae_backend: str = "bundled",
        language: str = DEFAULT_LANGUAGE,
    ):
        self.device = device
        self.max_speaker_ref_seconds = max_speaker_ref_seconds

        self.dacvae = load_dacvae(
            dacvae_model,
            backend=dacvae_backend,
            device=device,
            freeze=True,
        )
        self.representation_contract = build_representation_contract(
            self.dacvae,
            codec_source=dacvae_model,
        )
        self.sample_rate = self.dacvae.sample_rate
        self.max_speaker_ref_samples = max_speaker_ref_seconds * self.sample_rate
        self.language = normalize_language(language)

        self.tokenizer = tiktoken.get_encoding("cl100k_base")

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

        waveform = waveform.unsqueeze(0).to(self.device)
        latents = self.dacvae.encode(waveform)

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
        speaker_audio = speaker_audio.unsqueeze(0).to(self.device)

        latents = self.dacvae.encode(speaker_audio)
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
        return encode_tts_text(text, self.tokenizer, language=language or self.language)

    def process_sample(
        self,
        sample: dict,
        speaker_ref_audio: list = None,
        *,
        language: str | None = None,
        speaker_language: str | None = None,
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
        reference_language = normalize_language(speaker_language or target_language)
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

            if not text:
                return None

            latents = self.extract_latents(audio, sr)
            conditioning_ids = self.tokenize_text(text, target_language)

            result = {
                "latents": latents,
                LATENT_NUM_FRAMES_COLUMN: infer_latent_num_frames(latents),
                "conditioning_ids": conditioning_ids,
                "language": target_language,
            }

            # Extract speaker reference latents if provided
            if speaker_ref_audio:
                speaker_latents = self.extract_speaker_latents(speaker_ref_audio)
                if speaker_latents is not None:
                    result["speaker_latents"] = speaker_latents
                    result["speaker_language"] = reference_language

            return attach_representation_contract(result, self.representation_contract)
        except RepresentationContractError:
            raise
        except Exception:
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
    max_speaker_ref_seconds: int = 120,
    dacvae_backend: str = "bundled",
    language: str = DEFAULT_LANGUAGE,
    language_column: str | None = None,
    max_reference_utterances: int = 5,
    reference_seed: int = 1234,
    dataset_revision: str | None = None,
    dataset_download_workers: int = DEFAULT_DATASET_DOWNLOAD_WORKERS,
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

    if is_main:
        print("=" * 80)
        print("NAR-VAE Dataset Preparation")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Output: {output_dir}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Speaker-ID training: {use_speaker_id}")
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

    if is_main:
        print(f"Dataset size: {len(ds)} samples")

    preparer = DatasetPreparer(
        dacvae_model=dacvae_model,
        device=device,
        max_speaker_ref_seconds=max_speaker_ref_seconds,
        dacvae_backend=dacvae_backend,
        language=language,
    )

    if is_main:
        print(f"DACVAE sample rate: {preparer.sample_rate}")

    # Build speaker index if using speaker-ID training
    speaker_to_indices = None
    if use_speaker_id:
        if is_main:
            print(f"\nBuilding speaker index from '{speaker_id_column}' column...")

        speaker_to_indices = defaultdict(list)
        for idx in range(len(ds)):
            speaker_id = ds[idx].get(speaker_id_column)
            if speaker_id is not None:
                speaker_to_indices[speaker_id].append(idx)

        if is_main:
            print(f"Found {len(speaker_to_indices)} unique speakers")
            speaker_counts = [len(v) for v in speaker_to_indices.values()]
            print(
                f"Samples per speaker: min={min(speaker_counts)}, max={max(speaker_counts)}, avg={np.mean(speaker_counts):.1f}"
            )

    if is_distributed:
        indices = list(shard_indices(len(ds), rank=rank, world_size=world_size))
        ds_shard = ds.select(indices)
        # Map original indices for speaker reference lookup
        shard_to_original = {i: indices[i] for i in range(len(indices))}
    else:
        ds_shard = ds
        shard_to_original = {i: i for i in range(len(ds))}

    processed_examples = []
    skipped_count = 0

    iterator = tqdm(range(len(ds_shard)), desc=f"Rank {rank}", disable=not is_main)
    for i in iterator:
        try:
            example = ds_shard[i]
            original_idx = shard_to_original[i]
            target_language = normalize_language(
                example.get(language_column, language) if language_column else language
            )

            # Get speaker reference audio if using speaker-ID training
            speaker_ref_audio = None
            speaker_reference_language = target_language
            if use_speaker_id and speaker_to_indices is not None:
                speaker_id = example.get(speaker_id_column)
                if speaker_id is not None and speaker_id in speaker_to_indices:
                    ref_indices = select_reference_indices(
                        speaker_to_indices,
                        speaker_id=speaker_id,
                        target_index=original_idx,
                        maximum_utterances=max_reference_utterances,
                        seed=reference_seed,
                    )
                    speaker_ref_audio, speaker_reference_language = (
                        collect_reference_audio_with_language(
                            ds,
                            ref_indices,
                            audio_column="audio",
                            language_column=language_column,
                            fallback_language=target_language,
                        )
                    )

            result = preparer.process_sample(
                example,
                speaker_ref_audio,
                language=target_language,
                speaker_language=speaker_reference_language,
            )
            if result is not None:
                processed_examples.append(result)
            else:
                skipped_count += 1
        except RepresentationContractError:
            raise
        except Exception:
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
            print(
                f"  - speaker_latents: DACVAE encoded speaker reference (up to {max_speaker_ref_seconds}s)"
            )
        print("\nTokenizer info:")
        print("  Tokenizer: tiktoken cl100k_base")
        print(f"  Vocab size: {TOKENIZER_LENGTH}")
        print(f"  Special tokens: {START_OF_TEXT} to {PAD_TOKEN}")

    cleanup_distributed()
