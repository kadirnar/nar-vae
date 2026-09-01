import gc
import io
import json
import os
import shutil
import warnings

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torchaudio
import torchaudio.transforms as T
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
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
from nar_vae.distributed import (
    cleanup_distributed as cleanup_process_group,
)
from nar_vae.distributed import initialize_distributed
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


class EmiliaPreparer:
    """Prepares Emilia data for NAR-VAE training."""

    def __init__(
        self,
        dacvae_model: str | os.PathLike[str] | HubDACVAESource,
        device: str,
        dacvae_backend: str = "bundled",
        language: str = DEFAULT_LANGUAGE,
    ):
        self.device = device

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

    def tokenize_text(self, text: str, language: str | None = None) -> list:
        """
        Tokenize text with special tokens and emotion tags.

        Emotion tags like <laugh>, <sigh>, <cry> are converted to special tokens
        instead of being encoded as regular text.
        """
        return encode_tts_text(text, self.tokenizer, language=language or self.language)

    def process_sample(self, sample: dict) -> dict:
        """
        Process a single Emilia sample.

        Emilia format:
        - sample["mp3"] = {"array": ndarray, "sampling_rate": int}
        - sample["json"] = {"text": str, ...}
        """
        try:
            audio_data = sample.get("mp3")
            if audio_data is None:
                return None

            if isinstance(audio_data, dict):
                if "array" in audio_data:
                    audio = np.array(audio_data["array"])
                    sr = audio_data.get("sampling_rate", self.sample_rate)
                elif "bytes" in audio_data:
                    audio_io = io.BytesIO(audio_data["bytes"])
                    waveform, sr = torchaudio.load(audio_io)
                    audio = waveform.numpy().flatten()
                else:
                    return None
            else:
                return None

            if len(audio) == 0:
                return None

            json_data = sample.get("json")
            if json_data is None:
                return None

            if isinstance(json_data, dict):
                text = json_data.get("text", "")
            elif isinstance(json_data, str):
                parsed = json.loads(json_data)
                text = parsed.get("text", "")
            elif isinstance(json_data, bytes):
                parsed = json.loads(json_data.decode("utf-8"))
                text = parsed.get("text", "")
            else:
                return None

            if not text:
                return None

            latents = self.extract_latents(audio, sr)
            conditioning_ids = self.tokenize_text(text, self.language)

            result = {
                "latents": latents,
                LATENT_NUM_FRAMES_COLUMN: infer_latent_num_frames(latents),
                "conditioning_ids": conditioning_ids,
                "language": self.language,
            }
            return attach_representation_contract(result, self.representation_contract)
        except RepresentationContractError:
            raise
        except Exception:
            return None

    def process_batch(self, batch_data: list) -> dict:
        """Process a batch of samples."""
        processed = {
            "latents": [],
            LATENT_NUM_FRAMES_COLUMN: [],
            "conditioning_ids": [],
            "language": [],
            REPRESENTATION_CONTRACT_COLUMN: [],
        }

        for sample in batch_data:
            result = self.process_sample(sample)
            if result is not None:
                processed["latents"].append(result["latents"])
                processed[LATENT_NUM_FRAMES_COLUMN].append(result[LATENT_NUM_FRAMES_COLUMN])
                processed["conditioning_ids"].append(result["conditioning_ids"])
                processed["language"].append(result["language"])
                processed[REPRESENTATION_CONTRACT_COLUMN].append(
                    result[REPRESENTATION_CONTRACT_COLUMN]
                )

        return processed


def merge_parts(input_dir: str, output_dir: str, batch_size: int = 500):
    """Merge multiple dataset parts into a single dataset."""
    part_dirs = sorted(
        [os.path.join(input_dir, d) for d in os.listdir(input_dir) if d.startswith("part_")]
    )

    if not part_dirs:
        print("No parts found to merge")
        return None

    print(f"Found {len(part_dirs)} parts to merge")

    merged_datasets = []

    for i in tqdm(range(0, len(part_dirs), batch_size), desc="Loading batches"):
        batch_dirs = part_dirs[i : i + batch_size]
        batch_datasets = []

        for part_dir in tqdm(batch_dirs, desc=f"Batch {i // batch_size}", leave=False):
            try:
                ds = load_from_disk(part_dir)
                batch_datasets.append(ds)
            except Exception as e:
                print(f"Error loading {part_dir}: {e}")
                continue

        if batch_datasets:
            batch_merged = concatenate_datasets(batch_datasets)
            merged_datasets.append(batch_merged)
            print(f"Batch {i // batch_size}: {len(batch_merged)} samples")

    print("Final merge...")
    final_dataset = concatenate_datasets(merged_datasets)

    print(f"\nTotal samples: {len(final_dataset)}")
    print(f"Columns: {final_dataset.column_names}")

    print(f"Saving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    final_dataset.save_to_disk(output_dir)
    write_prepared_dataset_manifest(final_dataset, output_dir)

    print(f"\nDone! Merged dataset saved to: {output_dir}")

    return final_dataset


def prepare_emilia_dataset(
    output_dir: str,
    language: str,
    dataset_type: str,
    dacvae_model: str | os.PathLike[str] | HubDACVAESource,
    batch_size: int,
    max_samples: int,
    dacvae_backend: str = "bundled",
    dataset_revision: str | None = None,
    dataset_download_workers: int = DEFAULT_DATASET_DOWNLOAD_WORKERS,
):
    """Prepare Emilia dataset with distributed processing support."""
    source = resolve_dataset_source(
        "amphion/Emilia-Dataset",
        revision=dataset_revision,
        download_workers=dataset_download_workers,
    )
    local_rank, world_size, is_distributed = setup_distributed()
    rank = dist.get_rank() if is_distributed else 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0

    if is_main:
        print("=" * 80)
        print("Emilia Dataset Preparation")
        print("=" * 80)
        print(f"Dataset: {dataset_type}/{language}")
        print(f"Output: {output_dir}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print("=" * 80)
        os.makedirs(output_dir, exist_ok=True)

    if is_distributed:
        dist.barrier()

    path = f"{dataset_type}/{language}/**/*.tar"
    dataset = load_dataset(
        source.location,
        data_files={"train": path},
        split="train",
        streaming=True,
        **source.load_dataset_kwargs(),
    )

    if is_main:
        print("Dataset loaded in streaming mode")

    preparer = EmiliaPreparer(
        dacvae_model=dacvae_model,
        device=device,
        dacvae_backend=dacvae_backend,
        language=language,
    )

    if is_main:
        print(f"DACVAE sample rate: {preparer.sample_rate}")

    batch_data = []
    sample_count = 0
    part_idx = rank

    dataset_iter = iter(dataset)

    try:
        for i, sample in enumerate(tqdm(dataset_iter, desc=f"Rank {rank}", disable=not is_main)):
            if max_samples and i >= max_samples:
                break
            if is_distributed and i % world_size != rank:
                continue

            batch_data.append(sample)
            sample_count += 1

            if len(batch_data) >= batch_size:
                processed = preparer.process_batch(batch_data)

                if len(processed["latents"]) > 0:
                    ds = Dataset.from_dict(processed)
                    part_dir = os.path.join(output_dir, f"part_{part_idx:05d}")
                    ds.save_to_disk(part_dir)

                    if is_main:
                        print(f"Saved part {part_idx} with {len(ds)} samples")

                    part_idx += world_size if is_distributed else 1

                batch_data = []
                gc.collect()
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        if is_main:
            print("Interrupted! Saving remaining data...")

    if batch_data:
        processed = preparer.process_batch(batch_data)
        if len(processed["latents"]) > 0:
            ds = Dataset.from_dict(processed)
            part_dir = os.path.join(output_dir, f"part_{part_idx:05d}")
            ds.save_to_disk(part_dir)

    if is_distributed:
        dist.barrier()

    if is_main:
        print(f"\nRank {rank}: {sample_count} samples processed")

        merged_dir = output_dir + "_merged"
        final_dataset = merge_parts(output_dir, merged_dir)

        if final_dataset is not None:
            for d in os.listdir(output_dir):
                if d.startswith("part_"):
                    shutil.rmtree(os.path.join(output_dir, d), ignore_errors=True)

            if os.path.exists(merged_dir):
                for f in os.listdir(merged_dir):
                    shutil.move(os.path.join(merged_dir, f), output_dir)
                shutil.rmtree(merged_dir, ignore_errors=True)

            print("\nDataset columns (for training):")
            print("  - latents: DACVAE encoded audio")
            print("  - latent_num_frames: Persisted DACVAE frame count for efficient batching")
            print("  - conditioning_ids: Tokenized text with special tokens")
            print(f"  - {REPRESENTATION_CONTRACT_COLUMN}: Versioned text/codec representation")
            print("\nTokenizer info:")
            print("  Tokenizer: tiktoken cl100k_base")
            print(f"  Vocab size: {TOKENIZER_LENGTH}")
            print(f"  Special tokens: {START_OF_TEXT} to {PAD_TOKEN}")

    cleanup_distributed()
