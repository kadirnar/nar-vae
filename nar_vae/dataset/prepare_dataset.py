import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
import torch
import torchaudio
from tqdm import tqdm

from nar_vae.dacvae import HubDACVAESource, load_dacvae, normalize_dacvae_source
from nar_vae.dataset.identity import write_prepared_dataset_manifest
from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_COLUMN,
    attach_representation_contract,
    build_representation_contract,
)
from nar_vae.dataset.sampling import LATENT_NUM_FRAMES_COLUMN, infer_latent_num_frames
from nar_vae.dataset.sources import (
    DEFAULT_DATASET_DOWNLOAD_WORKERS,
    resolve_dataset_source,
)
from nar_vae.dataset.speaker_references import (
    build_speaker_index,
    collect_reference_audio_with_language,
    select_reference_indices,
)
from nar_vae.languages import DEFAULT_LANGUAGE, normalize_language
from nar_vae.text_frontend import FrozenTextFrontend, FrozenTextFrontendSpec
from nar_vae.tokenization import encode_tts_text
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS

_DATASETS_IMPORT_ERROR: ImportError | None = None
try:
    from datasets import Audio, Dataset, load_dataset
except ImportError as exc:
    _DATASETS_IMPORT_ERROR = exc
    Audio = None
    Dataset = None
    load_dataset = None


def _require_datasets() -> None:
    """Raise an actionable error when required dataset tools are unavailable."""
    if _DATASETS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Dataset preparation dependencies are unavailable. Reinstall the single package "
            "with `python -m pip install -e .`."
        ) from _DATASETS_IMPORT_ERROR


class DatasetPreparer:
    """
    Prepares TTS dataset for EchoDiT training.

    Converts audio to DACVAE latents and text to token IDs.
    """

    def __init__(
        self,
        dacvae_model: str | os.PathLike[str] | HubDACVAESource | None = None,
        dacvae_backend: str = "bundled",
        device: str = "cuda",
        target_sample_rate: int | None = None,
        max_duration: float = 30.0,  # Max audio duration in seconds
        min_duration: float = 0.5,  # Min audio duration in seconds
        max_reference_duration: float = DEFAULT_MAX_REFERENCE_SECONDS,
        language: str = DEFAULT_LANGUAGE,
        dacvae_revision: str | None = None,
        dacvae_filename: str | None = None,
        dacvae_sha256: str | None = None,
        text_frontend_spec: FrozenTextFrontendSpec | None = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_duration = max_duration
        self.min_duration = min_duration
        if max_reference_duration <= 0:
            raise ValueError("max_reference_duration must be positive")
        self.max_reference_duration = max_reference_duration
        self.language = normalize_language(language)
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
            device=self.device,
            freeze=True,
            expected_sha256=dacvae_sha256,
        )
        codec_sample_rate = int(self.dacvae.sample_rate)
        if target_sample_rate is not None and target_sample_rate != codec_sample_rate:
            raise ValueError(
                "target_sample_rate must match the loaded DACVAE sample rate: "
                f"{target_sample_rate} != {codec_sample_rate}."
            )
        self.target_sample_rate = codec_sample_rate
        self.frozen_text_frontend = (
            FrozenTextFrontend(text_frontend_spec, device=self.device)
            if text_frontend_spec is not None
            else None
        )
        self.representation_contract = build_representation_contract(
            self.dacvae,
            codec_source=codec_source,
            text_frontend=text_frontend_spec,
        )

        if self.frozen_text_frontend is None:
            print("Loading tokenizer: tiktoken cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            print(
                "Loading frozen text features: "
                f"{text_frontend_spec.model_id}@{text_frontend_spec.revision}"
            )
            self.tokenizer = None

        print(f"Ready on {self.device}")

    @torch.no_grad()
    def encode_audio(self, audio_path: str) -> np.ndarray | None:
        """
        Encode audio file to DACVAE latents.

        Args:
            audio_path: Path to audio file

        Returns:
            Latent array [128, T] or None if failed
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Check duration
            duration = waveform.shape[1] / self.target_sample_rate
            if duration < self.min_duration:
                print(f"Skipping {audio_path}: too short ({duration:.2f}s)")
                return None
            if duration > self.max_duration:
                # Truncate
                max_samples = int(self.max_duration * self.target_sample_rate)
                waveform = waveform[:, :max_samples]

            # Encode with DACVAE
            waveform = waveform.unsqueeze(0).to(self.device)  # [1, 1, samples]
            latents = self.dacvae.encode(waveform)  # [1, 128, T]

            return latents.squeeze(0).cpu().numpy()  # [128, T]

        except Exception as e:
            print(f"Error encoding {audio_path}: {e}")
            return None

    @torch.no_grad()
    def encode_audio_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray | None:
        """
        Encode audio array to DACVAE latents.

        Args:
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate of audio

        Returns:
            Latent array [128, T] or None if failed
        """
        try:
            # Convert to tensor
            if audio_array.ndim == 1:
                waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_array).float()
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Check duration
            duration = waveform.shape[1] / self.target_sample_rate
            if duration < self.min_duration or duration > self.max_duration:
                return None

            # Encode
            waveform = waveform.unsqueeze(0).to(self.device)
            latents = self.dacvae.encode(waveform)

            return latents.squeeze(0).cpu().numpy()

        except Exception as e:
            print(f"Error encoding audio array: {e}")
            return None

    def tokenize_text(self, text: str, language: str | None = None) -> list[int]:
        """
        Tokenize text with special tokens for TTS.

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        if self.tokenizer is None:
            raise RuntimeError("tokenize_text is available only for the legacy scratch frontend.")
        return encode_tts_text(
            text.strip(),
            self.tokenizer,
            language=language or self.language,
        )

    def encode_text_conditioning(
        self,
        text: str,
        language: str | None = None,
    ) -> tuple[list[int], np.ndarray | None]:
        """Return aligned IDs and optional cached frozen contextual states."""
        if getattr(self, "frozen_text_frontend", None) is None:
            return self.tokenize_text(text, language), None
        batch = self.frozen_text_frontend.encode(
            [text.strip()],
            inputs_are_phonemes=self.frozen_text_frontend.spec.input_mode == "phonemes",
        )
        token_count = int(batch.attention_mask[0].sum().item())
        return (
            batch.input_ids[0, :token_count].tolist(),
            batch.features[0, :token_count].numpy(),
        )

    @torch.no_grad()
    def encode_speaker_audio(
        self,
        references: list[tuple[np.ndarray, int]],
    ) -> np.ndarray | None:
        """Concatenate and encode bounded same-speaker reference utterances."""
        waveforms = []
        remaining = int(self.max_reference_duration * self.target_sample_rate)
        for audio_array, sample_rate in references:
            waveform = torch.as_tensor(np.asarray(audio_array)).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2 and waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.ndim != 2 or waveform.shape[-1] == 0 or sample_rate <= 0:
                continue
            if sample_rate != self.target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    sample_rate,
                    self.target_sample_rate,
                )
            waveform = waveform[..., :remaining]
            if waveform.shape[-1]:
                waveforms.append(waveform)
                remaining -= waveform.shape[-1]
            if remaining <= 0:
                break

        if not waveforms:
            return None
        waveform = torch.cat(waveforms, dim=-1).unsqueeze(0).to(self.device)
        return self.dacvae.encode(waveform).squeeze(0).cpu().numpy()

    def process_sample(
        self,
        audio_path: str | None = None,
        audio_array: np.ndarray | None = None,
        sample_rate: int | None = None,
        text: str = "",
        speaker_references: list[tuple[np.ndarray, int]] | None = None,
        language: str | None = None,
        speaker_language: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Process a single sample.

        Args:
            audio_path: Path to audio file (use this OR audio_array)
            audio_array: Audio samples as numpy array
            sample_rate: Sample rate (required if using audio_array)
            text: Transcript text

        Returns:
            Dictionary with 'latents' and 'conditioning_ids' or None
        """
        target_language = normalize_language(language or self.language)
        reference_language = normalize_language(speaker_language or target_language)

        # Encode audio
        if audio_path is not None:
            latents = self.encode_audio(audio_path)
        elif audio_array is not None and sample_rate is not None:
            latents = self.encode_audio_array(audio_array, sample_rate)
        else:
            return None

        if latents is None:
            return None

        # Tokenize text
        conditioning_ids, conditioning_features = self.encode_text_conditioning(
            text,
            target_language,
        )

        result = {
            "latents": latents,
            LATENT_NUM_FRAMES_COLUMN: infer_latent_num_frames(latents),
            "conditioning_ids": conditioning_ids,
            "language": target_language,
        }
        if conditioning_features is not None:
            result["conditioning_features"] = conditioning_features
        if speaker_references is not None:
            speaker_latents = self.encode_speaker_audio(speaker_references)
            if speaker_latents is None:
                return None
            result["speaker_latents"] = speaker_latents
            result["speaker_language"] = reference_language
        return attach_representation_contract(result, self.representation_contract)


def prepare_from_hf_dataset(
    dataset_name: str,
    output_dir: str,
    audio_column: str = "audio",
    text_column: str = "text",
    split: str = "train",
    max_samples: int | None = None,
    device: str = "cuda",
    dacvae_model: str | os.PathLike[str] | HubDACVAESource | None = None,
    dacvae_backend: str = "bundled",
    speaker_id_column: str | None = None,
    max_reference_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
    max_reference_utterances: int = 3,
    reference_seed: int = 1234,
    language: str = DEFAULT_LANGUAGE,
    language_column: str | None = None,
    dataset_revision: str | None = None,
    dataset_download_workers: int = DEFAULT_DATASET_DOWNLOAD_WORKERS,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
    dacvae_sha256: str | None = None,
    text_frontend_spec: FrozenTextFrontendSpec | None = None,
):
    """
    Prepare dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Output directory
        audio_column: Name of audio column
        text_column: Name of text column
        split: Dataset split
        max_samples: Maximum samples to process
        device: Device to use
    """
    _require_datasets()
    source = resolve_dataset_source(
        dataset_name,
        revision=dataset_revision,
        download_workers=dataset_download_workers,
    )
    print(f"\nLoading dataset: {dataset_name}")
    ds = load_dataset(
        source.location,
        split=split,
        **source.load_dataset_kwargs(),
    )

    preparer = DatasetPreparer(
        dacvae_model=dacvae_model,
        dacvae_backend=dacvae_backend,
        device=device,
        max_reference_duration=max_reference_seconds,
        language=language,
        dacvae_revision=dacvae_revision,
        dacvae_filename=dacvae_filename,
        dacvae_sha256=dacvae_sha256,
        text_frontend_spec=text_frontend_spec,
    )

    speaker_index = (
        build_speaker_index(ds, speaker_id_column) if speaker_id_column is not None else None
    )

    # Cast audio column
    if audio_column in ds.column_names:
        ds = ds.cast_column(
            audio_column,
            Audio(sampling_rate=preparer.target_sample_rate),
        )

    processed_samples = []
    total = min(len(ds), max_samples) if max_samples else len(ds)

    print(f"\nProcessing {total} samples...")
    for i, sample in enumerate(tqdm(ds, total=total)):
        if max_samples and i >= max_samples:
            break

        audio_data = sample[audio_column]
        text = sample[text_column]
        target_language = normalize_language(
            sample.get(language_column, language) if language_column else language
        )
        speaker_references = None
        speaker_language = None
        if speaker_index is not None:
            reference_indices = select_reference_indices(
                speaker_index,
                speaker_id=sample.get(speaker_id_column),
                target_index=i,
                maximum_utterances=max_reference_utterances,
                seed=reference_seed,
            )
            speaker_references, speaker_language = collect_reference_audio_with_language(
                ds,
                reference_indices,
                audio_column=audio_column,
                language_column=language_column,
                fallback_language=target_language,
            )
            if not speaker_references:
                continue

        result = preparer.process_sample(
            audio_array=audio_data["array"],
            sample_rate=audio_data["sampling_rate"],
            text=text,
            speaker_references=speaker_references,
            language=target_language,
            speaker_language=speaker_language,
        )

        if result is not None:
            processed_samples.append(result)

    print(f"\nProcessed {len(processed_samples)} / {total} samples successfully")

    # Save dataset
    save_dataset(processed_samples, output_dir)


def prepare_from_local_folder(
    input_dir: str,
    output_dir: str,
    transcript_file: str = "transcripts.txt",
    audio_extensions: list[str] | None = None,
    device: str = "cuda",
    dacvae_model: str | os.PathLike[str] | HubDACVAESource | None = None,
    dacvae_backend: str = "bundled",
    language: str = DEFAULT_LANGUAGE,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
    dacvae_sha256: str | None = None,
    text_frontend_spec: FrozenTextFrontendSpec | None = None,
):
    """
    Prepare dataset from local folder.

    Expected structure:
        input_dir/
            audio1.wav
            audio2.wav
            transcripts.txt  (format: "audio1.wav|transcript text")

    Args:
        input_dir: Input directory with audio files
        output_dir: Output directory
        transcript_file: Name of transcript file
        audio_extensions: Supported audio extensions
        device: Device to use
    """
    _require_datasets()
    input_path = Path(input_dir)
    transcript_path = input_path / transcript_file
    audio_extensions = audio_extensions or [".wav", ".mp3", ".flac", ".ogg"]

    # Load transcripts
    transcripts = {}
    if transcript_path.exists():
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    filename, text = line.split("|", 1)
                    transcripts[filename.strip()] = text.strip()
    else:
        print(f"Warning: {transcript_file} not found, looking for .txt files")
        # Try to find individual .txt files
        for audio_file in input_path.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                txt_file = audio_file.with_suffix(".txt")
                if txt_file.exists():
                    with open(txt_file, encoding="utf-8") as f:
                        transcripts[audio_file.name] = f.read().strip()

    if not transcripts:
        raise ValueError("No transcripts found!")

    print(f"Found {len(transcripts)} transcripts")

    preparer = DatasetPreparer(
        dacvae_model=dacvae_model,
        dacvae_backend=dacvae_backend,
        device=device,
        language=language,
        dacvae_revision=dacvae_revision,
        dacvae_filename=dacvae_filename,
        dacvae_sha256=dacvae_sha256,
        text_frontend_spec=text_frontend_spec,
    )

    processed_samples = []
    for filename, text in tqdm(transcripts.items()):
        audio_path = input_path / filename

        if not audio_path.exists():
            # Try different extensions
            found = False
            for ext in audio_extensions:
                alt_path = audio_path.with_suffix(ext)
                if alt_path.exists():
                    audio_path = alt_path
                    found = True
                    break
            if not found:
                print(f"Audio not found: {filename}")
                continue

        result = preparer.process_sample(
            audio_path=str(audio_path),
            text=text,
            language=language,
        )
        if result is not None:
            processed_samples.append(result)

    print(f"\nProcessed {len(processed_samples)} samples successfully")
    save_dataset(processed_samples, output_dir)


def prepare_from_csv(
    input_file: str,
    output_dir: str,
    audio_column: str = "audio_path",
    text_column: str = "text",
    base_dir: str | None = None,
    device: str = "cuda",
    dacvae_model: str | os.PathLike[str] | HubDACVAESource | None = None,
    dacvae_backend: str = "bundled",
    language: str = DEFAULT_LANGUAGE,
    dacvae_revision: str | None = None,
    dacvae_filename: str | None = None,
    dacvae_sha256: str | None = None,
    text_frontend_spec: FrozenTextFrontendSpec | None = None,
):
    """
    Prepare dataset from CSV file.

    CSV format:
        audio_path,text
        /path/to/audio1.wav,"transcript text"

    Args:
        input_file: Path to CSV file
        output_dir: Output directory
        audio_column: Name of audio path column
        text_column: Name of text column
        base_dir: Base directory for relative paths
        device: Device to use
    """
    _require_datasets()
    samples = []
    with open(input_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(
                {
                    "audio_path": row[audio_column],
                    "text": row[text_column],
                }
            )

    print(f"Found {len(samples)} samples in CSV")

    preparer = DatasetPreparer(
        dacvae_model=dacvae_model,
        dacvae_backend=dacvae_backend,
        device=device,
        language=language,
        dacvae_revision=dacvae_revision,
        dacvae_filename=dacvae_filename,
        dacvae_sha256=dacvae_sha256,
        text_frontend_spec=text_frontend_spec,
    )

    processed_samples = []
    for sample in tqdm(samples):
        audio_path = sample["audio_path"]
        if base_dir and not os.path.isabs(audio_path):
            audio_path = os.path.join(base_dir, audio_path)

        result = preparer.process_sample(
            audio_path=audio_path,
            text=sample["text"],
            language=language,
        )
        if result is not None:
            processed_samples.append(result)

    print(f"\nProcessed {len(processed_samples)} samples successfully")
    save_dataset(processed_samples, output_dir)


def save_dataset(
    samples: list[dict[str, Any]],
    output_dir: str,
    *,
    allow_legacy_representation: bool = False,
):
    """Save processed samples as HuggingFace dataset."""
    _require_datasets()
    if not samples:
        raise ValueError("No valid samples were prepared.")
    os.makedirs(output_dir, exist_ok=True)

    optional_speaker = ["speaker_latents" in sample for sample in samples]
    if any(optional_speaker) and not all(optional_speaker):
        raise ValueError("speaker_latents must be present in every prepared sample or none.")
    fields = ["latents", "conditioning_ids"]
    optional_conditioning_features = ["conditioning_features" in sample for sample in samples]
    if any(optional_conditioning_features) and not all(optional_conditioning_features):
        raise ValueError("conditioning_features must be present in every prepared sample or none.")
    if all(optional_conditioning_features):
        declared_feature_dtypes: set[str] = set()
        for index, sample in enumerate(samples):
            representation = sample.get(REPRESENTATION_CONTRACT_COLUMN)
            frontend = (
                representation.get("text_frontend") if isinstance(representation, dict) else None
            )
            declared_dtype = frontend.get("feature_dtype") if isinstance(frontend, dict) else None
            if declared_dtype not in {"float16", "float32"}:
                raise ValueError(
                    f"Sample {index} has cached features without a supported frozen frontend "
                    "dtype contract."
                )
            actual_dtype = str(np.asarray(sample["conditioning_features"]).dtype)
            if actual_dtype != declared_dtype:
                raise ValueError(
                    f"Sample {index} conditioning_features use {actual_dtype}, but the frozen "
                    f"frontend contract declares {declared_dtype}."
                )
            declared_feature_dtypes.add(declared_dtype)
        if len(declared_feature_dtypes) != 1:
            raise ValueError("All cached frozen features must use one declared dtype.")
        fields.append("conditioning_features")
    optional_frame_lengths = [LATENT_NUM_FRAMES_COLUMN in sample for sample in samples]
    if any(optional_frame_lengths) and not all(optional_frame_lengths):
        raise ValueError(
            f"{LATENT_NUM_FRAMES_COLUMN} must be present in every prepared sample or none."
        )
    if all(optional_frame_lengths):
        fields.append(LATENT_NUM_FRAMES_COLUMN)
    representation_contracts = [REPRESENTATION_CONTRACT_COLUMN in sample for sample in samples]
    if any(representation_contracts) and not all(representation_contracts):
        raise ValueError(
            f"{REPRESENTATION_CONTRACT_COLUMN} must be present in every prepared sample or none."
        )
    if not any(representation_contracts) and not allow_legacy_representation:
        raise ValueError(
            f"{REPRESENTATION_CONTRACT_COLUMN} is required for newly prepared data. "
            "Set allow_legacy_representation=True only when migrating legacy rows."
        )
    if all(representation_contracts):
        fields.append(REPRESENTATION_CONTRACT_COLUMN)
    optional_language = ["language" in sample for sample in samples]
    if any(optional_language) and not all(optional_language):
        raise ValueError("language must be present in every prepared sample or none.")
    if all(optional_language):
        fields.append("language")
    if all(optional_speaker):
        fields.append("speaker_latents")
        optional_speaker_language = ["speaker_language" in sample for sample in samples]
        if any(optional_speaker_language) and not all(optional_speaker_language):
            raise ValueError(
                "speaker_language must be present with every speaker reference or none."
            )
        if all(optional_speaker_language):
            fields.append("speaker_language")
    dataset_dict = {field: [sample[field] for sample in samples] for field in fields}

    ds = Dataset.from_dict(dataset_dict)
    if all(optional_conditioning_features):
        arrow_feature = ds.features["conditioning_features"]
        while hasattr(arrow_feature, "feature"):
            arrow_feature = arrow_feature.feature
        arrow_dtype = getattr(arrow_feature, "dtype", None)
        declared_dtype = next(iter(declared_feature_dtypes))
        if arrow_dtype != declared_dtype:
            raise ValueError(
                "Prepared Arrow feature dtype does not match the frozen frontend contract: "
                f"{arrow_dtype!r} != {declared_dtype!r}."
            )
    ds.save_to_disk(output_dir)
    write_prepared_dataset_manifest(ds, output_dir)

    print(f"\nDataset saved to: {output_dir}")
    print(f"Total samples: {len(ds)}")
