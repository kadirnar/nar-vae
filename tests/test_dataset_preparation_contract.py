"""CPU-only contracts for reproducible acoustic dataset preparation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from nar_vae.dacvae import HubDACVAESource
from nar_vae.dataset.emilia_prepare import EmiliaPreparer, prepare_emilia_dataset
from nar_vae.dataset.finetune_prepare import DataPreparer, prepare_finetune_dataset
from nar_vae.dataset.prepare import DatasetPreparer, prepare_dataset
from nar_vae.dataset.prepare_dataset import (
    DatasetPreparer as FileDatasetPreparer,
)
from nar_vae.dataset.prepare_dataset import prepare_from_hf_dataset
from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_COLUMN,
    REPRESENTATION_CONTRACT_VERSION,
    TEXT_FRONTEND_NAME,
    TEXT_FRONTEND_VERSION,
    RepresentationContractError,
    attach_representation_contract,
    build_representation_contract,
)
from nar_vae.dataset.sources import resolve_dataset_source

DATASET_REVISION_A = "a" * 40
DATASET_REVISION_B = "b" * 40
DATASET_REVISION_C = "c" * 40


def fake_codec(
    latent_width: int = 4,
    *,
    identifier: str = "codec/source-v1",
    revision: str | None = None,
    filename: str | None = None,
):
    return SimpleNamespace(
        nar_vae_backend="bundled",
        nar_vae_codec_identifier=identifier,
        nar_vae_codec_revision=revision,
        nar_vae_codec_filename=filename,
        nar_vae_codec_sha256="f" * 64,
        sample_rate=48000,
        hop_length=1920,
        quantizer=SimpleNamespace(
            out_proj=SimpleNamespace(in_channels=latent_width),
        ),
    )


def empty_dataset():
    dataset = MagicMock()
    dataset.column_names = []
    dataset.__len__.return_value = 0
    dataset.__iter__.return_value = iter(())
    return dataset


def saved_dataset():
    dataset = MagicMock()
    dataset.__len__.return_value = 0
    return dataset


class DatasetSourceContractTest(unittest.TestCase):
    def test_remote_source_requires_pinned_revision_and_bounded_workers(self):
        source = resolve_dataset_source(
            "speech/example",
            revision=DATASET_REVISION_A,
            download_workers=6,
        )
        self.assertFalse(source.is_local)
        self.assertEqual(source.revision, DATASET_REVISION_A)
        self.assertEqual(
            source.load_dataset_kwargs(),
            {"num_proc": 6, "revision": DATASET_REVISION_A},
        )

        for revision in (
            None,
            "",
            "  ",
            "main",
            "refs/heads/main",
            "origin/main",
            "dataset-v1",
        ):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "revision"):
                resolve_dataset_source(
                    "speech/example",
                    revision=revision,
                    download_workers=6,
                )

        for workers in (True, 0, 33, 1.5):
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(ValueError, "between 1 and 32"),
            ):
                resolve_dataset_source(
                    "speech/example",
                    revision=DATASET_REVISION_A,
                    download_workers=workers,
                )

    def test_local_source_never_receives_a_remote_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            source = resolve_dataset_source(
                directory,
                revision="main",
                download_workers=2,
            )

        self.assertTrue(source.is_local)
        self.assertIsNone(source.revision)
        self.assertEqual(source.load_dataset_kwargs(), {"num_proc": 2})

    def test_explicit_missing_local_path_does_not_fall_back_to_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                resolve_dataset_source(
                    missing,
                    revision=DATASET_REVISION_A,
                    download_workers=2,
                )


class RepresentationContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = build_representation_contract(
            fake_codec(),
            codec_source="codec/source-v1",
        )

    def test_contract_records_the_resolved_codec_and_text_frontend(self):
        self.assertEqual(
            self.contract.to_dict(),
            {
                "contract_version": REPRESENTATION_CONTRACT_VERSION,
                "text_frontend_name": TEXT_FRONTEND_NAME,
                "text_frontend_version": TEXT_FRONTEND_VERSION,
                "codec_source": "codec/source-v1",
                "codec_backend": "bundled",
                "codec_revision": None,
                "codec_filename": None,
                "codec_sha256": "f" * 64,
                "sample_rate": 48000,
                "hop_length": 1920,
                "latent_width": 4,
            },
        )

    def test_contract_records_a_revision_pinned_hub_codec(self):
        source = HubDACVAESource("codec/source", "a" * 40, "artifacts/weights.pth")
        contract = build_representation_contract(
            fake_codec(
                identifier=source.repo_id,
                revision=source.revision,
                filename=source.filename,
            ),
            codec_source=source,
        )

        self.assertEqual(contract.codec_source, "codec/source")
        self.assertEqual(contract.codec_revision, "a" * 40)
        self.assertEqual(contract.codec_filename, "artifacts/weights.pth")
        self.assertEqual(contract.codec_sha256, "f" * 64)

    def test_attaching_contract_validates_target_and_speaker_latent_widths(self):
        row = {
            "latents": np.zeros((4, 3), dtype=np.float32),
            "speaker_latents": np.zeros((4, 2), dtype=np.float32),
        }
        attach_representation_contract(row, self.contract)
        self.assertEqual(row[REPRESENTATION_CONTRACT_COLUMN], self.contract.to_dict())

        for field_name in ("latents", "speaker_latents"):
            invalid = {
                "latents": np.zeros((4, 3), dtype=np.float32),
                "speaker_latents": np.zeros((4, 2), dtype=np.float32),
            }
            invalid[field_name] = np.zeros((3, 2), dtype=np.float32)
            with (
                self.subTest(field_name=field_name),
                self.assertRaisesRegex(
                    RepresentationContractError,
                    "latent width 3",
                ),
            ):
                attach_representation_contract(invalid, self.contract)

    def test_every_preparer_attaches_the_shared_row_contract(self):
        standard = DatasetPreparer.__new__(DatasetPreparer)
        standard.language = "en"
        standard.sample_rate = 48000
        standard.representation_contract = self.contract
        standard.extract_latents = lambda audio, sr: np.zeros((4, 3), dtype=np.float32)
        standard.tokenize_text = lambda text, language=None: [1, 2]
        row = standard.process_sample(
            {"audio": {"array": [0.0], "sampling_rate": 48000}, "text": "hello"}
        )
        self.assertEqual(row[REPRESENTATION_CONTRACT_COLUMN], self.contract.to_dict())
        standard.extract_latents = lambda audio, sr: np.zeros((3, 3), dtype=np.float32)
        with self.assertRaisesRegex(RepresentationContractError, "latent width 3"):
            standard.process_sample(
                {"audio": {"array": [0.0], "sampling_rate": 48000}, "text": "hello"}
            )

        emilia = EmiliaPreparer.__new__(EmiliaPreparer)
        emilia.language = "en"
        emilia.sample_rate = 48000
        emilia.representation_contract = self.contract
        emilia.extract_latents = lambda audio, sr: np.zeros((4, 3), dtype=np.float32)
        emilia.tokenize_text = lambda text, language=None: [1, 2]
        batch = emilia.process_batch(
            [{"mp3": {"array": [0.0], "sampling_rate": 48000}, "json": {"text": "hi"}}]
        )
        self.assertEqual(batch[REPRESENTATION_CONTRACT_COLUMN], [self.contract.to_dict()])

        finetune = DataPreparer.__new__(DataPreparer)
        finetune.language = "en"
        finetune.sample_rate = 48000
        finetune.representation_contract = self.contract
        finetune.extract_latents = lambda audio, sr: np.zeros((4, 3), dtype=np.float32)
        finetune.tokenize_text = lambda text, language=None: [1, 2]
        row = finetune.process_example(
            {"audio": {"array": [0.0], "sampling_rate": 48000}, "text": "hello"}
        )
        self.assertEqual(row[REPRESENTATION_CONTRACT_COLUMN], self.contract.to_dict())

        file_preparer = FileDatasetPreparer.__new__(FileDatasetPreparer)
        file_preparer.language = "en"
        file_preparer.representation_contract = self.contract
        file_preparer.encode_audio_array = lambda audio, sr: np.zeros((4, 3), dtype=np.float32)
        file_preparer.tokenize_text = lambda text, language=None: [1, 2]
        row = file_preparer.process_sample(
            audio_array=np.zeros(4, dtype=np.float32),
            sample_rate=48000,
            text="hello",
        )
        self.assertEqual(row[REPRESENTATION_CONTRACT_COLUMN], self.contract.to_dict())


class DatasetLoadingThreadingTest(unittest.TestCase):
    def test_standard_remote_source_threads_revision_and_worker_bound(self):
        raw = empty_dataset()
        prepared = saved_dataset()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("nar_vae.dataset.prepare.setup_distributed", return_value=(0, 1, False)),
                patch("nar_vae.dataset.prepare.snapshot_download") as snapshot_download,
                patch("nar_vae.dataset.prepare.load_dataset", return_value=raw) as load_dataset,
                patch(
                    "nar_vae.dataset.prepare.DatasetPreparer",
                    return_value=SimpleNamespace(sample_rate=48000),
                ),
                patch("nar_vae.dataset.prepare.Dataset.from_list", return_value=prepared),
                patch("nar_vae.dataset.prepare.write_prepared_dataset_manifest") as write_manifest,
                patch("nar_vae.dataset.prepare.cleanup_distributed"),
            ):
                prepare_dataset(
                    "speech/example",
                    str(Path(directory) / "prepared"),
                    "train",
                    "codec/source-v1",
                    0,
                    "",
                    dataset_revision=DATASET_REVISION_A,
                    dataset_download_workers=5,
                )

        snapshot_download.assert_called_once_with(
            repo_id="speech/example",
            repo_type="dataset",
            revision=DATASET_REVISION_A,
            max_workers=5,
        )
        load_dataset.assert_called_once_with(
            "speech/example",
            split="train",
            revision=DATASET_REVISION_A,
            num_proc=5,
        )
        write_manifest.assert_called_once_with(prepared, str(Path(directory) / "prepared"))

    def test_standard_local_source_skips_snapshot_and_revision(self):
        raw = empty_dataset()
        prepared = saved_dataset()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("nar_vae.dataset.prepare.setup_distributed", return_value=(0, 1, False)),
                patch("nar_vae.dataset.prepare.snapshot_download") as snapshot_download,
                patch("nar_vae.dataset.prepare.load_dataset", return_value=raw) as load_dataset,
                patch(
                    "nar_vae.dataset.prepare.DatasetPreparer",
                    return_value=SimpleNamespace(sample_rate=48000),
                ),
                patch("nar_vae.dataset.prepare.Dataset.from_list", return_value=prepared),
                patch("nar_vae.dataset.prepare.write_prepared_dataset_manifest") as write_manifest,
                patch("nar_vae.dataset.prepare.cleanup_distributed"),
            ):
                prepare_dataset(
                    directory,
                    str(Path(directory) / "prepared"),
                    "train",
                    "codec/source-v1",
                    0,
                    "",
                    dataset_revision="main",
                    dataset_download_workers=3,
                )

        snapshot_download.assert_not_called()
        load_dataset.assert_called_once_with(directory, split="train", num_proc=3)
        write_manifest.assert_called_once_with(prepared, str(Path(directory) / "prepared"))

    def test_all_other_hub_preparation_paths_thread_revision_and_workers(self):
        raw = empty_dataset()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("nar_vae.dataset.prepare_dataset.load_dataset", return_value=raw) as load_hf,
                patch("nar_vae.dataset.prepare_dataset.DatasetPreparer"),
                patch("nar_vae.dataset.prepare_dataset.save_dataset"),
            ):
                prepare_from_hf_dataset(
                    "speech/example",
                    str(Path(directory) / "hf"),
                    dataset_revision=DATASET_REVISION_A,
                    dataset_download_workers=4,
                )
            load_hf.assert_called_once_with(
                "speech/example",
                split="train",
                revision=DATASET_REVISION_A,
                num_proc=4,
            )

            with (
                patch(
                    "nar_vae.dataset.finetune_prepare.setup_distributed",
                    return_value=(0, 1, False),
                ),
                patch(
                    "nar_vae.dataset.finetune_prepare.load_dataset", return_value=raw
                ) as load_finetune,
                patch(
                    "nar_vae.dataset.finetune_prepare.DataPreparer",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "nar_vae.dataset.finetune_prepare.Dataset.from_list",
                    return_value=(prepared_finetune := saved_dataset()),
                ),
                patch(
                    "nar_vae.dataset.finetune_prepare.write_prepared_dataset_manifest"
                ) as write_finetune_manifest,
                patch("nar_vae.dataset.finetune_prepare.cleanup_distributed"),
            ):
                prepare_finetune_dataset(
                    "speech/example",
                    str(Path(directory) / "finetune"),
                    dataset_revision=DATASET_REVISION_B,
                    dataset_download_workers=7,
                )
            load_finetune.assert_called_once_with(
                "speech/example",
                split="train",
                revision=DATASET_REVISION_B,
                num_proc=7,
            )
            write_finetune_manifest.assert_called_once_with(
                prepared_finetune,
                str(Path(directory) / "finetune"),
            )

            with (
                patch(
                    "nar_vae.dataset.emilia_prepare.setup_distributed",
                    return_value=(0, 1, False),
                ),
                patch(
                    "nar_vae.dataset.emilia_prepare.load_dataset", return_value=[]
                ) as load_emilia,
                patch(
                    "nar_vae.dataset.emilia_prepare.EmiliaPreparer",
                    return_value=SimpleNamespace(sample_rate=48000),
                ),
                patch("nar_vae.dataset.emilia_prepare.merge_parts", return_value=None),
                patch("nar_vae.dataset.emilia_prepare.cleanup_distributed"),
            ):
                prepare_emilia_dataset(
                    str(Path(directory) / "emilia"),
                    "en",
                    "Emilia",
                    "codec/source-v1",
                    8,
                    0,
                    dataset_revision=DATASET_REVISION_C,
                    dataset_download_workers=9,
                )
            load_emilia.assert_called_once_with(
                "amphion/Emilia-Dataset",
                data_files={"train": "Emilia/en/**/*.tar"},
                split="train",
                streaming=True,
                revision=DATASET_REVISION_C,
                num_proc=9,
            )


if __name__ == "__main__":
    unittest.main()
