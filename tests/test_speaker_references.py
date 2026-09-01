"""Tests for leak-free speaker reference selection."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np

from nar_vae.dataset.prepare_dataset import save_dataset
from nar_vae.dataset.representation import REPRESENTATION_CONTRACT_COLUMN
from nar_vae.dataset.speaker_references import (
    build_speaker_index,
    select_reference_indices,
    validate_zero_shot_splits,
)


class FakeDataset:
    column_names = ["speaker_id"]

    def __init__(self):
        self.rows = [{"speaker_id": "a"}, {"speaker_id": "a"}, {"speaker_id": "b"}]

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


class ManifestDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]


def manifest_split(prefix, speakers):
    return ManifestDataset(
        [
            {
                "utterance_id": f"{prefix}-{speaker}-{session}",
                "speaker_id": speaker,
                "session_id": session,
                "language": language,
            }
            for speaker, languages in speakers.items()
            for session, language in enumerate(languages, start=1)
        ]
    )


class SpeakerReferenceTest(unittest.TestCase):
    def test_target_utterance_is_never_selected(self):
        dataset = FakeDataset()
        index = build_speaker_index(dataset, "speaker_id")

        self.assertEqual(
            select_reference_indices(
                index,
                speaker_id="a",
                target_index=0,
                maximum_utterances=3,
                seed=1,
            ),
            [1],
        )
        self.assertEqual(
            select_reference_indices(
                index,
                speaker_id="b",
                target_index=2,
                maximum_utterances=3,
                seed=1,
            ),
            [],
        )

    def test_reference_selection_can_require_a_different_session(self):
        speaker_index = {"a": [0, 1, 2]}

        self.assertEqual(
            select_reference_indices(
                speaker_index,
                speaker_id="a",
                target_index=0,
                maximum_utterances=3,
                seed=1,
                session_ids=["studio-a", "studio-a", "studio-b"],
                target_session_id="studio-a",
                require_different_session=True,
            ),
            [2],
        )

    def test_zero_shot_splits_require_disjoint_speakers_and_cross_session_data(self):
        splits = {
            "train": manifest_split("train", {"train-a": ["en", "es"]}),
            "validation": manifest_split("validation", {"dev-a": ["en", "en"]}),
            "test": manifest_split("test", {"test-a": ["tr", "en"]}),
        }

        summary = validate_zero_shot_splits(
            splits,
            utterance_id_column="utterance_id",
            session_id_column="session_id",
            require_cross_session_references=True,
        )

        self.assertEqual(summary["utterances"], 6)
        self.assertEqual(summary["speakers"], 3)
        self.assertEqual(summary["languages"], ("en", "es", "tr"))
        self.assertTrue(summary["cross_session_references"])

    def test_zero_shot_splits_reject_speaker_leakage(self):
        splits = {
            "train": manifest_split("train", {"shared": ["en", "es"]}),
            "validation": manifest_split("validation", {"dev-a": ["en", "en"]}),
            "test": manifest_split("test", {"shared": ["tr", "en"]}),
        }

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_zero_shot_splits(splits)

    def test_zero_shot_splits_reject_speakers_without_a_reference_peer(self):
        splits = {
            "train": manifest_split("train", {"train-a": ["en"]}),
            "validation": manifest_split("validation", {"dev-a": ["en", "en"]}),
            "test": manifest_split("test", {"test-a": ["tr", "en"]}),
        }

        with self.assertRaisesRegex(ValueError, "at least two utterances"):
            validate_zero_shot_splits(splits)

    def test_zero_shot_splits_reject_single_session_speakers_when_required(self):
        splits = {
            "train": manifest_split("train", {"train-a": ["en", "es"]}),
            "validation": manifest_split("validation", {"dev-a": ["en", "en"]}),
            "test": manifest_split("test", {"test-a": ["tr", "en"]}),
        }
        splits["train"].rows[1]["session_id"] = 1

        with self.assertRaisesRegex(ValueError, "at least two sessions"):
            validate_zero_shot_splits(
                splits,
                session_id_column="session_id",
                require_cross_session_references=True,
            )

    def test_prepared_dataset_preserves_speaker_latents(self):
        sample = {
            "latents": np.zeros((2, 3), dtype=np.float32),
            "latent_num_frames": 3,
            "conditioning_ids": [1],
            "speaker_latents": np.zeros((2, 4), dtype=np.float32),
            REPRESENTATION_CONTRACT_COLUMN: {"contract_version": 1},
        }
        dataset = MagicMock()
        dataset.__len__.return_value = 1

        with TemporaryDirectory() as directory:
            with (
                patch(
                    "nar_vae.dataset.prepare_dataset.Dataset.from_dict",
                    return_value=dataset,
                ) as from_dict,
                patch(
                    "nar_vae.dataset.prepare_dataset.write_prepared_dataset_manifest"
                ) as write_manifest,
            ):
                save_dataset([sample], directory)

        self.assertIn("speaker_latents", from_dict.call_args.args[0])
        self.assertEqual(from_dict.call_args.args[0]["latent_num_frames"], [3])
        self.assertIn(REPRESENTATION_CONTRACT_COLUMN, from_dict.call_args.args[0])
        dataset.save_to_disk.assert_called_once()
        write_manifest.assert_called_once_with(dataset, directory)

    def test_prepared_dataset_rejects_partial_frame_metadata(self):
        samples = [
            {
                "latents": np.zeros((2, 3), dtype=np.float32),
                "latent_num_frames": 3,
                "conditioning_ids": [1],
            },
            {
                "latents": np.zeros((2, 2), dtype=np.float32),
                "conditioning_ids": [2],
            },
        ]

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "latent_num_frames must be present"):
                save_dataset(samples, directory)

    def test_legacy_rows_require_an_explicit_save_opt_in(self):
        sample = {
            "latents": np.zeros((2, 3), dtype=np.float32),
            "latent_num_frames": 3,
            "conditioning_ids": [1],
        }
        dataset = MagicMock()
        dataset.__len__.return_value = 1

        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "allow_legacy_representation=True"):
                save_dataset([sample], directory)

            with (
                patch(
                    "nar_vae.dataset.prepare_dataset.Dataset.from_dict",
                    return_value=dataset,
                ),
                patch(
                    "nar_vae.dataset.prepare_dataset.write_prepared_dataset_manifest"
                ) as write_manifest,
            ):
                save_dataset(
                    [sample],
                    directory,
                    allow_legacy_representation=True,
                )

        dataset.save_to_disk.assert_called_once()
        write_manifest.assert_called_once_with(dataset, directory)


if __name__ == "__main__":
    unittest.main()
