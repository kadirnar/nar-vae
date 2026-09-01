"""Network-free tests for immutable prepared-dataset content identities."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nar_vae.dataset.identity import (
    PREPARED_DATASET_MANIFEST_FILENAME,
    DatasetIdentityError,
    resolve_hub_dataset_identity,
    resolve_local_prepared_dataset_identity,
    write_prepared_dataset_manifest,
)

try:
    from datasets import Dataset, load_from_disk
except ImportError:  # pragma: no cover - lightweight inference environment
    Dataset = None
    load_from_disk = None


@unittest.skipIf(Dataset is None, "datasets dependency is unavailable")
class LocalPreparedDatasetIdentityTest(unittest.TestCase):
    def _save(self, directory: Path, values: list[int]):
        dataset = Dataset.from_dict(
            {
                "value": values,
                "conditioning_ids": [[value, value + 1] for value in values],
            }
        )
        dataset.save_to_disk(directory)
        manifest = write_prepared_dataset_manifest(dataset, directory)
        return dataset, manifest

    def test_saved_artifact_inventory_binds_rows_and_persisted_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_dataset, first_manifest = self._save(root / "first", [1, 2])
            second_dataset, second_manifest = self._save(root / "second", [1, 3])
            _, repeated_manifest = self._save(root / "repeated", [1, 2])

            first = resolve_local_prepared_dataset_identity(
                load_from_disk(root / "first"),
                root / "first",
            )
            second = resolve_local_prepared_dataset_identity(second_dataset, root / "second")

            self.assertNotEqual(first["content_sha256"], second["content_sha256"])
            self.assertNotEqual(
                first_manifest["artifact_sha256"], second_manifest["artifact_sha256"]
            )
            self.assertEqual(first_manifest, repeated_manifest)
            self.assertEqual(first["num_rows"], 2)
            self.assertEqual(first["columns"], ["value", "conditioning_ids"])
            self.assertEqual(first["fingerprint"], first_dataset._fingerprint)
            self.assertTrue((root / "first" / PREPARED_DATASET_MANIFEST_FILENAME).is_file())

    def test_hub_identity_hashes_prepared_row_bytes_not_only_runtime_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._save(root / "first", [1, 2])
            self._save(root / "second", [1, 3])
            first_dataset = load_from_disk(root / "first")
            second_dataset = load_from_disk(root / "second")
            first_dataset._fingerprint = "same-runtime-fingerprint"
            second_dataset._fingerprint = "same-runtime-fingerprint"

            first = resolve_hub_dataset_identity(
                first_dataset,
                repo_id="owner/prepared-data",
                revision="a" * 40,
                split="train",
                snapshot_dir=root / "first",
            )
            second = resolve_hub_dataset_identity(
                second_dataset,
                repo_id="owner/prepared-data",
                revision="a" * 40,
                split="train",
                snapshot_dir=root / "second",
            )

            self.assertNotEqual(first["content_sha256"], second["content_sha256"])

    def test_modified_or_uninventoried_prepared_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            dataset, _ = self._save(root, [1, 2])
            arrow = next(root.glob("*.arrow"))
            arrow.write_bytes(arrow.read_bytes() + b"mutation")

            with self.assertRaisesRegex(DatasetIdentityError, "does not match its SHA-256"):
                resolve_local_prepared_dataset_identity(dataset, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            dataset, _ = self._save(root, [1, 2])
            (root / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DatasetIdentityError, "inventory has changed"):
                resolve_local_prepared_dataset_identity(dataset, root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            dataset, _ = self._save(root, [1, 2])
            nested = root / "unbound"
            nested.mkdir()
            (nested / PREPARED_DATASET_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DatasetIdentityError, "inventory has changed"):
                resolve_local_prepared_dataset_identity(dataset, root)

    def test_manifest_metadata_must_match_the_loaded_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            self._save(root, [1, 2])
            loaded = load_from_disk(root)
            manifest_path = root / PREPARED_DATASET_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["num_rows"] = 3
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DatasetIdentityError, "row count"):
                resolve_local_prepared_dataset_identity(loaded, root)

    def test_local_training_rejects_an_unmanifested_legacy_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            dataset = Dataset.from_dict({"value": [1]})
            dataset.save_to_disk(root)

            with self.assertRaisesRegex(DatasetIdentityError, "manifest; missing"):
                resolve_local_prepared_dataset_identity(dataset, root)


class HubDatasetIdentityTest(unittest.TestCase):
    def test_mutable_hub_revision_is_rejected(self):
        class FakeDataset:
            column_names = ["value"]
            _fingerprint = None

            def __len__(self):
                return 1

        with self.assertRaisesRegex(DatasetIdentityError, "40-character commit"):
            resolve_hub_dataset_identity(
                FakeDataset(),
                repo_id="owner/prepared-data",
                revision="main",
                split="train",
                snapshot_dir=".",
            )


if __name__ == "__main__":
    unittest.main()
