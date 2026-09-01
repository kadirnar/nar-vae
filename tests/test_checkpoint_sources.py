"""Network-free contracts for local and pinned remote checkpoint sources."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import torch

from vyvotts.checkpoint import FlowCheckpoint, HubCheckpointSource


class CheckpointSourceTest(unittest.TestCase):
    def test_hub_source_rejects_missing_revision_before_network(self):
        with patch("huggingface_hub.hf_hub_download") as download:
            with self.assertRaisesRegex(ValueError, "40-character Hub commit"):
                HubCheckpointSource(
                    repo_id="owner/model",
                    revision="",
                    base_filename="weights/base.bin",
                    ema_filename="weights/ema.bin",
                )

        download.assert_not_called()

    def test_plain_owner_name_is_a_missing_local_path_without_network(self):
        with patch("huggingface_hub.hf_hub_download") as download:
            with self.assertRaisesRegex(FileNotFoundError, "use HubCheckpointSource"):
                FlowCheckpoint.load("missing-nar-vae-owner/missing-nar-vae-model")

        download.assert_not_called()

    def test_hub_source_downloads_base_and_ema_at_the_pinned_revision(self):
        revision = "a" * 40
        source = HubCheckpointSource(
            repo_id="owner/model",
            revision=revision,
            base_filename="weights/base.bin",
            ema_filename="weights/ema.bin",
        )
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "models--owner--model" / "snapshots" / revision
            base_path = snapshot / source.base_filename
            ema_path = snapshot / source.ema_filename
            manifest_path = snapshot / source.manifest_filename
            base_path.parent.mkdir(parents=True)
            torch.save({"weight": torch.zeros(1)}, base_path)
            torch.save({"shadow": {"weight": torch.ones(1)}}, ema_path)
            manifest_path.write_text("{}", encoding="utf-8")

            def download(*, repo_id, filename, revision):
                self.assertEqual(repo_id, source.repo_id)
                if filename == source.base_filename:
                    return base_path
                if filename == source.ema_filename:
                    return ema_path
                return manifest_path

            with patch("huggingface_hub.hf_hub_download", side_effect=download) as mocked:
                checkpoint = FlowCheckpoint.load(source)

        self.assertEqual(
            mocked.call_args_list,
            [
                call(
                    repo_id=source.repo_id,
                    filename=source.base_filename,
                    revision=revision,
                ),
                call(
                    repo_id=source.repo_id,
                    filename=source.ema_filename,
                    revision=revision,
                ),
                call(
                    repo_id=source.repo_id,
                    filename=source.manifest_filename,
                    revision=revision,
                ),
            ],
        )
        self.assertTrue(checkpoint.is_ema)
        self.assertIsNotNone(checkpoint.base_state_dict)
        self.assertIsNotNone(checkpoint.provenance)
        provenance = checkpoint.provenance
        assert provenance is not None
        self.assertEqual(provenance.kind, "huggingface_hub")
        self.assertEqual(provenance.source, source.repo_id)
        self.assertEqual(provenance.requested_revision, revision)
        self.assertEqual(provenance.resolved_revision, revision)
        self.assertEqual(provenance.commit, revision)
        self.assertEqual(provenance.base_filename, source.base_filename)
        self.assertEqual(provenance.ema_filename, source.ema_filename)
        self.assertEqual(provenance.selected_filename, source.ema_filename)
        self.assertEqual(provenance.path, ema_path.resolve())
        self.assertEqual(provenance.base_path, base_path.resolve())
        self.assertEqual(provenance.manifest_path, manifest_path.resolve())


if __name__ == "__main__":
    unittest.main()
