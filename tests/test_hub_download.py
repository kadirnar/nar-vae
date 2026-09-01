"""Tests for the packaged Hugging Face downloader."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from vyvotts.hub import (
    DEFAULT_IGNORE_PATTERNS,
    download_snapshot,
    resolve_revision,
)


class HubDownloadTest(unittest.TestCase):
    REPO_ID = "owner/model"
    REVISION = "a" * 40

    def test_snapshot_revision_must_be_a_full_commit(self):
        self.assertEqual(
            resolve_revision(self.REPO_ID, self.REVISION),
            self.REVISION,
        )
        with self.assertRaisesRegex(ValueError, "explicit 40-character"):
            resolve_revision(self.REPO_ID, None)
        with self.assertRaisesRegex(ValueError, "explicit 40-character"):
            resolve_revision("owner/another-model", "main")

    @patch("vyvotts.hub.get_token")
    @patch("vyvotts.hub.snapshot_download")
    def test_download_uses_environment_token_and_defaults(self, snapshot_download, get_token):
        snapshot_download.return_value = "/tmp/downloaded"

        with patch.dict(os.environ, {"HF_TOKEN": "environment-token"}):
            result = download_snapshot(
                repo_id=self.REPO_ID,
                repo_type="model",
                local_dir=Path("target"),
                revision=self.REVISION,
            )

        self.assertEqual(result, Path("/tmp/downloaded"))
        get_token.assert_not_called()
        snapshot_download.assert_called_once_with(
            repo_id=self.REPO_ID,
            repo_type="model",
            revision=self.REVISION,
            local_dir="target",
            ignore_patterns=list(DEFAULT_IGNORE_PATTERNS),
            allow_patterns=None,
            max_workers=8,
            token="environment-token",
        )

    @patch("vyvotts.hub.get_token", return_value="cached-token")
    @patch("vyvotts.hub.snapshot_download")
    def test_explicit_patterns_and_cached_token(self, snapshot_download, get_token):
        snapshot_download.return_value = "/tmp/downloaded"

        with patch.dict(os.environ, {"HF_TOKEN": ""}):
            download_snapshot(
                repo_id="owner/dataset",
                repo_type="dataset",
                revision="a" * 40,
                local_dir="target",
                ignore_patterns=[],
                allow_patterns=["data/*.parquet"],
                max_workers=2,
            )

        get_token.assert_called_once_with()
        snapshot_download.assert_called_once_with(
            repo_id="owner/dataset",
            repo_type="dataset",
            revision="a" * 40,
            local_dir="target",
            ignore_patterns=[],
            allow_patterns=["data/*.parquet"],
            max_workers=2,
            token="cached-token",
        )

    def test_rejects_non_positive_worker_count(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            download_snapshot(
                repo_id=self.REPO_ID,
                repo_type="model",
                local_dir="target",
                max_workers=0,
            )

    @patch("vyvotts.hub.get_token")
    @patch("vyvotts.hub.snapshot_download")
    def test_missing_revision_fails_before_network(self, snapshot_download, get_token):
        with self.assertRaisesRegex(ValueError, "explicit 40-character"):
            download_snapshot(
                repo_id="owner/model",
                repo_type="model",
                local_dir="target",
            )

        get_token.assert_not_called()
        snapshot_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
