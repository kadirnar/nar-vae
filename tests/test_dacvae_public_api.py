"""Regression tests for the raw bundled DACVAE API's local-only boundary."""

import unittest
from importlib.util import find_spec
from unittest.mock import patch


@unittest.skipUnless(find_spec("audiotools") is not None, "bundled codec dependency not installed")
class BundledDACVAEPublicAPITest(unittest.TestCase):
    def test_raw_class_does_not_download_hub_shaped_paths(self):
        from nar_vae.dacvae import DACVAE

        with (
            patch("huggingface_hub.hf_hub_download") as download,
            self.assertRaises((FileNotFoundError, ValueError)),
        ):
            DACVAE.load("facebook/unpinned-codec")

        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
