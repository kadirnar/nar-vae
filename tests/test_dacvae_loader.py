"""Tests for DACVAE backend selection and metadata-aware loading."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_backend_module():
    module_path = Path(__file__).parents[1] / "nar_vae" / "dacvae" / "loader.py"
    spec = importlib.util.spec_from_file_location("nar_vae_dacvae_loader_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loader = load_backend_module()

MODEL_KWARGS = {
    "encoder_dim": 64,
    "encoder_rates": [2, 8, 10, 12],
    "latent_dim": 1024,
    "decoder_dim": 1536,
    "decoder_rates": [12, 10, 8, 2],
    "n_codebooks": 16,
    "codebook_size": 1024,
    "codebook_dim": 128,
    "quantizer_dropout": False,
    "sample_rate": 48000,
}


class FakeFastDACVAE:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.hop_length = 1920
        self.quantizer = types.SimpleNamespace(out_proj=types.SimpleNamespace(in_channels=128))
        self.loaded_state_dict = None
        self.metadata = None

    def load_state_dict(self, state_dict, strict):
        self.loaded_state_dict = state_dict
        self.strict = strict


class DACVAELoaderTest(unittest.TestCase):
    def test_invalid_backend_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown DACVAE backend"):
            loader.resolve_dacvae_backend("unknown")

    def test_auto_falls_back_to_bundled(self):
        with patch.object(loader, "_installed_fast_dacvae_distribution", return_value=None):
            self.assertEqual(loader.resolve_dacvae_backend("auto"), "bundled")

    def test_explicit_fast_backend_rejects_a_missing_install_without_network(self):
        with (
            patch.object(loader, "_installed_fast_dacvae_distribution", return_value=None),
            self.assertRaisesRegex(ImportError, "is not installed"),
        ):
            loader.resolve_dacvae_backend("fast")

    def test_fast_backend_accepts_exact_vcs_provenance_without_network(self):
        direct_url = json.dumps(
            {
                "url": loader.FAST_DACVAE_SOURCE_URL,
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": loader.FAST_DACVAE_REVISION,
                    "requested_revision": loader.FAST_DACVAE_REVISION,
                },
            }
        )
        installed_distribution = types.SimpleNamespace(
            read_text=lambda filename: direct_url if filename == "direct_url.json" else None
        )
        with (
            patch.object(
                loader,
                "_installed_fast_dacvae_distribution",
                return_value=installed_distribution,
            ),
            patch.object(loader.util, "find_spec", return_value=object()),
        ):
            self.assertEqual(loader.resolve_dacvae_backend("fast"), "fast")
            self.assertTrue(loader.is_fast_dacvae_available())

    def test_fast_backend_rejects_missing_vcs_provenance_without_network(self):
        installed_distribution = types.SimpleNamespace(read_text=lambda filename: None)
        with (
            patch.object(
                loader,
                "_installed_fast_dacvae_distribution",
                return_value=installed_distribution,
            ),
            patch.object(loader.util, "find_spec", return_value=object()),
            self.assertRaisesRegex(ImportError, "direct_url.json"),
        ):
            loader.resolve_dacvae_backend("fast")

    def test_fast_backend_rejects_mismatched_vcs_commit_without_network(self):
        direct_url = json.dumps(
            {
                "url": loader.FAST_DACVAE_SOURCE_URL,
                "vcs_info": {"vcs": "git", "commit_id": "0" * 40},
            }
        )
        installed_distribution = types.SimpleNamespace(read_text=lambda filename: direct_url)
        with (
            patch.object(
                loader,
                "_installed_fast_dacvae_distribution",
                return_value=installed_distribution,
            ),
            patch.object(loader.util, "find_spec", return_value=object()),
            self.assertRaisesRegex(ImportError, "not the reviewed commit"),
        ):
            loader.resolve_dacvae_backend("auto")

    def test_fast_backend_rejects_mismatched_vcs_source_without_network(self):
        direct_url = json.dumps(
            {
                "url": "https://example.invalid/fast-dacvae.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": loader.FAST_DACVAE_REVISION,
                },
            }
        )
        installed_distribution = types.SimpleNamespace(read_text=lambda filename: direct_url)
        with (
            patch.object(
                loader,
                "_installed_fast_dacvae_distribution",
                return_value=installed_distribution,
            ),
            patch.object(loader.util, "find_spec", return_value=object()),
            self.assertRaisesRegex(ImportError, "reviewed source URL"),
        ):
            loader.resolve_dacvae_backend("fast")

    def test_bundled_backend_does_not_inspect_fast_backend_provenance(self):
        with patch.object(
            loader,
            "_installed_fast_dacvae_distribution",
            side_effect=AssertionError("bundled backend inspected fast-dacvae"),
        ):
            self.assertEqual(loader.resolve_dacvae_backend("bundled"), "bundled")

    def test_fast_backend_rejects_conflicting_dacvae_namespace(self):
        official_dacvae = types.SimpleNamespace(DACVAE=object)
        with (
            patch.object(loader, "resolve_dacvae_backend", return_value="fast"),
            patch.object(loader, "import_module", return_value=official_dacvae),
            self.assertRaisesRegex(ImportError, "not provided by fast-dacvae"),
        ):
            loader.get_dacvae_class("fast")

    def test_fast_loader_uses_checkpoint_metadata_and_strict_weights(self):
        artifact = {
            "state_dict": {"encoder.weight": object()},
            "metadata": {"kwargs": MODEL_KWARGS},
        }
        fake_torch = types.SimpleNamespace(
            load=lambda *args, **kwargs: artifact,
        )

        with (
            patch.object(
                loader,
                "_resolve_fast_checkpoint",
                return_value=Path("weights.pth"),
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            model = loader._load_fast_dacvae(FakeFastDACVAE, "owner/model")

        self.assertEqual(model.kwargs, MODEL_KWARGS)
        self.assertTrue(model.strict)
        self.assertEqual(model.loaded_state_dict, artifact["state_dict"])

    def test_fast_loader_rejects_missing_metadata(self):
        artifact = {"state_dict": {"encoder.weight": object()}}
        fake_torch = types.SimpleNamespace(
            load=lambda *args, **kwargs: artifact,
        )

        with (
            patch.object(
                loader,
                "_resolve_fast_checkpoint",
                return_value=Path("weights.pth"),
            ),
            patch.dict(sys.modules, {"torch": fake_torch}),
            self.assertRaisesRegex(ValueError, "metadata.kwargs is missing"),
        ):
            loader._load_fast_dacvae(FakeFastDACVAE, "owner/model")

    def test_expected_latent_size_rejects_mismatch(self):
        fake_backend = types.SimpleNamespace(load=lambda path: FakeFastDACVAE())

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.touch()
            with (
                patch.object(loader, "get_dacvae_class", return_value=fake_backend),
                self.assertRaisesRegex(ValueError, "latent width does not match"),
            ):
                loader.load_dacvae(
                    checkpoint,
                    backend="bundled",
                    expected_latent_size=64,
                    verbose=False,
                )

    def test_unpinned_hub_id_fails_before_download(self):
        with (
            patch("huggingface_hub.hf_hub_download") as download,
            self.assertRaisesRegex(ValueError, "unpinned local source"),
        ):
            loader.load_dacvae("owner/model", backend="bundled", verbose=False)
        download.assert_not_called()

    def test_plain_hub_id_with_commit_downloads_default_artifact(self):
        class FakeCodec:
            sample_rate = 44100
            hop_length = 512
            quantizer = types.SimpleNamespace(out_proj=types.SimpleNamespace(in_channels=128))

            def eval(self):
                return self

        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            checkpoint.write_bytes(b"codec")
            backend = types.SimpleNamespace(load=lambda path: FakeCodec())
            with (
                patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)) as download,
                patch.object(loader, "get_dacvae_class", return_value=backend),
            ):
                codec = loader.load_dacvae(
                    "owner/model",
                    dacvae_revision=revision,
                    backend="bundled",
                    verbose=False,
                )

        download.assert_called_once_with(
            repo_id="owner/model",
            filename="weights.pth",
            revision=revision,
        )
        self.assertEqual(codec.nar_vae_codec_identifier, "owner/model")
        self.assertEqual(codec.nar_vae_codec_revision, revision)
        self.assertEqual(codec.nar_vae_codec_filename, "weights.pth")

    def test_plain_hub_id_rejects_mutable_revision_before_download(self):
        with (
            patch("huggingface_hub.hf_hub_download") as download,
            self.assertRaisesRegex(ValueError, "40-character"),
        ):
            loader.load_dacvae(
                "owner/model",
                dacvae_revision="main",
                backend="bundled",
                verbose=False,
            )
        download.assert_not_called()

    def test_explicit_local_path_rejects_hub_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, "only valid"):
                loader.load_dacvae(
                    checkpoint,
                    dacvae_revision="a" * 40,
                    backend="bundled",
                    verbose=False,
                )

    def test_codec_sha_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            checkpoint.write_bytes(b"codec bytes")
            with (
                patch.object(loader, "get_dacvae_class") as backend,
                self.assertRaisesRegex(ValueError, "representation SHA-256"),
            ):
                loader.load_dacvae(
                    checkpoint,
                    backend="bundled",
                    expected_sha256="0" * 64,
                    verbose=False,
                )
        backend.assert_not_called()

    def test_typed_hub_codec_download_is_revision_pinned(self):
        class FakeCodec:
            sample_rate = 44100
            hop_length = 512
            quantizer = types.SimpleNamespace(out_proj=types.SimpleNamespace(in_channels=128))

            def eval(self):
                return self

            def requires_grad_(self, value):
                self.frozen = not value
                return self

        revision = "a" * 40
        source = loader.HubDACVAESource("owner/model", revision, "codec/weights.pth")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pth"
            checkpoint.touch()
            backend = types.SimpleNamespace(load=lambda path: FakeCodec())
            with (
                patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)) as download,
                patch.object(loader, "get_dacvae_class", return_value=backend),
            ):
                codec = loader.load_dacvae(
                    source,
                    backend="bundled",
                    freeze=True,
                    verbose=False,
                )

        download.assert_called_once_with(
            repo_id="owner/model",
            filename="codec/weights.pth",
            revision=revision,
        )
        self.assertEqual(codec.nar_vae_codec_identifier, "owner/model")
        self.assertEqual(codec.nar_vae_codec_revision, revision)
        self.assertEqual(codec.nar_vae_codec_filename, "codec/weights.pth")

    def test_typed_hub_codec_rejects_a_mismatched_cache_commit(self):
        source = loader.HubDACVAESource("owner/model", "a" * 40)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "snapshots" / ("b" * 40) / "weights.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            with (
                patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)),
                self.assertRaisesRegex(RuntimeError, "different DACVAE commit"),
            ):
                loader.load_dacvae(source, backend="bundled", verbose=False)


if __name__ == "__main__":
    unittest.main()
