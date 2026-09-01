"""Network-free contracts for acoustic model and representation manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nar_vae.checkpoint import (
    CheckpointProvenance,
    DurationCheckpointInfo,
    LanguageCheckpointInfo,
    MonotonicAlignmentCheckpointInfo,
    ReferenceLanguageCheckpointInfo,
)
from nar_vae.dataset.representation import TEXT_FRONTEND_NAME
from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.model_manifest import (
    MODEL_MANIFEST_FILENAME,
    ModelManifestError,
    load_model_manifest,
    validate_inference_manifest,
    validate_loaded_codec,
    validate_manifest_weight,
    validate_sft_parent_manifest,
    write_model_manifest,
)


def model_config(codec_source: str) -> dict:
    return {
        "model_preset": "tiny",
        "dacvae_model": codec_source,
        "dacvae_backend": "bundled",
        "dacvae_sample_rate": 44100,
        "dacvae_hop_length": 512,
        "dacvae_latent_dim": 128,
        "dacvae_sha256": "c" * 64,
        "text_vocab_size": 100312,
        "speaker_patch_size": 4,
        "norm_eps": 1e-6,
        "use_speaker_conditioning": False,
        "use_language_conditioning": False,
        "supported_languages": ["en"],
        "supported_reference_languages": None,
        "use_duration_predictor": True,
        "duration_predictor_hidden_size": 256,
        "duration_predictor_num_layers": 2,
        "duration_predictor_use_speaker": False,
        "use_mas_duration": True,
        "duration_alignment_hidden_size": 64,
    }


class ModelManifestTest(unittest.TestCase):
    def _write(self, root: Path, config: dict | None = None):
        weights = root / "pytorch_model.bin"
        weights.write_bytes(b"model weights")
        manifest = write_model_manifest(
            root,
            config or model_config("./codec/weights.pth"),
            stage="pretrain",
            checkpoint_files=(weights.name,),
        )
        return weights, manifest

    def test_export_binds_weight_shape_capabilities_and_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, written = self._write(root)
            loaded = load_model_manifest(root / MODEL_MANIFEST_FILENAME)

            self.assertEqual(loaded.raw, written.raw)
            self.assertEqual(loaded.stage, "pretrain")
            self.assertEqual(loaded.architecture["latent_size"], 128)
            self.assertTrue(loaded.capabilities["monotonic_alignment"])
            self.assertEqual(loaded.representation["text_frontend_name"], TEXT_FRONTEND_NAME)
            self.assertEqual(loaded.representation["codec_source"], "./codec/weights.pth")
            self.assertIsNone(loaded.representation["codec_revision"])
            validate_manifest_weight(loaded, weights)

            weights.write_bytes(b"different model weights")
            with self.assertRaisesRegex(ModelManifestError, "manifest SHA-256"):
                validate_manifest_weight(loaded, weights)

    def test_manifest_v2_records_exact_language_pairs_authoritatively(self):
        config = model_config("./codec/weights.pth")
        config.update(
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=["en", "es"],
            supported_reference_languages=["ja"],
            supported_language_pairs=[["Spanish", "English"], ["en", "Japanese"]],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root, config)

            self.assertEqual(manifest.raw["schema_version"], 2)
            self.assertEqual(
                manifest.capabilities["supported_language_pairs"],
                [["es", "en"], ["en", "ja"]],
            )
            self.assertEqual(
                manifest.capabilities["supported_reference_languages"],
                ["en", "ja"],
            )

            raw = dict(manifest.raw)
            raw["schema_version"] = 1
            (root / MODEL_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ModelManifestError, "Unsupported.*schema"):
                load_model_manifest(root / MODEL_MANIFEST_FILENAME)

    def test_manifest_parser_rejects_multilingual_speaker_topology_without_exact_pairs(self):
        config = model_config("./codec/weights.pth")
        config.update(
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=["en", "es"],
            supported_language_pairs=[["en", "en"], ["es", "es"]],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root, config)
            raw = json.loads(json.dumps(manifest.raw))
            raw["capabilities"]["supported_reference_languages"] = []
            raw["capabilities"]["supported_language_pairs"] = []
            manifest_path = root / MODEL_MANIFEST_FILENAME
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")

            with self.assertRaisesRegex(
                ModelManifestError,
                "require exact supported_language_pairs",
            ):
                load_model_manifest(manifest_path)

    def test_manifest_parser_allows_pairless_single_conditioning_topologies(self):
        configurations = {
            "speaker-only": {
                "use_speaker_conditioning": True,
            },
            "language-only": {
                "use_language_conditioning": True,
                "supported_languages": ["en", "es"],
            },
        }
        for name, overrides in configurations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                config = model_config("./codec/weights.pth")
                config.update(overrides)
                root = Path(directory)
                _, manifest = self._write(root, config)

                loaded = load_model_manifest(root / MODEL_MANIFEST_FILENAME)

                self.assertEqual(loaded.capabilities["supported_language_pairs"], [])
                self.assertEqual(loaded.raw, manifest.raw)

    def test_hub_shaped_codec_requires_commit_filename_and_sha(self):
        unpinned = model_config("owner/codec")
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "pytorch_model.bin"
            weights.write_bytes(b"weights")
            with self.assertRaisesRegex(ModelManifestError, "unpinned local source"):
                write_model_manifest(
                    directory,
                    unpinned,
                    stage="pretrain",
                    checkpoint_files=(weights.name,),
                )

            pinned = dict(
                unpinned,
                dacvae_revision="d" * 40,
                dacvae_filename="codec/weights.pth",
            )
            manifest = write_model_manifest(
                directory,
                pinned,
                stage="pretrain",
                checkpoint_files=(weights.name,),
            )
            self.assertEqual(manifest.representation["codec_revision"], "d" * 40)
            self.assertEqual(manifest.representation["codec_filename"], "codec/weights.pth")

    def test_sft_preserves_parent_architecture_and_codec_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, parent = self._write(root, model_config("./codec/a.pth"))

            self.assertEqual(
                validate_sft_parent_manifest(weights, model_config("./codec/a.pth")).sha256,
                parent.sha256,
            )
            with self.assertRaisesRegex(ModelManifestError, "codec/frontend"):
                validate_sft_parent_manifest(weights, model_config("./codec/b.pth"))

            changed_shape = model_config("./codec/a.pth")
            changed_shape["speaker_patch_size"] = 8
            with self.assertRaisesRegex(ModelManifestError, "architecture"):
                validate_sft_parent_manifest(weights, changed_shape)

            speaker_migration = model_config("./codec/a.pth")
            speaker_migration["use_speaker_conditioning"] = True
            with self.assertRaisesRegex(ModelManifestError, "architecture"):
                validate_sft_parent_manifest(weights, speaker_migration)
            speaker_migration["initialize_speaker_conditioning"] = True
            validate_sft_parent_manifest(weights, speaker_migration)

    def test_sft_rejects_language_and_cross_lingual_capability_downgrades(self):
        multilingual = model_config("./codec/a.pth")
        multilingual.update(
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=["en", "es"],
            supported_reference_languages=["en", "es"],
        )
        with tempfile.TemporaryDirectory() as directory:
            weights, _ = self._write(Path(directory), multilingual)

            no_language = dict(multilingual)
            no_language.update(
                use_language_conditioning=False,
                supported_languages=["en"],
                supported_reference_languages=None,
            )
            with self.assertRaisesRegex(ModelManifestError, "cannot remove"):
                validate_sft_parent_manifest(weights, no_language)

            changed_registry = dict(multilingual)
            changed_registry["supported_languages"] = ["en", "de"]
            with self.assertRaisesRegex(ModelManifestError, "cannot remove"):
                validate_sft_parent_manifest(weights, changed_registry)

    def test_sft_allows_only_explicit_additive_language_migrations(self):
        parent_config = model_config("./codec/a.pth")
        with tempfile.TemporaryDirectory() as directory:
            weights, _ = self._write(Path(directory), parent_config)
            multilingual = dict(parent_config)
            multilingual.update(
                use_language_conditioning=True,
                supported_languages=["en", "es"],
                initialize_language_conditioning=True,
            )

            validate_sft_parent_manifest(weights, multilingual)

            multilingual["initialize_language_conditioning"] = False
            with self.assertRaisesRegex(ModelManifestError, "explicit validated additive"):
                validate_sft_parent_manifest(weights, multilingual)

    def test_inference_rejects_same_width_wrong_codec_and_wrong_frontend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, manifest = self._write(root, model_config("./codec/a.pth"))

            with self.assertRaisesRegex(ModelManifestError, "DACVAE source"):
                validate_inference_manifest(
                    manifest,
                    checkpoint_path=weights,
                    selected_filename=weights.name,
                    architecture=manifest.architecture,
                    capabilities=manifest.capabilities,
                    codec_source="./codec/b.pth",
                    codec_backend="bundled",
                )

            raw = json.loads((root / MODEL_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            raw["representation"]["text_frontend_name"] = "another/frontend"
            (root / MODEL_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
            incompatible = load_model_manifest(root / MODEL_MANIFEST_FILENAME)
            with self.assertRaisesRegex(ModelManifestError, "text frontend"):
                validate_inference_manifest(
                    incompatible,
                    checkpoint_path=weights,
                    selected_filename=weights.name,
                    architecture=incompatible.architecture,
                    capabilities=incompatible.capabilities,
                    codec_source="./codec/a.pth",
                    codec_backend="bundled",
                )

    def test_partial_ema_inference_also_hashes_the_loaded_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "pytorch_model.bin"
            ema = root / "pytorch_model_ema.bin"
            base.write_bytes(b"full base")
            ema.write_bytes(b"sparse ema")
            manifest = write_model_manifest(
                root,
                model_config("./codec/a.pth"),
                stage="sft",
                checkpoint_files=(base.name, ema.name),
                parent_manifest={
                    "manifest_sha256": "a" * 64,
                    "stage": "pretrain",
                    "weights_sha256": "b" * 64,
                    "representation_sha256": "c" * 64,
                },
            )
            base.write_bytes(b"tampered full base")

            with self.assertRaisesRegex(ModelManifestError, "manifest SHA-256"):
                validate_inference_manifest(
                    manifest,
                    checkpoint_path=ema,
                    selected_filename=ema.name,
                    base_checkpoint_path=base,
                    base_filename=base.name,
                    architecture=manifest.architecture,
                    capabilities=manifest.capabilities,
                    codec_source="./codec/a.pth",
                    codec_backend="bundled",
                )

    def test_loaded_codec_checks_source_backend_rate_hop_and_width(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            codec = SimpleNamespace(
                nar_vae_codec_identifier="./codec/weights.pth",
                nar_vae_backend="bundled",
                nar_vae_codec_revision=None,
                nar_vae_codec_filename=None,
                nar_vae_codec_sha256="c" * 64,
                sample_rate=44100,
                hop_length=512,
                quantizer=SimpleNamespace(
                    out_proj=SimpleNamespace(in_channels=128),
                ),
            )
            validate_loaded_codec(manifest, codec)
            codec.hop_length = 256
            with self.assertRaisesRegex(ModelManifestError, "Loaded DACVAE facts"):
                validate_loaded_codec(manifest, codec)

    def test_inference_rejects_a_missing_manifest_before_loading_codec(self):
        checkpoint = Mock()
        checkpoint.path = Path("checkpoint.bin")
        checkpoint.provenance = CheckpointProvenance(
            kind="local",
            source="checkpoint.bin",
            requested_revision=None,
            resolved_revision=None,
            base_filename="checkpoint.bin",
            ema_filename=None,
            selected_filename="checkpoint.bin",
            path=Path("checkpoint.bin"),
            base_path=Path("checkpoint.bin"),
            manifest_filename=MODEL_MANIFEST_FILENAME,
            manifest_path=Path(MODEL_MANIFEST_FILENAME),
        )
        checkpoint.infer_text_vocab_size.return_value = 100312
        checkpoint.infer_speaker_conditioning.return_value = False
        checkpoint.language_capability.return_value = LanguageCheckpointInfo(False)
        checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(
            False
        )
        checkpoint.duration_capability.return_value = DurationCheckpointInfo(False)
        checkpoint.monotonic_alignment_capability.return_value = MonotonicAlignmentCheckpointInfo(
            False
        )
        with (
            patch("nar_vae.inference.FlowCheckpoint.load", return_value=checkpoint),
            patch("nar_vae.inference.load_dacvae") as implicit_codec,
            self.assertRaisesRegex(ValueError, "dacvae_model is required"),
        ):
            FlowMatchingTTSInference("checkpoint.bin", device="cpu")
        implicit_codec.assert_not_called()

        with (
            patch("nar_vae.inference.FlowCheckpoint.load", return_value=checkpoint),
            patch(
                "nar_vae.inference.load_model_manifest",
                side_effect=ModelManifestError("missing manifest"),
            ),
            patch("nar_vae.inference.load_dacvae") as load_codec,
            self.assertRaisesRegex(ModelManifestError, "missing manifest"),
        ):
            FlowMatchingTTSInference(
                "checkpoint.bin",
                dacvae_model="codec.pth",
                device="cpu",
            )
        load_codec.assert_not_called()

    def test_inference_rejects_missing_manifest_before_deserializing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "pytorch_model.bin"
            checkpoint.write_bytes(b"not a serialized checkpoint")
            with (
                patch("nar_vae.checkpoint.torch.load") as deserialize,
                self.assertRaisesRegex(ModelManifestError, "missing"),
            ):
                FlowMatchingTTSInference(
                    checkpoint,
                    dacvae_model="codec.pth",
                    device="cpu",
                )
            deserialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
