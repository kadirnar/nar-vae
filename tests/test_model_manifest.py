"""Network-free contracts for acoustic model and representation manifests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from nar_vae.checkpoint import (
    CheckpointProvenance,
    DurationCheckpointInfo,
    LanguageCheckpointInfo,
    LegacyArchitectureCheckpointError,
    MonotonicAlignmentCheckpointInfo,
    ReferenceLanguageCheckpointInfo,
)
from nar_vae.dacvae_encoding import DACVAE_POSTERIOR_SAMPLING_POLICY
from nar_vae.dataset.representation import (
    LEGACY_CL100K_TEXT_FRONTEND_NAME,
    LEGACY_CL100K_TEXT_FRONTEND_VERSION,
    TEXT_FRONTEND_NAME,
)
from nar_vae.frozen_text_provider import FROZEN_TEXT_REPRESENTATION_NAME
from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.model_manifest import (
    LEGACY_MODEL_MANIFEST_SCHEMA_VERSION,
    MODEL_MANIFEST_FILENAME,
    MODEL_MANIFEST_SCHEMA_VERSION,
    ORIGIN_MODEL_MANIFEST_SCHEMA_VERSION,
    PREVIOUS_MODEL_MANIFEST_SCHEMA_VERSION,
    ModelManifestError,
    load_model_manifest,
    validate_inference_manifest,
    validate_loaded_codec,
    validate_manifest_weight,
    validate_sft_parent_manifest,
    write_model_manifest,
)
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.post_training.nar_vae_stage import model_export_config_from_manifest


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_config(codec_source: str) -> dict:
    return {
        "model_preset": "tiny",
        "dacvae_model": codec_source,
        "dacvae_backend": "bundled",
        "dacvae_sample_rate": 44100,
        "dacvae_hop_length": 512,
        "dacvae_latent_dim": 128,
        "dacvae_sha256": "c" * 64,
        "generative_objective": "vp_diffusion_v",
        "text_conditioning_mode": "frozen_features",
        "text_num_layers": 0,
        "text_vocab_size": 23,
        "pad_token": 1,
        "conditioning_feature_size": 4,
        "conditioning_feature_dtype": "float16",
        "frozen_text_alignment": "hf_non_special_tokens_v1",
        "frozen_text_cache_version": 1,
        "frozen_text_config_sha256": "a" * 64,
        "frozen_text_encoder_id": "example/encoder",
        "frozen_text_encoder_revision": "a" * 40,
        "frozen_text_frontend": "phonemes",
        "frozen_text_hidden_layer": -1,
        "frozen_text_model_filename": "model.safetensors",
        "frozen_text_model_sha256": "b" * 64,
        "frozen_text_tokenizer_filename": "tokenizer.json",
        "frozen_text_tokenizer_id": "example/tokenizer",
        "frozen_text_tokenizer_revision": "a" * 40,
        "frozen_text_tokenizer_sha256": "c" * 64,
        "target_patch_size": 1,
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


def _tiny_legacy_v3_model(*, architecture_version: int = 3) -> FlowMatchingEchoDiT:
    return FlowMatchingEchoDiT(
        latent_size=2,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=100312,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        timestep_embed_size=8,
        adaln_rank=2,
        use_speaker_conditioning=True,
        architecture_version=architecture_version,
    )


def _schema2_raw(weight_sha256: str) -> dict:
    return {
        "schema_version": ORIGIN_MODEL_MANIFEST_SCHEMA_VERSION,
        "library": "nar-vae",
        "stage": "pretrain",
        "weights": {"pytorch_model.bin": weight_sha256},
        "architecture": {
            "latent_size": 2,
            "model_size": 8,
            "num_layers": 1,
            "num_heads": 2,
            "intermediate_size": 16,
            "text_model_size": 8,
            "text_num_layers": 1,
            "text_num_heads": 2,
            "text_intermediate_size": 16,
            "speaker_model_size": 8,
            "speaker_num_layers": 1,
            "speaker_num_heads": 2,
            "speaker_intermediate_size": 16,
            "timestep_embed_size": 8,
            "adaln_rank": 2,
            "text_vocab_size": 100312,
            "speaker_patch_size": 2,
            "use_speaker_conditioning": True,
            "use_mas_duration": False,
            "norm_eps": 1e-6,
        },
        "capabilities": {
            "speaker_conditioning": True,
            "language_conditioning": False,
            "supported_languages": ["en"],
            "supported_reference_languages": [],
            "supported_language_pairs": [],
            "duration_predictor": False,
            "duration_predictor_hidden_size": 0,
            "duration_predictor_num_layers": 0,
            "duration_predictor_use_speaker": False,
            "monotonic_alignment": False,
            "duration_alignment_hidden_size": 0,
        },
        "representation": {
            "contract_version": 2,
            "text_frontend_name": LEGACY_CL100K_TEXT_FRONTEND_NAME,
            "text_frontend_version": LEGACY_CL100K_TEXT_FRONTEND_VERSION,
            "codec_source": "codec.pth",
            "codec_backend": "bundled",
            "codec_revision": None,
            "codec_filename": None,
            "codec_sha256": "c" * 64,
            "sample_rate": 16,
            "hop_length": 4,
            "latent_width": 2,
        },
        "parent": None,
    }


class _TinyCodec(torch.nn.Module):
    sample_rate = 16
    hop_length = 4

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.quantizer = SimpleNamespace(out_proj=SimpleNamespace(in_channels=2))
        self.nar_vae_codec_identifier = "codec.pth"
        self.nar_vae_backend = "bundled"
        self.nar_vae_codec_revision = None
        self.nar_vae_codec_filename = None
        self.nar_vae_codec_sha256 = "c" * 64

    @staticmethod
    def decode(latents):
        return latents[:, :1]


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
            self.assertEqual(loaded.architecture["speaker_num_summary_tokens"], 0)
            self.assertTrue(loaded.capabilities["monotonic_alignment"])
            self.assertEqual(
                loaded.representation["text_frontend_name"],
                FROZEN_TEXT_REPRESENTATION_NAME,
            )
            self.assertEqual(loaded.representation["codec_source"], "./codec/weights.pth")
            self.assertIsNone(loaded.representation["codec_revision"])
            self.assertEqual(
                loaded.representation["codec_encoding_policy"],
                DACVAE_POSTERIOR_SAMPLING_POLICY,
            )
            validate_manifest_weight(loaded, weights)

            weights.write_bytes(b"different model weights")
            with self.assertRaisesRegex(ModelManifestError, "manifest SHA-256"):
                validate_manifest_weight(loaded, weights)

    def test_pretraining_writer_rejects_scratch_token_exports_defense_in_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "pytorch_model.bin"
            weights.write_bytes(b"scratch weights")
            scratch = model_config("./codec/weights.pth")
            scratch["text_conditioning_mode"] = "scratch_tokens"
            with self.assertRaisesRegex(ModelManifestError, "require frozen_features"):
                write_model_manifest(
                    root,
                    scratch,
                    stage="pretrain",
                    checkpoint_files=(weights.name,),
                )
            self.assertFalse((root / MODEL_MANIFEST_FILENAME).exists())

            for name, objective in (("omitted", None), ("rectified_flow", "rectified_flow")):
                invalid = model_config("./codec/weights.pth")
                if objective is None:
                    invalid.pop("generative_objective")
                else:
                    invalid["generative_objective"] = objective
                with (
                    self.subTest(objective=name),
                    self.assertRaisesRegex(
                        ModelManifestError,
                        "generative_objective: vp_diffusion_v",
                    ),
                ):
                    write_model_manifest(
                        root,
                        invalid,
                        stage="pretrain",
                        checkpoint_files=(weights.name,),
                    )
                self.assertFalse((root / MODEL_MANIFEST_FILENAME).exists())

    def test_frozen_export_binds_provider_axis_and_frontend_for_inference(self):
        config = model_config("./codec/weights.pth")
        config.update(
            text_conditioning_mode="frozen_features",
            text_num_layers=0,
            text_vocab_size=23,
            pad_token=1,
            conditioning_feature_size=4,
            conditioning_feature_dtype="float16",
            frozen_text_alignment="hf_non_special_tokens_v1",
            frozen_text_cache_version=1,
            frozen_text_config_sha256="a" * 64,
            frozen_text_encoder_id="example/encoder",
            frozen_text_encoder_revision="a" * 40,
            frozen_text_frontend="phonemes",
            frozen_text_hidden_layer=-1,
            frozen_text_model_filename="model.safetensors",
            frozen_text_model_sha256="b" * 64,
            frozen_text_tokenizer_filename="tokenizer.json",
            frozen_text_tokenizer_id="example/tokenizer",
            frozen_text_tokenizer_revision="a" * 40,
            frozen_text_tokenizer_sha256="c" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, written = self._write(root, config)
            loaded = load_model_manifest(root / MODEL_MANIFEST_FILENAME)

            self.assertEqual(
                loaded.representation["text_frontend_name"],
                FROZEN_TEXT_REPRESENTATION_NAME,
            )
            self.assertEqual(loaded.text_conditioning["provider_vocab_size"], 23)
            self.assertEqual(loaded.text_conditioning["provider_pad_token"], 1)
            validate_inference_manifest(
                loaded,
                checkpoint_path=weights,
                selected_filename=weights.name,
                architecture=loaded.architecture,
                capabilities=loaded.capabilities,
                generation=loaded.generation,
                text_conditioning=loaded.text_conditioning,
                codec_source="./codec/weights.pth",
                codec_backend="bundled",
            )
            self.assertEqual(written.raw, loaded.raw)

    def test_current_manifest_records_exact_language_pairs_authoritatively(self):
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

            self.assertEqual(
                manifest.raw["schema_version"],
                MODEL_MANIFEST_SCHEMA_VERSION,
            )
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
            speaker_migration["speaker_num_summary_tokens"] = 8
            validate_sft_parent_manifest(weights, speaker_migration)

    def test_manifest_binds_fixed_speaker_summary_topology(self):
        config = model_config("./codec/a.pth")
        config.update(
            use_speaker_conditioning=True,
            speaker_num_summary_tokens=8,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, manifest = self._write(root, config)

            self.assertEqual(manifest.architecture["speaker_num_summary_tokens"], 8)

            changed_topology = dict(config)
            changed_topology["speaker_num_summary_tokens"] = 4
            with self.assertRaisesRegex(ModelManifestError, "architecture"):
                validate_sft_parent_manifest(weights, changed_topology)

    def test_manifest_rejects_invalid_or_unconditioned_summary_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            manifest_path = root / MODEL_MANIFEST_FILENAME

            for invalid_value in (-1, True, 1.5):
                raw = json.loads(json.dumps(manifest.raw))
                raw["architecture"]["speaker_num_summary_tokens"] = invalid_value
                manifest_path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(invalid_value=invalid_value):
                    with self.assertRaisesRegex(ModelManifestError, "non-negative"):
                        load_model_manifest(manifest_path)

            raw = json.loads(json.dumps(manifest.raw))
            raw["architecture"]["speaker_num_summary_tokens"] = 8
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ModelManifestError, "speaker-conditioning"):
                load_model_manifest(manifest_path)

    def test_previous_schema4_manifest_preserves_raw_identity_without_seeded_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            raw = json.loads(json.dumps(manifest.raw))
            raw["schema_version"] = PREVIOUS_MODEL_MANIFEST_SCHEMA_VERSION
            raw["representation"]["contract_version"] = 2
            raw["representation"].pop("codec_encoding_policy")
            expected_sha256 = canonical_sha256(raw)
            manifest_path = root / MODEL_MANIFEST_FILENAME
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = load_model_manifest(manifest_path)

            self.assertEqual(loaded.raw, raw)
            self.assertEqual(loaded.sha256, expected_sha256)
            self.assertNotIn("codec_encoding_policy", loaded.representation)
            self.assertIn("generation", loaded.raw)
            self.assertIn("text_conditioning", loaded.raw)
            with self.assertRaisesRegex(ModelManifestError, "cannot be relabeled"):
                model_export_config_from_manifest(loaded)

    def test_public_sft_writer_cannot_promote_schema3_schema4_or_unbound_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain_dir = root / "pretrain"
            sft_dir = root / "sft"
            pretrain_dir.mkdir()
            sft_dir.mkdir()
            config = model_config("./codec/weights.pth")

            _, current_pretrain = self._write(pretrain_dir, config)
            sft_weights = sft_dir / "pytorch_model.bin"
            sft_weights.write_bytes(b"sft weights")

            for schema_version in (
                LEGACY_MODEL_MANIFEST_SCHEMA_VERSION,
                PREVIOUS_MODEL_MANIFEST_SCHEMA_VERSION,
            ):
                legacy_raw = json.loads(json.dumps(current_pretrain.raw))
                legacy_raw["schema_version"] = schema_version
                legacy_raw["representation"]["contract_version"] = 2
                legacy_raw["representation"].pop("codec_encoding_policy")
                if schema_version == LEGACY_MODEL_MANIFEST_SCHEMA_VERSION:
                    legacy_raw["architecture"].pop("speaker_num_summary_tokens")
                    legacy_raw["architecture"]["text_num_layers"] = 1
                    legacy_raw["architecture"]["text_vocab_size"] = 530
                    legacy_raw["representation"]["text_frontend_name"] = TEXT_FRONTEND_NAME
                    legacy_raw["representation"]["text_frontend_version"] = 2
                    legacy_raw.pop("generation")
                    legacy_raw.pop("text_conditioning")
                (pretrain_dir / MODEL_MANIFEST_FILENAME).write_text(
                    json.dumps(legacy_raw),
                    encoding="utf-8",
                )
                legacy_pretrain = load_model_manifest(pretrain_dir / MODEL_MANIFEST_FILENAME)

                with (
                    self.subTest(schema_version=schema_version),
                    self.assertRaisesRegex(
                        ModelManifestError,
                        "architecture|codec/frontend representation",
                    ),
                ):
                    write_model_manifest(
                        sft_dir,
                        config,
                        stage="sft",
                        checkpoint_files=(sft_weights.name,),
                        parent_manifest=legacy_pretrain,
                    )
                self.assertFalse((sft_dir / MODEL_MANIFEST_FILENAME).exists())

            with self.assertRaisesRegex(ModelManifestError, "fully validated pretraining"):
                write_model_manifest(
                    sft_dir,
                    config,
                    stage="sft",
                    checkpoint_files=(sft_weights.name,),
                    parent_manifest={
                        "manifest_sha256": "a" * 64,
                        "stage": "pretrain",
                        "weights_sha256": "b" * 64,
                        "representation_sha256": "c" * 64,
                    },
                )
            self.assertFalse((sft_dir / MODEL_MANIFEST_FILENAME).exists())

    def test_public_grpo_writer_cannot_relabel_a_schema4_sft_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain_dir = root / "pretrain"
            sft_dir = root / "sft"
            grpo_dir = root / "grpo"
            pretrain_dir.mkdir()
            sft_dir.mkdir()
            grpo_dir.mkdir()
            config = model_config("./codec/weights.pth")

            _, pretrain_manifest = self._write(pretrain_dir, config)
            sft_weights = sft_dir / "pytorch_model.bin"
            sft_weights.write_bytes(b"sft weights")
            current_sft = write_model_manifest(
                sft_dir,
                config,
                stage="sft",
                checkpoint_files=(sft_weights.name,),
                parent_manifest=pretrain_manifest,
            )
            legacy_raw = json.loads(json.dumps(current_sft.raw))
            legacy_raw["schema_version"] = PREVIOUS_MODEL_MANIFEST_SCHEMA_VERSION
            legacy_raw["representation"]["contract_version"] = 2
            legacy_raw["representation"].pop("codec_encoding_policy")
            (sft_dir / MODEL_MANIFEST_FILENAME).write_text(
                json.dumps(legacy_raw),
                encoding="utf-8",
            )
            legacy_sft = load_model_manifest(sft_dir / MODEL_MANIFEST_FILENAME)

            grpo_weights = grpo_dir / "pytorch_model.bin"
            grpo_weights.write_bytes(b"grpo weights")
            with self.assertRaisesRegex(
                ModelManifestError,
                "legacy representations cannot be relabeled",
            ):
                write_model_manifest(
                    grpo_dir,
                    config,
                    stage="grpo",
                    checkpoint_files=(grpo_weights.name,),
                    parent_manifest=legacy_sft,
                    parent_checkpoint_path=sft_weights,
                )

            self.assertFalse((grpo_dir / MODEL_MANIFEST_FILENAME).exists())

    def test_public_grpo_writer_cannot_relabel_a_current_sft_representation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain_dir = root / "pretrain"
            sft_dir = root / "sft"
            grpo_dir = root / "grpo"
            pretrain_dir.mkdir()
            sft_dir.mkdir()
            grpo_dir.mkdir()
            config = model_config("./codec/weights.pth")

            _, pretrain_manifest = self._write(pretrain_dir, config)
            sft_weights = sft_dir / "pytorch_model.bin"
            sft_weights.write_bytes(b"sft weights")
            sft_manifest = write_model_manifest(
                sft_dir,
                config,
                stage="sft",
                checkpoint_files=(sft_weights.name,),
                parent_manifest=pretrain_manifest,
            )
            grpo_weights = grpo_dir / "pytorch_model.bin"
            grpo_weights.write_bytes(b"grpo weights")
            changed_config = dict(config)
            changed_config["dacvae_sha256"] = "d" * 64

            with self.assertRaisesRegex(ModelManifestError, "codec/frontend representation"):
                write_model_manifest(
                    grpo_dir,
                    changed_config,
                    stage="grpo",
                    checkpoint_files=(grpo_weights.name,),
                    parent_manifest=sft_manifest,
                    parent_checkpoint_path=sft_weights,
                )

            self.assertFalse((grpo_dir / MODEL_MANIFEST_FILENAME).exists())

    def test_legacy_schema3_manifest_preserves_raw_identity_and_old_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            raw = json.loads(json.dumps(manifest.raw))
            raw["schema_version"] = LEGACY_MODEL_MANIFEST_SCHEMA_VERSION
            raw["representation"]["contract_version"] = 2
            raw["representation"].pop("codec_encoding_policy")
            raw["representation"]["text_frontend_name"] = TEXT_FRONTEND_NAME
            raw["representation"]["text_frontend_version"] = 2
            raw["architecture"].pop("speaker_num_summary_tokens")
            raw["architecture"]["text_num_layers"] = 1
            raw["architecture"]["text_vocab_size"] = 530
            raw.pop("generation")
            raw.pop("text_conditioning")
            expected_sha256 = canonical_sha256(raw)
            manifest_path = root / MODEL_MANIFEST_FILENAME
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = load_model_manifest(manifest_path)

            self.assertEqual(loaded.architecture["speaker_num_summary_tokens"], 0)
            self.assertEqual(loaded.raw, raw)
            self.assertEqual(loaded.sha256, expected_sha256)
            self.assertNotIn("speaker_num_summary_tokens", loaded.raw["architecture"])
            self.assertNotIn("codec_encoding_policy", loaded.representation)

    def test_origin_schema2_manifest_preserves_raw_hash_and_authenticates_v3_cl100k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            raw = json.loads(json.dumps(manifest.raw))
            raw["schema_version"] = ORIGIN_MODEL_MANIFEST_SCHEMA_VERSION
            raw["representation"]["contract_version"] = 2
            raw["representation"].pop("codec_encoding_policy")
            raw["representation"]["text_frontend_name"] = LEGACY_CL100K_TEXT_FRONTEND_NAME
            raw["representation"]["text_frontend_version"] = LEGACY_CL100K_TEXT_FRONTEND_VERSION
            raw["architecture"].pop("architecture_version")
            raw["architecture"].pop("speaker_num_summary_tokens")
            raw["architecture"].pop("target_patch_size")
            raw["architecture"]["text_num_layers"] = 1
            raw["architecture"]["text_vocab_size"] = 100312
            raw.pop("generation")
            raw.pop("text_conditioning")
            expected_sha256 = canonical_sha256(raw)
            manifest_path = root / MODEL_MANIFEST_FILENAME
            manifest_path.write_text(json.dumps(raw), encoding="utf-8")

            loaded = load_model_manifest(manifest_path)

            self.assertEqual(loaded.raw, raw)
            self.assertEqual(loaded.sha256, expected_sha256)
            self.assertEqual(loaded.architecture["architecture_version"], 3)
            self.assertEqual(loaded.architecture["target_patch_size"], 1)
            self.assertEqual(loaded.architecture["speaker_num_summary_tokens"], 0)
            self.assertEqual(loaded.generation["objective"], "rectified_flow")
            self.assertEqual(loaded.text_conditioning["mode"], "scratch_tokens")
            self.assertNotIn("architecture_version", loaded.raw["architecture"])

            tampered = json.loads(json.dumps(raw))
            tampered["representation"]["text_frontend_name"] = TEXT_FRONTEND_NAME
            manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(
                ModelManifestError,
                "representation frontend contradicts",
            ):
                load_model_manifest(manifest_path)

            invalid = json.loads(json.dumps(raw))
            invalid["architecture"]["text_num_layers"] = 0
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ModelManifestError, "trained text encoder"):
                load_model_manifest(manifest_path)

            invalid = json.loads(json.dumps(raw))
            invalid["architecture"]["text_vocab_size"] = 100286
            manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ModelManifestError, "include the legacy pad token"):
                load_model_manifest(manifest_path)

    def test_schema2_real_v3_constructor_frontend_and_synthesis_are_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "pytorch_model.bin"
            torch.save(_tiny_legacy_v3_model().state_dict(), weights)
            raw = _schema2_raw(hashlib.sha256(weights.read_bytes()).hexdigest())
            (root / MODEL_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
            codec = _TinyCodec()

            with patch("nar_vae.inference.load_dacvae", return_value=codec):
                runtime = FlowMatchingTTSInference(
                    weights,
                    dacvae_model="codec.pth",
                    device="cpu",
                    latent_size=2,
                    model_size=8,
                    num_layers=1,
                    num_heads=2,
                    intermediate_size=16,
                    text_num_layers=1,
                    text_model_size=8,
                    text_num_heads=2,
                    text_intermediate_size=16,
                    speaker_model_size=8,
                    speaker_num_layers=1,
                    speaker_num_heads=2,
                    speaker_intermediate_size=16,
                    timestep_embed_size=8,
                    adaln_rank=2,
                )

            prepared = runtime._prepare_conditioning("Hello, world!", "en")
            self.assertEqual(
                prepared.conditioning_ids.tolist(),
                [[100282, 9906, 11, 1917, 0, 100283, 100284, 100280]],
            )
            with (
                patch(
                    "nar_vae.inference.ODESolver.sample",
                    side_effect=lambda **kwargs: torch.zeros(kwargs["latent_shape"]),
                ) as sample,
                patch(
                    "nar_vae.inference.encode_dacvae_posterior_legacy_global_rng",
                    return_value=torch.ones(1, 2, 2),
                ) as legacy_encode,
            ):
                audio = runtime.synthesize(
                    "Hello",
                    duration=1.5,
                    num_steps=1,
                    show_progress=False,
                    reference_audio=torch.zeros(8),
                    reference_sample_rate=16,
                )
            self.assertEqual(tuple(audio.shape), (6,))
            self.assertEqual(sample.call_args.kwargs["conditioning_ids"][0, 0].item(), 100282)
            self.assertIsNotNone(sample.call_args.kwargs["speaker_latent"])
            self.assertEqual(tuple(sample.call_args.kwargs["speaker_latent"].shape), (1, 2, 2))
            legacy_encode.assert_called_once()
            self.assertEqual(runtime.flow_model.architecture_version, 3)
            self.assertEqual(
                {
                    name: (
                        runtime.generation_profile(name).num_steps,
                        runtime.generation_profile(name).solver,
                        runtime.generation_profile(name).cache_mode,
                    )
                    for name in ("quality", "balanced", "fast", "turbo")
                },
                {
                    "quality": (50, "heun", "none"),
                    "balanced": (32, "euler", "none"),
                    "fast": (16, "euler", "none"),
                    "turbo": (16, "euler", "cache_dit"),
                },
            )

    def test_manifest_and_state_architecture_versions_cross_reject_before_codec_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "pytorch_model.bin"
            torch.save(_tiny_legacy_v3_model(architecture_version=4).state_dict(), weights)
            raw = _schema2_raw(hashlib.sha256(weights.read_bytes()).hexdigest())
            (root / MODEL_MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
            constructor = dict(
                flow_model_path=weights,
                dacvae_model="codec.pth",
                device="cpu",
                latent_size=2,
                model_size=8,
                num_layers=1,
                num_heads=2,
                intermediate_size=16,
                text_num_layers=1,
                text_model_size=8,
                text_num_heads=2,
                text_intermediate_size=16,
                speaker_model_size=8,
                speaker_num_layers=1,
                speaker_num_heads=2,
                speaker_intermediate_size=16,
                timestep_embed_size=8,
                adaln_rank=2,
            )
            with (
                patch("nar_vae.inference.load_dacvae") as load_codec,
                self.assertRaisesRegex(LegacyArchitectureCheckpointError, "does not match"),
            ):
                FlowMatchingTTSInference(**constructor)
            load_codec.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "pytorch_model.bin"
            # Use a valid current frozen manifest shell while binding the v3 bytes.
            _, current = self._write(root, model_config("codec.pth"))
            torch.save(_tiny_legacy_v3_model().state_dict(), weights)
            current_raw = json.loads(json.dumps(current.raw))
            current_raw["weights"] = {
                weights.name: hashlib.sha256(weights.read_bytes()).hexdigest()
            }
            (root / MODEL_MANIFEST_FILENAME).write_text(
                json.dumps(current_raw),
                encoding="utf-8",
            )
            with (
                patch("nar_vae.inference.load_dacvae") as load_codec,
                self.assertRaisesRegex(
                    LegacyArchitectureCheckpointError,
                    "do not contain versioned",
                ),
            ):
                FlowMatchingTTSInference(weights, dacvae_model="codec.pth", device="cpu")
            load_codec.assert_not_called()

    def test_current_manifest_requires_the_exact_seeded_codec_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            manifest_path = root / MODEL_MANIFEST_FILENAME

            mutations = {
                "missing": lambda representation: representation.pop("codec_encoding_policy"),
                "wrong": lambda representation: representation.__setitem__(
                    "codec_encoding_policy",
                    "posterior_mean_v1",
                ),
                "unknown": lambda representation: representation.__setitem__(
                    "codec_seed",
                    1,
                ),
            }
            for name, mutate in mutations.items():
                raw = json.loads(json.dumps(manifest.raw))
                mutate(raw["representation"])
                manifest_path.write_text(json.dumps(raw), encoding="utf-8")
                with self.subTest(name=name), self.assertRaises(ModelManifestError):
                    load_model_manifest(manifest_path)

    def test_manifest_version_axes_reject_float_and_boolean_json_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = self._write(root)
            manifest_path = root / MODEL_MANIFEST_FILENAME
            mutations = {
                "schema-float": ("schema_version", 5.0, "schema_version must be an integer"),
                "schema-bool": ("schema_version", True, "schema_version must be an integer"),
                "contract-float": (
                    "representation.contract_version",
                    3.0,
                    "representation contract version",
                ),
                "contract-bool": (
                    "representation.contract_version",
                    True,
                    "representation contract version",
                ),
                "frontend-float": (
                    "representation.text_frontend_version",
                    1.0,
                    "text_frontend_version must be a positive integer",
                ),
                "frontend-bool": (
                    "representation.text_frontend_version",
                    True,
                    "text_frontend_version must be a positive integer",
                ),
            }
            for name, (path, value, message) in mutations.items():
                raw = json.loads(json.dumps(manifest.raw))
                if path == "schema_version":
                    raw[path] = value
                else:
                    _, field = path.split(".")
                    raw["representation"][field] = value
                manifest_path.write_text(json.dumps(raw), encoding="utf-8")
                with (
                    self.subTest(name=name),
                    self.assertRaisesRegex(
                        ModelManifestError,
                        message,
                    ),
                ):
                    load_model_manifest(manifest_path)

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
            with self.assertRaisesRegex(
                ModelManifestError,
                "representation frontend contradicts",
            ):
                load_model_manifest(root / MODEL_MANIFEST_FILENAME)

    def test_partial_ema_inference_also_hashes_the_loaded_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain = root / "pretrain"
            sft = root / "sft"
            pretrain.mkdir()
            sft.mkdir()
            config = model_config("./codec/a.pth")
            _, parent_manifest = self._write(pretrain, config)
            base = sft / "pytorch_model.bin"
            ema = sft / "pytorch_model_ema.bin"
            base.write_bytes(b"full base")
            ema.write_bytes(b"sparse ema")
            manifest = write_model_manifest(
                sft,
                config,
                stage="sft",
                checkpoint_files=(base.name, ema.name),
                parent_manifest=parent_manifest,
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
        checkpoint.infer_text_vocab_size.return_value = 530
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
