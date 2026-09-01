"""Tests for EchoDiT checkpoint compatibility inference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nar_vae.checkpoint import (
    FlowCheckpoint,
    LegacySpeakerCheckpointError,
    load_pretrained_checkpoint,
    resolve_flow_checkpoint,
)
from nar_vae.voice import SPEAKER_CONDITIONING_VERSION, SPEAKER_PATCH_LAYOUT_VERSION


class FlowCheckpointTest(unittest.TestCase):
    def test_training_preload_validator_runs_immediately_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pytorch_model.bin"
            torch.save({"weight": torch.ones(1, 1)}, checkpoint_path)
            model = torch.nn.Linear(1, 1, bias=False)
            original_load = torch.load
            events = []

            def validate(path):
                events.append(("validate", path))

            def deserialize(*args, **kwargs):
                events.append(("deserialize", Path(args[0]).resolve()))
                return original_load(*args, **kwargs)

            with patch("nar_vae.checkpoint.torch.load", side_effect=deserialize):
                load_pretrained_checkpoint(
                    model,
                    checkpoint_path,
                    preload_validator=validate,
                )

            self.assertEqual(
                events,
                [
                    ("validate", checkpoint_path.resolve()),
                    ("deserialize", checkpoint_path.resolve()),
                ],
            )

    def test_infers_architecture_facts_from_local_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pytorch_model.bin"
            torch.save(
                {
                    "dit.text_encoder.text_embedding.weight": torch.empty(100287, 2),
                    "dit.speaker_encoder.in_proj.weight": torch.empty(2, 2),
                },
                checkpoint_path,
            )

            checkpoint = FlowCheckpoint.load(checkpoint_path)

            self.assertEqual(checkpoint.infer_text_vocab_size(1), 100287)
            self.assertFalse(checkpoint.infer_speaker_conditioning(True))

    def test_snapshot_directory_resolves_main_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            checkpoint_path = snapshot / "pytorch_model.bin"
            checkpoint_path.touch()

            self.assertEqual(resolve_flow_checkpoint(snapshot), checkpoint_path)

    def test_snapshot_directory_prefers_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            checkpoint_dir = snapshot
            base_path = checkpoint_dir / "pytorch_model.bin"
            ema_path = checkpoint_dir / "pytorch_model_ema.bin"
            base_path.touch()
            ema_path.touch()

            self.assertEqual(resolve_flow_checkpoint(snapshot), ema_path)
            self.assertEqual(resolve_flow_checkpoint(snapshot, prefer_ema=False), base_path)

    def test_partial_ema_requires_base_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            ema_path = Path(directory) / "pytorch_model_ema.bin"
            torch.save({"weight": torch.ones(1)}, ema_path)

            with self.assertRaisesRegex(FileNotFoundError, "requires its full base"):
                FlowCheckpoint.load(ema_path)

    def test_speaker_capability_requires_versioned_patch_layout(self):
        valid = FlowCheckpoint(
            path=Path("valid.bin"),
            state_dict={
                "null_speaker_embed": torch.zeros(1, 2, 4),
                "speaker_conditioning_version": torch.tensor(SPEAKER_CONDITIONING_VERSION),
                "speaker_patch_layout_version": torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                "speaker_patch_size_metadata": torch.tensor(4),
                "dit.speaker_encoder.in_proj.weight": torch.empty(2, 8),
            },
        )
        legacy = FlowCheckpoint(
            path=Path("legacy.bin"),
            state_dict={"null_speaker_embed": torch.zeros(1, 2, 4)},
        )

        self.assertTrue(valid.infer_speaker_conditioning(False))
        self.assertEqual(valid.infer_speaker_patch_size(1), 4)
        with self.assertRaisesRegex(LegacySpeakerCheckpointError, "ambiguous"):
            legacy.infer_speaker_conditioning(False)

    def test_speaker_metadata_can_be_ignored_by_a_disabled_model(self):
        model = torch.nn.Linear(1, 1, bias=False)
        checkpoint = FlowCheckpoint(
            path=Path("speaker.bin"),
            state_dict={
                "weight": torch.ones(1, 1),
                "null_speaker_embed": torch.zeros(1, 1, 1),
                "speaker_conditioning_version": torch.tensor(SPEAKER_CONDITIONING_VERSION),
                "speaker_patch_layout_version": torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                "speaker_patch_size_metadata": torch.tensor(1),
            },
        )

        checkpoint.load_into(model)

        self.assertEqual(model.weight.item(), 1.0)

    def test_wrapped_ema_overlays_base(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            base_path = checkpoint_dir / "pytorch_model.bin"
            ema_path = checkpoint_dir / "ema_model.bin"
            torch.save({"weight": torch.zeros(1, 1)}, base_path)
            torch.save({"shadow": {"weight": torch.ones(1, 1)}, "decay": 0.9}, ema_path)

            checkpoint = FlowCheckpoint.load(ema_path)
            model = torch.nn.Linear(1, 1, bias=False)
            checkpoint.load_into(model)

            self.assertEqual(model.weight.item(), 1.0)

    def test_training_loader_requires_explicit_speaker_initialization(self):
        class TinySpeakerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.dit = torch.nn.Module()
                self.dit.speaker_encoder = torch.nn.Module()
                self.dit.speaker_encoder.in_proj = torch.nn.Linear(4, 1, bias=False)
                self.register_buffer("null_speaker_embed", torch.zeros(1, 1, 4))
                self.register_buffer(
                    "speaker_conditioning_version",
                    torch.tensor(SPEAKER_CONDITIONING_VERSION),
                )
                self.register_buffer(
                    "speaker_patch_layout_version",
                    torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                )
                self.register_buffer("speaker_patch_size_metadata", torch.tensor(4))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "text-only.bin"
            torch.save(
                {
                    "module.weight": torch.ones(1),
                    "module.dit.speaker_encoder.in_proj.weight": torch.zeros(1, 4),
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(RuntimeError, "initialize_speaker_conditioning"):
                load_pretrained_checkpoint(TinySpeakerModel(), checkpoint_path)

            model = TinySpeakerModel()
            result = load_pretrained_checkpoint(
                model,
                checkpoint_path,
                initialize_speaker_conditioning=True,
            )

            self.assertEqual(
                set(result.missing_keys),
                {
                    "null_speaker_embed",
                    "speaker_conditioning_version",
                    "speaker_patch_layout_version",
                    "speaker_patch_size_metadata",
                },
            )
            self.assertEqual(model.weight.item(), 1.0)

    def test_training_loader_rejects_unversioned_speaker_checkpoint(self):
        class TinySpeakerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.dit = torch.nn.Module()
                self.dit.speaker_encoder = torch.nn.Module()
                self.dit.speaker_encoder.in_proj = torch.nn.Linear(4, 1, bias=False)
                self.register_buffer("null_speaker_embed", torch.zeros(1, 1, 4))
                self.register_buffer(
                    "speaker_conditioning_version",
                    torch.tensor(SPEAKER_CONDITIONING_VERSION),
                )
                self.register_buffer(
                    "speaker_patch_layout_version",
                    torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION),
                )
                self.register_buffer("speaker_patch_size_metadata", torch.tensor(4))

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "legacy.bin"
            torch.save(
                {
                    "weight": torch.ones(1),
                    "null_speaker_embed": torch.zeros(1, 1, 4),
                },
                checkpoint_path,
            )

            with self.assertRaises(LegacySpeakerCheckpointError):
                load_pretrained_checkpoint(TinySpeakerModel(), checkpoint_path)


if __name__ == "__main__":
    unittest.main()
