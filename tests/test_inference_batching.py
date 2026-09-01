"""CPU-only contracts for compatible-shape tensor batching."""

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nar_vae.configuration import load_inference_settings
from nar_vae.inference import FlowMatchingTTSInference


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def make_runtime() -> FlowMatchingTTSInference:
    runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
    runtime.device = torch.device("cpu")
    runtime.latent_size = 2
    runtime.frame_rate = 4
    runtime.sample_rate = 16
    runtime.tokenizer = FakeTokenizer()
    runtime.settings = load_inference_settings()
    runtime.supports_voice_cloning = False
    runtime.uses_language_conditioning = False
    runtime.uses_learned_duration = False
    runtime.uses_mas_duration = False
    runtime.supported_languages = ("en",)
    runtime.supported_reference_languages = ()
    runtime.supports_cross_lingual = False
    runtime.checkpoint_path = Path("checkpoint.bin")
    runtime.flow_model = object()
    runtime._decode = lambda latents: latents[:, :1, :]
    return runtime


class FakeMASFlowModel:
    def __init__(self):
        self.calls = []

    def predict_token_duration_frames(
        self,
        conditioning_ids,
        attention_mask=None,
        speaker_latent=None,
        language_ids=None,
        *,
        total_frames,
    ):
        self.calls.append(
            {
                "conditioning_ids": conditioning_ids,
                "attention_mask": attention_mask,
                "speaker_latent": speaker_latent,
                "language_ids": language_ids,
                "total_frames": total_frames,
            }
        )
        valid = (
            attention_mask.to(dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(conditioning_ids, dtype=torch.bool)
        )
        durations = valid.to(dtype=torch.long)
        durations[:, 0] += total_frames - valid.sum(dim=1)
        return durations


class InferenceBatchingTest(unittest.TestCase):
    def test_public_inference_defaults_do_not_enable_uncalibrated_cfg(self):
        synthesize = inspect.signature(FlowMatchingTTSInference.synthesize).parameters
        to_file = inspect.signature(FlowMatchingTTSInference.synthesize_to_file).parameters
        batch = inspect.signature(FlowMatchingTTSInference.synthesize_batch).parameters

        for parameters in (synthesize, to_file):
            self.assertEqual(parameters["cfg_scale"].default, 1.0)
            self.assertEqual(parameters["cfg_mode"].default, "joint")
            self.assertIsNone(parameters["cfg_scale_text"].default)
            self.assertIsNone(parameters["cfg_scale_speaker"].default)
        self.assertEqual(batch["cfg_scale"].default, 1.0)
        self.assertIsNone(batch["max_duration"].default)

    def test_compatible_requests_share_one_ode_and_decode_batch(self):
        runtime = make_runtime()
        calls = []

        def fake_sample(**kwargs):
            calls.append(kwargs)
            batch, channels, frames = kwargs["latent_shape"]
            values = torch.arange(batch, dtype=torch.float32)[:, None, None]
            return values.expand(batch, channels, frames).clone()

        with patch("nar_vae.inference.ODESolver.sample", side_effect=fake_sample):
            audios = runtime.synthesize_batch(["a", "longer"], num_steps=4)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["latent_shape"], (2, 2, 6))
        self.assertEqual(tuple(calls[0]["conditioning_ids"].shape), (2, 10))
        self.assertEqual(calls[0]["conditioning_mask"].sum(dim=1).tolist(), [5, 10])
        self.assertEqual(calls[0]["cfg_scale"], 1.0)
        self.assertEqual(calls[0]["cfg_mode"], "joint")
        self.assertIsNone(calls[0]["cfg_scale_text"])
        self.assertIsNone(calls[0]["cfg_scale_speaker"])
        self.assertFalse(calls[0]["fuse_cfg_branches"])
        self.assertEqual(tuple(audios[0].shape), (6,))
        self.assertEqual(tuple(audios[1].shape), (6,))
        torch.testing.assert_close(audios[0], torch.zeros(6))
        torch.testing.assert_close(audios[1], torch.ones(6))

    def test_mas_batch_predicts_and_forwards_exact_padded_token_durations(self):
        runtime = make_runtime()
        runtime.uses_mas_duration = True
        runtime.flow_model = FakeMASFlowModel()
        calls = []

        def fake_sample(**kwargs):
            calls.append(kwargs)
            return torch.zeros(kwargs["latent_shape"])

        with patch("nar_vae.inference.ODESolver.sample", side_effect=fake_sample):
            runtime.synthesize_batch(["a", "longer"], num_steps=1)

        self.assertEqual(len(runtime.flow_model.calls), 1)
        prediction_call = runtime.flow_model.calls[0]
        self.assertEqual(prediction_call["total_frames"], 6)
        self.assertEqual(
            prediction_call["attention_mask"].sum(dim=1).tolist(),
            [5, 10],
        )
        token_durations = calls[0]["token_durations"]
        self.assertEqual(token_durations.sum(dim=1).tolist(), [6, 6])
        self.assertEqual(
            token_durations.masked_select(~calls[0]["conditioning_mask"]).count_nonzero().item(),
            0,
        )

    def test_mas_single_request_forwards_exact_token_durations(self):
        runtime = make_runtime()
        runtime.uses_mas_duration = True
        runtime.flow_model = FakeMASFlowModel()
        calls = []

        def fake_sample(**kwargs):
            calls.append(kwargs)
            return torch.zeros(kwargs["latent_shape"])

        with patch("nar_vae.inference.ODESolver.sample", side_effect=fake_sample):
            runtime.synthesize("a", duration=1.5, num_steps=1, show_progress=False)

        self.assertEqual(runtime.flow_model.calls[0]["total_frames"], 6)
        self.assertEqual(calls[0]["token_durations"].sum().item(), 6)

    def test_only_exact_latent_lengths_are_grouped_and_order_is_preserved(self):
        runtime = make_runtime()
        calls = []

        def fake_sample(**kwargs):
            calls.append(kwargs["latent_shape"])
            batch, channels, frames = kwargs["latent_shape"]
            return torch.full((batch, channels, frames), float(frames))

        with patch("nar_vae.inference.ODESolver.sample", side_effect=fake_sample):
            audios = runtime.synthesize_batch(
                ["a", "x" * 30, "b"],
                num_steps=2,
            )

        self.assertEqual(calls, [(2, 2, 6), (1, 2, 9)])
        self.assertEqual([audio.shape[0] for audio in audios], [6, 9, 6])
        self.assertEqual([audio[0].item() for audio in audios], [6.0, 9.0, 6.0])

    def test_decoder_must_preserve_the_batch_dimension(self):
        runtime = make_runtime()
        runtime._decode = lambda latents: torch.zeros(latents.shape[-1])

        with (
            patch(
                "nar_vae.inference.ODESolver.sample",
                return_value=torch.zeros(2, 2, 6),
            ),
            self.assertRaisesRegex(RuntimeError, "preserve the generated batch dimension"),
        ):
            runtime.synthesize_batch(["a", "b"])

    def test_batch_duration_limit_rejects_instead_of_truncating(self):
        runtime = make_runtime()

        with (
            patch("nar_vae.inference.ODESolver.sample") as sample,
            self.assertRaisesRegex(ValueError, "batch maximum"),
        ):
            runtime.synthesize_batch(["x" * 30], max_duration=2.0)

        sample.assert_not_called()

    def test_batch_duration_limit_defaults_to_the_configured_ceiling(self):
        runtime = make_runtime()

        def fake_sample(**kwargs):
            return torch.zeros(kwargs["latent_shape"])

        with patch("nar_vae.inference.ODESolver.sample", side_effect=fake_sample):
            audio = runtime.synthesize_batch(["x" * 30])

        self.assertEqual(audio[0].shape[0], 9)


if __name__ == "__main__":
    unittest.main()
