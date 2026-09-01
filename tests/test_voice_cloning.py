"""Voice-cloning API tests that do not require a large checkpoint."""

import unittest
from inspect import signature
from pathlib import Path
from unittest.mock import patch

import torch

from nar_vae.inference import FlowMatchingTTSInference, VoiceCloningUnsupportedError
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.solvers.ode_solver import ODESolver
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS


class FakeCodec:
    def __init__(self):
        self.last_waveform = None

    def encode(self, waveform):
        self.last_waveform = waveform
        return torch.ones(1, 2, 5, device=waveform.device)


def make_runtime(*, supported: bool) -> FlowMatchingTTSInference:
    runtime = FlowMatchingTTSInference.__new__(FlowMatchingTTSInference)
    runtime.supports_voice_cloning = supported
    runtime.checkpoint_path = Path("checkpoint.bin")
    runtime.device = torch.device("cpu")
    runtime.sample_rate = 48000
    runtime.hop_length = 1920
    runtime.latent_size = 2
    runtime.speaker_patch_size = 4
    runtime.max_reference_seconds = 2.0
    runtime.dacvae = FakeCodec()
    return runtime


def make_tiny_cfg_model() -> FlowMatchingEchoDiT:
    model = FlowMatchingEchoDiT(
        latent_size=2,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=32,
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
        adaln_rank=4,
        use_speaker_conditioning=True,
        use_language_conditioning=False,
        supported_languages=("en",),
    ).eval()
    torch.nn.init.normal_(model.dit.out_proj.weight, std=0.1)
    return model


class VoiceCloningTest(unittest.TestCase):
    def test_inference_uses_the_shared_reference_duration_default(self):
        parameter = signature(FlowMatchingTTSInference).parameters["max_reference_seconds"]

        self.assertEqual(parameter.default, DEFAULT_MAX_REFERENCE_SECONDS)

    def test_public_text_only_checkpoint_rejects_reference_early(self):
        runtime = make_runtime(supported=False)

        with self.assertRaisesRegex(VoiceCloningUnsupportedError, "unavailable"):
            runtime.encode_reference_audio(torch.ones(24000), sample_rate=24000)

    def test_reference_is_mono_resampled_encoded_and_patch_aligned(self):
        runtime = make_runtime(supported=True)
        stereo = torch.stack((torch.ones(24000), torch.zeros(24000)))

        latent = runtime.encode_reference_audio(stereo, sample_rate=24000)

        self.assertEqual(tuple(runtime.dacvae.last_waveform.shape), (1, 1, 48000))
        self.assertEqual(tuple(latent.shape), (1, 2, 8))
        torch.testing.assert_close(latent[..., :5], torch.ones(1, 2, 5))
        torch.testing.assert_close(latent[..., 5:], torch.zeros(1, 2, 3))

    def test_reference_is_cropped_to_the_shared_duration_default(self):
        runtime = make_runtime(supported=True)
        runtime.max_reference_seconds = DEFAULT_MAX_REFERENCE_SECONDS
        overlong_samples = int((DEFAULT_MAX_REFERENCE_SECONDS + 1.0) * runtime.sample_rate)

        runtime.encode_reference_audio(
            torch.ones(overlong_samples),
            sample_rate=runtime.sample_rate,
        )

        self.assertEqual(
            runtime.dacvae.last_waveform.shape[-1],
            int(DEFAULT_MAX_REFERENCE_SECONDS * runtime.sample_rate),
        )

    def test_multiple_references_are_concatenated_under_one_budget(self):
        runtime = make_runtime(supported=True)

        runtime.encode_reference_audio(
            [torch.zeros(12000), torch.ones(24000)],
            sample_rate=[48000, 48000],
            max_seconds=0.5,
        )

        waveform = runtime.dacvae.last_waveform
        self.assertEqual(tuple(waveform.shape), (1, 1, 24000))
        torch.testing.assert_close(waveform[..., :12000], torch.zeros(1, 1, 12000))
        torch.testing.assert_close(waveform[..., 12000:], torch.ones(1, 1, 12000))

        with self.assertRaisesRegex(ValueError, "sample rates must align"):
            runtime.encode_reference_audio(
                [torch.zeros(100), torch.ones(100)],
                sample_rate=[48000],
            )

    def test_reference_and_preencoded_latent_are_mutually_exclusive(self):
        runtime = make_runtime(supported=True)

        with self.assertRaisesRegex(ValueError, "either reference_audio or speaker_latent"):
            runtime._resolve_speaker_latent(
                reference_audio=torch.ones(48000),
                reference_sample_rate=48000,
                speaker_latent=torch.ones(1, 2, 4),
            )

    def test_empty_latent_and_invalid_reference_limits_are_rejected(self):
        runtime = make_runtime(supported=True)

        with self.assertRaisesRegex(ValueError, "at least one frame"):
            runtime._align_speaker_latent(torch.empty(1, 2, 0))
        with self.assertRaisesRegex(ValueError, "sample_rate must be positive"):
            runtime.encode_reference_audio(torch.ones(100), sample_rate=0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            runtime.encode_reference_audio(
                torch.ones(48000),
                sample_rate=48000,
                max_seconds=0,
            )
        with self.assertRaisesRegex(ValueError, "too short to encode"):
            runtime.encode_reference_audio(
                torch.ones(48000),
                sample_rate=48000,
                max_seconds=0.001,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            runtime.encode_reference_audio(
                torch.ones(48000),
                sample_rate=48000,
                max_seconds=float("nan"),
            )

    def test_cfg_null_reference_matches_reference_length(self):
        model = make_tiny_cfg_model()
        reference = torch.ones(1, 2, 8)
        encoded = model.encode_inference_conditioning(
            torch.ones(1, 2, dtype=torch.long),
            torch.ones(1, 2, dtype=torch.bool),
            reference,
            cfg_mode="joint",
            speaker_mask=torch.ones(1, 4, dtype=torch.bool),
        )

        conditional_mask, null_mask = encoded.variants[0].speaker_mask.chunk(2)
        torch.testing.assert_close(conditional_mask, torch.ones(1, 4, dtype=torch.bool))
        torch.testing.assert_close(
            null_mask,
            torch.tensor([[True, False, False, False]]),
        )

    def test_independent_cfg_is_fused_without_changing_the_formula(self):
        torch.manual_seed(13)
        model = make_tiny_cfg_model()
        common = dict(
            latents=torch.randn(1, 2, 4),
            conditioning_ids=torch.tensor([[1, 2]]),
            timesteps=torch.tensor([0.5]),
            cfg_scale=2.0,
            cfg_mode="independent",
            cfg_scale_text=2.5,
            cfg_scale_speaker=2.0,
            speaker_latent=torch.ones(1, 2, 4),
        )
        fused = model.forward_with_cfg(**common, fuse_cfg_branches=True)
        sequential = model.forward_with_cfg(**common, fuse_cfg_branches=False)
        torch.testing.assert_close(fused, sequential)

    def test_alternating_cfg_matches_prepared_and_fallback_formulas(self):
        torch.manual_seed(17)
        model = make_tiny_cfg_model()
        common = {
            "latents": torch.randn(1, 2, 4),
            "conditioning_ids": torch.tensor([[1, 2]]),
            "timesteps": torch.tensor([0.5]),
            "cfg_scale": 2.0,
            "cfg_mode": "alternating",
            "cfg_scale_text": 2.5,
            "cfg_scale_speaker": 2.0,
            "speaker_latent": torch.ones(1, 2, 4),
        }
        prepared = model.prepare_fused_cfg_conditioning(
            common["conditioning_ids"],
            speaker_latent=common["speaker_latent"],
            cfg_mode="alternating",
        )

        for step_idx in (0, 1):
            with self.subTest(step_idx=step_idx):
                direct = model.forward_with_cfg(**common, step_idx=step_idx)
                cached = model.forward_with_prepared_cfg(
                    common["latents"],
                    common["timesteps"],
                    prepared,
                    cfg_scale=common["cfg_scale"],
                    cfg_scale_text=common["cfg_scale_text"],
                    cfg_scale_speaker=common["cfg_scale_speaker"],
                    step_idx=step_idx,
                )
                torch.testing.assert_close(direct, cached)

    def test_prepared_cfg_reuses_encoder_caches_and_preserves_the_formula(self):
        class FakeDiT(torch.nn.Module):
            speaker_patch_size = 4

            def __init__(self):
                super().__init__()
                self.text_cache_calls = 0
                self.speaker_cache_calls = 0
                self.forward_calls = 0

            def encode_text(self, conditioning_ids, text_mask, language_ids=None):
                del text_mask
                del language_ids
                return conditioning_ids.float().sum(dim=1, keepdim=True).unsqueeze(-1)

            def project_text_kv_cache(self, text_state):
                self.text_cache_calls += 1
                values = text_state[:, 0]
                return [(values, values)]

            def get_kv_cache_speaker(self, speaker_latent, speaker_mask=None):
                del speaker_mask
                self.speaker_cache_calls += 1
                values = speaker_latent.float().mean(dim=(1, 2), keepdim=True)
                return [(values, values)]

            def forward(self, **kwargs):
                self.forward_calls += 1
                text = kwargs["kv_cache_text"][0][0].reshape(-1, 1, 1)
                speaker = kwargs["kv_cache_speaker"][0][0].reshape(-1, 1, 1)
                return (text + speaker).expand_as(kwargs["x"])

        model = FlowMatchingEchoDiT.__new__(FlowMatchingEchoDiT)
        torch.nn.Module.__init__(model)
        model.latent_size = 2
        model.speaker_patch_size = 4
        model.use_speaker_conditioning = True
        model.register_buffer("null_speaker_embed", torch.zeros(1, 2, 4))
        model.dit = FakeDiT()
        prepared = model.prepare_fused_cfg_conditioning(
            torch.ones(2, 2, dtype=torch.long),
            speaker_latent=torch.ones(2, 2, 4),
            cfg_mode="independent",
        )

        output = model.forward_with_prepared_cfg(
            torch.zeros(2, 2, 2),
            torch.zeros(2),
            prepared,
            cfg_scale=2.0,
            cfg_scale_text=2.5,
            cfg_scale_speaker=2.0,
            step_idx=0,
        )
        second_output = model.forward_with_prepared_cfg(
            torch.zeros(2, 2, 2),
            torch.ones(2),
            prepared,
            cfg_scale=2.0,
            cfg_scale_text=2.5,
            cfg_scale_speaker=2.0,
            step_idx=1,
        )

        torch.testing.assert_close(output, torch.full_like(output, 10.0))
        torch.testing.assert_close(second_output, output)
        self.assertEqual(model.dit.text_cache_calls, 1)
        self.assertEqual(model.dit.speaker_cache_calls, 1)
        self.assertEqual(model.dit.forward_calls, 2)

        with patch.object(torch, "randn", return_value=torch.zeros(2, 2, 2)):
            sampled = ODESolver.sample(
                model=model,
                conditioning_ids=torch.ones(2, 2, dtype=torch.long),
                num_steps=2,
                latent_shape=(2, 2, 2),
                solver="euler",
                cfg_scale=2.0,
                cfg_mode="independent",
                cfg_scale_text=2.5,
                cfg_scale_speaker=2.0,
                speaker_latent=torch.ones(2, 2, 4),
                fuse_cfg_branches=True,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(sampled, torch.full_like(sampled, 10.0))
        self.assertEqual(model.dit.text_cache_calls, 2)
        self.assertEqual(model.dit.speaker_cache_calls, 2)
        self.assertEqual(model.dit.forward_calls, 4)

    def test_frame_level_speaker_mask_is_reduced_to_patches(self):
        class FakeDiT(torch.nn.Module):
            speaker_patch_size = 4

            def __init__(self):
                super().__init__()
                self.seen_mask = None
                self.cache_latent = None

            def encode_text(self, conditioning_ids, text_mask, language_ids=None):
                del conditioning_ids, text_mask
                del language_ids
                return torch.zeros(1, 1, 1)

            def project_text_kv_cache(self, text_state):
                del text_state
                return []

            def get_kv_cache_speaker(self, speaker_latent, speaker_mask=None):
                self.cache_latent = speaker_latent
                self.cache_mask = speaker_mask
                return []

            def forward(self, **kwargs):
                self.seen_mask = kwargs["speaker_mask"]
                return torch.zeros_like(kwargs["x"])

        model = FlowMatchingEchoDiT.__new__(FlowMatchingEchoDiT)
        torch.nn.Module.__init__(model)
        model.latent_size = 2
        model.speaker_patch_size = 4
        model.use_speaker_conditioning = True
        model.cfg_dropout_text = 0.0
        model.cfg_dropout_speaker = 0.0
        model.dit = FakeDiT()
        model.eval()

        model(
            latents=torch.zeros(1, 2, 2),
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            timesteps=torch.zeros(1),
            speaker_latent=torch.ones(1, 2, 8),
            speaker_mask=torch.tensor([[True, False, True, False, False, False, False, False]]),
        )

        torch.testing.assert_close(model.dit.seen_mask, torch.tensor([[True, False]]))
        torch.testing.assert_close(model.dit.cache_mask, torch.tensor([[True, False]]))
        torch.testing.assert_close(
            model.dit.cache_latent,
            torch.tensor(
                [
                    [
                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
        )

    def test_unmasked_inference_uses_none_as_the_all_valid_sentinel(self):
        class FakeDiT(torch.nn.Module):
            speaker_patch_size = 4

            def __init__(self):
                super().__init__()
                self.text_mask = object()
                self.speaker_mask = object()
                self.in_proj = torch.nn.Linear(1, 1, bias=False)

            def encode_text(self, conditioning_ids, text_mask, language_ids=None):
                del conditioning_ids
                del language_ids
                self.text_mask = text_mask
                return torch.zeros(1, 1, 1)

            def project_text_kv_cache(self, text_state):
                del text_state
                return []

            def get_kv_cache_speaker(self, speaker_latent, speaker_mask=None):
                del speaker_latent
                self.speaker_mask = speaker_mask
                return []

            def forward(self, **kwargs):
                self.forward_text_mask = kwargs["text_mask"]
                self.forward_speaker_mask = kwargs["speaker_mask"]
                return torch.zeros_like(kwargs["x"])

        model = FlowMatchingEchoDiT.__new__(FlowMatchingEchoDiT)
        torch.nn.Module.__init__(model)
        model.latent_size = 2
        model.speaker_patch_size = 4
        model.use_speaker_conditioning = False
        model.cfg_dropout_text = 0.0
        model.cfg_dropout_speaker = 0.0
        model.dit = FakeDiT()
        model.eval()

        model(
            latents=torch.zeros(1, 2, 2),
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            timesteps=torch.zeros(1),
        )

        self.assertIsNone(model.dit.text_mask)
        self.assertIsNone(model.dit.speaker_mask)
        self.assertIsNone(model.dit.forward_text_mask)
        self.assertIsNone(model.dit.forward_speaker_mask)


if __name__ == "__main__":
    unittest.main()
