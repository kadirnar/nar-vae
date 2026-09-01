"""Voice-cloning API tests that do not require a large checkpoint."""

import unittest
from inspect import signature
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from nar_vae.dacvae_encoding import derive_dacvae_posterior_seed
from nar_vae.inference import FlowMatchingTTSInference, VoiceCloningUnsupportedError
from nar_vae.models.flow_matching import FlowMatchingEchoDiT, PreparedCFGConditioning
from nar_vae.solvers.ode_solver import ODESolver
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS

CODEC_SHA256 = "c" * 64


class FakeCodec:
    def __init__(self):
        self.last_waveform = None
        self.last_seed = None
        self.encode = MagicMock()


def fake_seeded_encode(codec, waveform, *, seed):
    codec.last_waveform = waveform
    codec.last_seed = seed
    return torch.ones(1, 2, 5, device=waveform.device)


class ValueConditioning:
    """Minimal prepared-conditioning protocol used by CFG formula tests."""

    def __init__(self, values):
        self.values = values

    def slice_batch(self, start, stop):
        return ValueConditioning(self.values[start:stop])


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
    runtime.model_manifest = SimpleNamespace(
        representation={"codec_sha256": CODEC_SHA256},
    )
    return runtime


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

        with patch(
            "nar_vae.inference.encode_dacvae_posterior_seeded",
            side_effect=fake_seeded_encode,
        ):
            latent = runtime.encode_reference_audio(stereo, sample_rate=24000)

        self.assertEqual(tuple(runtime.dacvae.last_waveform.shape), (1, 1, 48000))
        self.assertEqual(
            runtime.dacvae.last_seed,
            derive_dacvae_posterior_seed(
                runtime.dacvae.last_waveform[0],
                codec_sha256=CODEC_SHA256,
            ),
        )
        self.assertEqual(tuple(latent.shape), (1, 2, 8))
        torch.testing.assert_close(latent[..., :5], torch.ones(1, 2, 5))
        torch.testing.assert_close(latent[..., 5:], torch.zeros(1, 2, 3))
        runtime.dacvae.encode.assert_not_called()

    def test_reference_is_cropped_to_the_shared_duration_default(self):
        runtime = make_runtime(supported=True)
        runtime.max_reference_seconds = DEFAULT_MAX_REFERENCE_SECONDS
        overlong_samples = int((DEFAULT_MAX_REFERENCE_SECONDS + 1.0) * runtime.sample_rate)

        with patch(
            "nar_vae.inference.encode_dacvae_posterior_seeded",
            side_effect=fake_seeded_encode,
        ):
            runtime.encode_reference_audio(
                torch.ones(overlong_samples),
                sample_rate=runtime.sample_rate,
            )

        self.assertEqual(
            runtime.dacvae.last_waveform.shape[-1],
            int(DEFAULT_MAX_REFERENCE_SECONDS * runtime.sample_rate),
        )
        runtime.dacvae.encode.assert_not_called()

    def test_short_reference_seed_binds_explicit_hop_padding_not_global_rng(self):
        runtime = make_runtime(supported=True)
        reference = torch.ones(100)

        with patch(
            "nar_vae.inference.encode_dacvae_posterior_seeded",
            side_effect=fake_seeded_encode,
        ):
            torch.manual_seed(1)
            first = runtime.encode_reference_audio(reference, sample_rate=runtime.sample_rate)
            first_seed = runtime.dacvae.last_seed
            torch.manual_seed(999)
            second = runtime.encode_reference_audio(reference, sample_rate=runtime.sample_rate)

        self.assertEqual(tuple(runtime.dacvae.last_waveform.shape), (1, 1, runtime.hop_length))
        self.assertEqual(first_seed, runtime.dacvae.last_seed)
        self.assertEqual(
            first_seed,
            derive_dacvae_posterior_seed(
                torch.nn.functional.pad(reference.unsqueeze(0), (0, runtime.hop_length - 100)),
                codec_sha256=CODEC_SHA256,
            ),
        )
        torch.testing.assert_close(first, second)
        runtime.dacvae.encode.assert_not_called()

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

    def test_lightweight_runtime_preserves_the_legacy_fp32_speaker_contract(self):
        runtime = make_runtime(supported=True)

        aligned = runtime._align_speaker_latent(torch.ones(1, 2, 5, dtype=torch.float64))

        self.assertEqual(aligned.dtype, torch.float32)
        self.assertEqual(tuple(aligned.shape), (1, 2, 8))

    def test_bfloat16_acoustic_state_is_cast_at_speaker_and_codec_boundaries(self):
        runtime = make_runtime(supported=True)
        runtime.flow_model = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)

        aligned = runtime._align_speaker_latent(torch.ones(1, 2, 4))

        self.assertEqual(aligned.dtype, torch.bfloat16)
        runtime.dacvae = torch.nn.Linear(1, 1, bias=False)
        decoded_inputs = []

        def decode(latents):
            decoded_inputs.append(latents)
            return latents

        runtime._decode = decode
        decoded = runtime._decode_generated_latents(aligned)
        self.assertEqual(decoded_inputs[0].dtype, torch.float32)
        self.assertEqual(decoded.dtype, torch.float32)

    def test_cfg_null_reference_is_a_learned_encoded_state(self):
        model = FlowMatchingEchoDiT(
            latent_size=2,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=8,
            text_vocab_size=8,
            text_model_size=4,
            text_num_layers=0,
            text_num_heads=1,
            text_intermediate_size=8,
            speaker_patch_size=4,
            speaker_model_size=4,
            speaker_num_layers=0,
            speaker_num_heads=1,
            speaker_intermediate_size=8,
            timestep_embed_size=4,
            adaln_rank=2,
            use_speaker_conditioning=True,
        ).eval()
        with torch.no_grad():
            model.null_speaker_state.fill_(5.0)
        reference = torch.ones(1, 2, 8)
        with patch.object(
            model.dit,
            "encode_speaker",
            wraps=model.dit.encode_speaker,
        ) as encode_speaker:
            encoded = model.encode_inference_conditioning(
                torch.ones(1, 2, dtype=torch.long),
                speaker_latent=reference,
                speaker_mask=torch.tensor([[True, False]]),
                cfg_mode="joint",
            )

        self.assertEqual(encode_speaker.call_count, 1)
        fused = encoded.variants[0]
        self.assertEqual(fused.speaker_state.shape, (2, 3, 4))
        torch.testing.assert_close(fused.speaker_state[1, 0], torch.full((4,), 5.0))
        self.assertEqual(fused.speaker_state[1, 1:].count_nonzero().item(), 0)
        torch.testing.assert_close(
            fused.speaker_mask[1],
            torch.tensor([True, False, False]),
        )

    def test_independent_cfg_is_fused_without_changing_the_formula(self):
        model = FlowMatchingEchoDiT.__new__(FlowMatchingEchoDiT)
        torch.nn.Module.__init__(model)
        forward_calls = []
        prepared = PreparedCFGConditioning(
            mode="independent",
            branch_count=3,
            variants=(ValueConditioning(torch.tensor([3.0, 1.0, 2.0])),),
            conditional=ValueConditioning(torch.tensor([3.0])),
        )

        def fake_encode(self, *args, **kwargs):
            del self, args, kwargs
            return object()

        def fake_finalize(self, encoded, *, token_durations=None):
            del self, encoded, token_durations
            return prepared.conditional, prepared

        def fake_forward_prepared(self, latents, timesteps, conditioning):
            del self, timesteps
            forward_calls.append(latents.shape[0])
            return conditioning.values[:, None, None].expand_as(latents)

        model.encode_inference_conditioning = MethodType(fake_encode, model)
        model.finalize_inference_conditioning = MethodType(fake_finalize, model)
        model.forward_prepared = MethodType(fake_forward_prepared, model)
        output = model.forward_with_cfg(
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.long),
            torch.zeros(1),
            cfg_scale=2.0,
            cfg_mode="independent",
            cfg_scale_text=2.5,
            cfg_scale_speaker=2.0,
            speaker_latent=torch.ones(1, 2, 4),
            fuse_cfg_branches=True,
        )

        self.assertEqual(forward_calls, [3])
        torch.testing.assert_close(output, torch.full_like(output, 10.0))

        forward_calls.clear()
        sequential_output = model.forward_with_cfg(
            torch.zeros(1, 2, 2),
            torch.ones(1, 2, dtype=torch.long),
            torch.zeros(1),
            cfg_scale=2.0,
            cfg_mode="independent",
            cfg_scale_text=2.5,
            cfg_scale_speaker=2.0,
            speaker_latent=torch.ones(1, 2, 4),
        )

        self.assertEqual(forward_calls, [1, 1, 1])
        torch.testing.assert_close(sequential_output, output)

    def test_alternating_cfg_matches_prepared_formulas(self):
        model = FlowMatchingEchoDiT.__new__(FlowMatchingEchoDiT)
        torch.nn.Module.__init__(model)

        def fake_forward_prepared(self, latents, timesteps, conditioning):
            del self, timesteps
            return conditioning.values[:, None, None].expand_as(latents)

        model.forward_prepared = MethodType(fake_forward_prepared, model)
        conditional = ValueConditioning(torch.tensor([3.0]))
        prepared = PreparedCFGConditioning(
            mode="alternating",
            branch_count=2,
            variants=(
                ValueConditioning(torch.tensor([3.0, 1.0])),
                ValueConditioning(torch.tensor([3.0, 2.0])),
            ),
            conditional=conditional,
        )

        def fake_encode(self, *args, **kwargs):
            del self, args, kwargs
            return object()

        def fake_finalize(self, encoded, *, token_durations=None):
            del self, encoded, token_durations
            return conditional, prepared

        model.encode_inference_conditioning = MethodType(fake_encode, model)
        model.finalize_inference_conditioning = MethodType(fake_finalize, model)
        common = {
            "latents": torch.zeros(1, 2, 2),
            "conditioning_ids": torch.ones(1, 2, dtype=torch.long),
            "timesteps": torch.zeros(1),
            "cfg_scale": 2.0,
            "cfg_mode": "alternating",
            "cfg_scale_text": 2.5,
            "cfg_scale_speaker": 2.0,
            "speaker_latent": torch.ones(1, 2, 4),
        }

        for step_idx, expected in ((0, 6.0), (1, 4.0)):
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
                torch.testing.assert_close(direct, torch.full_like(direct, expected))

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
        model.null_text_embed = torch.nn.Parameter(torch.zeros(1, 1, 1))
        model.null_speaker_state = torch.nn.Parameter(torch.zeros(1, 1, 4))
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
