"""Focused tests for fixed-token reference resampling."""

from __future__ import annotations

import unittest

import torch

from nar_vae.models.dit import EchoDiT
from nar_vae.models.flow_matching import FlowMatchingEchoDiT, create_flow_matching_echodit
from nar_vae.models.reference_resampler import ReferenceResampler


def _tiny_echo(*, summary_tokens: int = 0, use_speaker: bool = True) -> EchoDiT:
    return EchoDiT(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=20,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        speaker_num_summary_tokens=summary_tokens,
        timestep_embed_size=8,
        adaln_rank=4,
        use_speaker_conditioning=use_speaker,
    )


def _tiny_flow(*, summary_tokens: int = 0) -> FlowMatchingEchoDiT:
    return FlowMatchingEchoDiT(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=20,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        speaker_num_summary_tokens=summary_tokens,
        timestep_embed_size=8,
        adaln_rank=4,
        use_speaker_conditioning=True,
    )


class ReferenceResamplerTest(unittest.TestCase):
    def test_variable_reference_lengths_produce_one_fixed_downstream_shape(self):
        torch.manual_seed(1)
        resampler = ReferenceResampler(hidden_size=16, num_queries=6, num_heads=4)
        downstream = torch.nn.Linear(16, 3)

        short = downstream(resampler(torch.randn(2, 3, 16)))
        long = downstream(resampler(torch.randn(2, 29, 16)))

        self.assertEqual(short.shape, (2, 6, 3))
        self.assertEqual(long.shape, (2, 6, 3))
        self.assertEqual(resampler.GLOBAL_TOKEN_INDEX, 0)

    def test_masked_states_cannot_change_any_resampled_token(self):
        torch.manual_seed(2)
        resampler = ReferenceResampler(hidden_size=12, num_queries=4, num_heads=3).eval()
        states = torch.randn(2, 7, 12)
        mask = torch.tensor(
            [
                [True, True, True, False, False, False, False],
                [True, False, True, True, False, False, False],
            ]
        )
        changed = states.clone()
        changed[~mask] = float("nan")

        original_output = resampler(states, mask)
        changed_output = resampler(changed, mask)

        torch.testing.assert_close(original_output, changed_output, rtol=0.0, atol=0.0)

    def test_global_query_has_a_direct_masked_mean_timbre_path(self):
        torch.manual_seed(3)
        resampler = ReferenceResampler(hidden_size=8, num_queries=3, num_heads=2).eval()
        with torch.no_grad():
            resampler.out_proj.weight.zero_()
            for module in resampler.mlp:
                if isinstance(module, torch.nn.Linear):
                    module.weight.zero_()
                    module.bias.zero_()

        quiet = torch.zeros(1, 4, 8)
        timbre_shift = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0])
        shifted = quiet + timbre_shift
        quiet_tokens = resampler(quiet)
        shifted_tokens = resampler(shifted)

        self.assertFalse(torch.equal(quiet_tokens[:, 0], shifted_tokens[:, 0]))
        torch.testing.assert_close(quiet_tokens[:, 1:], shifted_tokens[:, 1:], rtol=0.0, atol=0.0)

    def test_gradients_reach_inputs_queries_and_low_parameter_adapter(self):
        torch.manual_seed(4)
        resampler = ReferenceResampler(
            hidden_size=16,
            num_queries=4,
            num_heads=4,
            mlp_ratio=0.5,
        )
        states = torch.randn(2, 5, 16, requires_grad=True)
        mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])

        resampler(states, mask).square().mean().backward()

        self.assertIsNotNone(states.grad)
        self.assertGreater(states.grad[mask].abs().sum().item(), 0.0)
        torch.testing.assert_close(states.grad[~mask], torch.zeros_like(states.grad[~mask]))
        self.assertIsNotNone(resampler.query_tokens.grad)
        self.assertGreater(resampler.query_tokens.grad.abs().sum().item(), 0.0)
        self.assertLess(sum(parameter.numel() for parameter in resampler.parameters()), 2000)

    def test_constructor_and_forward_arguments_are_strictly_validated(self):
        invalid_constructors = (
            {"hidden_size": 0},
            {"hidden_size": 8, "num_queries": 0},
            {"hidden_size": 8, "num_heads": 0},
            {"hidden_size": 10, "num_heads": 3},
            {"hidden_size": 8, "mlp_ratio": 0.0},
            {"hidden_size": 8, "norm_eps": float("nan")},
        )
        for kwargs in invalid_constructors:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReferenceResampler(**kwargs)

        resampler = ReferenceResampler(hidden_size=8, num_queries=2, num_heads=2)
        with self.assertRaisesRegex(ValueError, r"\[B, S, H\]"):
            resampler(torch.zeros(2, 8))
        with self.assertRaisesRegex(ValueError, "width"):
            resampler(torch.zeros(2, 3, 7))
        with self.assertRaisesRegex(ValueError, "floating-point"):
            resampler(torch.zeros(2, 3, 8, dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "speaker_mask must have shape"):
            resampler(torch.zeros(2, 3, 8), torch.ones(2, 2, dtype=torch.bool))
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            resampler(torch.zeros(2, 3, 8), torch.tensor([[True, False, False], [False] * 3]))


class ReferenceResamplerIntegrationTest(unittest.TestCase):
    def test_zero_summary_tokens_preserve_the_legacy_speaker_path_and_state_dict(self):
        torch.manual_seed(10)
        model = _tiny_echo(summary_tokens=0).eval()
        speaker_latent = torch.randn(2, 4, 6)
        patch_mask = torch.tensor([[True, True, False], [True, False, False]])

        expected = model.speaker_norm(model.speaker_encoder(speaker_latent, patch_mask))
        actual = model.encode_speaker(speaker_latent, patch_mask)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertIsNone(model.speaker_resampler)
        self.assertFalse(any("speaker_resampler" in key for key in model.state_dict()))

        wrapped = _tiny_flow(summary_tokens=0)
        self.assertNotIn("speaker_resampler_version", wrapped.state_dict())
        self.assertNotIn("speaker_num_summary_tokens_metadata", wrapped.state_dict())
        self.assertFalse(any("speaker_resampler" in key for key in wrapped.state_dict()))

    def test_enabled_echo_speaker_encoder_has_fixed_length_after_variable_references(self):
        torch.manual_seed(11)
        model = _tiny_echo(summary_tokens=3).eval()

        short = model.encode_speaker(torch.randn(2, 4, 4))
        long = model.encode_speaker(torch.randn(2, 4, 18))

        self.assertEqual(short.shape, (2, 3, 8))
        self.assertEqual(long.shape, (2, 3, 8))
        self.assertIsNotNone(model.speaker_resampler)
        self.assertTrue(any("speaker_resampler" in key for key in model.state_dict()))

    def test_flow_metadata_factory_and_invalid_topologies_are_explicit(self):
        model = _tiny_flow(summary_tokens=3)
        self.assertEqual(model.speaker_resampler_version.item(), 1)
        self.assertEqual(model.speaker_num_summary_tokens_metadata.item(), 3)

        factory_model = create_flow_matching_echodit(
            latent_size=4,
            model_size=8,
            num_layers=1,
            num_heads=2,
            intermediate_size=16,
            text_vocab_size=20,
            text_model_size=8,
            text_num_layers=1,
            text_num_heads=2,
            text_intermediate_size=16,
            speaker_patch_size=2,
            speaker_model_size=8,
            speaker_num_layers=1,
            speaker_num_heads=2,
            speaker_intermediate_size=16,
            speaker_num_summary_tokens=2,
            timestep_embed_size=8,
            adaln_rank=4,
            use_speaker_conditioning=True,
        )
        self.assertEqual(factory_model.dit.speaker_num_summary_tokens, 2)

        for invalid in (-1, True, 1.5):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(
                    ValueError,
                    "speaker_num_summary_tokens",
                ),
            ):
                _tiny_echo(summary_tokens=invalid)
        with self.assertRaisesRegex(ValueError, "use_speaker_conditioning=True"):
            _tiny_echo(summary_tokens=2, use_speaker=False)

    def test_real_and_cfg_prepared_conditioning_have_consistent_fixed_masks_and_kv(self):
        torch.manual_seed(12)
        model = _tiny_flow(summary_tokens=3).eval()
        text = torch.tensor([[1, 2]])
        speaker_latent = torch.randn(1, 4, 8)
        frame_mask = torch.tensor([[True, True, True, True, False, False, False, False]])

        prepared = model.prepare_inference_conditioning(
            text,
            speaker_latent=speaker_latent,
            speaker_mask=frame_mask,
        )
        self.assertIsNone(prepared.speaker_mask)
        self.assertEqual(prepared.kv_cache_speaker[0][0].shape[1], 3)

        encoded = model.encode_inference_conditioning(
            text,
            speaker_latent=speaker_latent,
            speaker_mask=frame_mask,
            cfg_mode="independent",
        )
        self.assertEqual(encoded.conditional.speaker_state.shape, (1, 3, 8))
        self.assertIsNone(encoded.conditional.speaker_mask)
        fused = encoded.variants[0]
        self.assertEqual(fused.speaker_state.shape, (3, 3, 8))
        torch.testing.assert_close(
            fused.speaker_mask,
            torch.tensor([[True, True, True], [True, True, True], [True, False, False]]),
        )

        _, prepared_cfg = model.finalize_inference_conditioning(encoded)
        self.assertIsNotNone(prepared_cfg)
        self.assertEqual(prepared_cfg.variants[0].speaker_mask.shape, (3, 3))
        self.assertEqual(prepared_cfg.variants[0].kv_cache_speaker[0][0].shape[1], 3)

        encoded_null = model.encode_inference_conditioning(text)
        self.assertEqual(encoded_null.conditional.speaker_state.shape, (1, 3, 8))
        torch.testing.assert_close(
            encoded_null.conditional.speaker_mask,
            torch.tensor([[True, False, False]]),
        )


if __name__ == "__main__":
    unittest.main()
