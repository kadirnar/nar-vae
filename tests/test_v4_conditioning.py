"""Focused NAR-VAE v4 conditioning and non-contiguous alignment tests."""

import unittest
from unittest.mock import patch

import torch

from nar_vae.languages import LANGUAGE_COUNT, language_id
from nar_vae.models.dit import EchoDiT, TextEncoder
from nar_vae.models.flow_matching import FlowMatchingEchoDiT


def tiny_flow(*, language: bool = False, speaker: bool = False, mas: bool = False):
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
        timestep_embed_size=8,
        adaln_rank=4,
        cfg_dropout_text=1.0,
        cfg_dropout_speaker=1.0,
        use_language_conditioning=language,
        supported_languages=("en", "es") if language else None,
        use_speaker_conditioning=speaker,
        supported_language_pairs=((("en", "en"), ("es", "en")) if language and speaker else None),
        use_duration_predictor=mas,
        duration_predictor_hidden_size=6,
        duration_predictor_num_layers=1,
        use_mas_duration=mas,
        duration_alignment_hidden_size=3,
    )


class V4ConditioningTest(unittest.TestCase):
    def test_text_encoder_accepts_stable_per_token_language_ids(self):
        encoder = TextEncoder(
            vocab_size=8,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=8,
            norm_eps=1e-6,
            num_languages=LANGUAGE_COUNT,
        )
        english = language_id("en")
        spanish = language_id("es")
        with torch.no_grad():
            encoder.text_embedding.weight.zero_()
            encoder.language_embedding.weight.zero_()
            encoder.language_embedding.weight[english].fill_(1.0)
            encoder.language_embedding.weight[spanish].fill_(2.0)

        state = encoder(
            torch.tensor([[1, 2, 3]]),
            language_ids=torch.tensor([[english, spanish, english]]),
        )

        torch.testing.assert_close(
            state,
            torch.tensor([[[1.0] * 4, [2.0] * 4, [1.0] * 4]]),
        )

    def test_language_conditioned_model_requires_an_explicit_global_target(self):
        model = tiny_flow(language=True).eval()

        with self.assertRaisesRegex(ValueError, "language_ids are required"):
            model.encode_inference_conditioning(torch.tensor([[1, 2]]))

    def test_global_target_language_is_added_to_timestep_adaln_conditioning(self):
        class RecordingCondition(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input = None

            def forward(self, state):
                self.input = state.detach().clone()
                return state.new_zeros((state.shape[0], 12))

        model = EchoDiT(
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
            timestep_embed_size=4,
            adaln_rank=2,
            num_languages=LANGUAGE_COUNT,
        )
        recorder = RecordingCondition()
        model.cond_module = recorder
        english = language_id("en")
        spanish = language_id("es")
        with torch.no_grad():
            model.global_language_embedding.weight.zero_()
            model.global_language_embedding.weight[english].fill_(1.0)
            model.global_language_embedding.weight[spanish].fill_(3.0)
        language_state = model.encode_global_language(
            torch.tensor([english, spanish]),
            batch_size=2,
            device=torch.device("cpu"),
        )

        model(
            x=torch.zeros(2, 2, 2),
            t=torch.full((2,), 0.5),
            text_mask=None,
            speaker_mask=None,
            kv_cache_text=[],
            kv_cache_speaker=[],
            global_language_state=language_state,
        )

        torch.testing.assert_close(recorder.input[1] - recorder.input[0], torch.full((4,), 2.0))

    def test_cfg_uses_learned_encoded_nulls_after_one_real_encoder_pass(self):
        model = tiny_flow(speaker=True).eval()
        with torch.no_grad():
            model.null_text_embed.fill_(7.0)
            model.null_speaker_state.fill_(9.0)
        with (
            patch.object(model.dit, "encode_text", wraps=model.dit.encode_text) as encode_text,
            patch.object(
                model.dit,
                "encode_speaker",
                wraps=model.dit.encode_speaker,
            ) as encode_speaker,
        ):
            encoded = model.encode_inference_conditioning(
                torch.tensor([[1, 2, 3]]),
                speaker_latent=torch.randn(1, 4, 4),
                cfg_mode="independent",
            )

        self.assertEqual(encode_text.call_count, 1)
        self.assertEqual(encode_speaker.call_count, 1)
        fused = encoded.variants[0]
        torch.testing.assert_close(fused.text_state[1, :1], torch.full((1, 8), 7.0))
        self.assertEqual(fused.text_state[1, 1:].count_nonzero().item(), 0)
        torch.testing.assert_close(fused.text_mask[1], torch.tensor([True, False, False]))
        torch.testing.assert_close(fused.speaker_state[2, :1], torch.full((1, 8), 9.0))
        self.assertEqual(fused.speaker_state[2, 1:].count_nonzero().item(), 0)
        torch.testing.assert_close(
            fused.speaker_mask[2],
            torch.tensor([True, False, False]),
        )

    def test_noncontiguous_alignment_scatter_preserves_full_public_shapes(self):
        model = tiny_flow(mas=True).eval()
        alignment_mask = torch.tensor([[False, True, False, True, False]])
        velocity, duration = model(
            latents=torch.randn(1, 4, 6),
            conditioning_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            timesteps=torch.tensor([0.5]),
            attention_mask=torch.ones(1, 5, dtype=torch.bool),
            alignment_mask=alignment_mask,
            use_cfg_dropout=False,
            return_duration_prediction=True,
            return_duration_alignment=True,
            duration_target_latents=torch.randn(1, 4, 6),
        )

        self.assertEqual(velocity.shape, (1, 4, 6))
        self.assertEqual(duration.token_durations.shape, (1, 5))
        self.assertEqual(duration.log_likelihoods.shape, (1, 5, 6))
        self.assertEqual(duration.hard_alignment.shape, (1, 5, 6))
        self.assertEqual(duration.token_durations[~alignment_mask].count_nonzero().item(), 0)
        self.assertEqual(duration.log_likelihoods[~alignment_mask].count_nonzero().item(), 0)
        self.assertEqual(duration.hard_alignment[~alignment_mask].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
