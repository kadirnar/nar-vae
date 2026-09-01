"""Exact temporal target-patching tests for the diffusion backbone."""

import unittest

import torch

from nar_vae.models.dit import EchoDiT


def tiny_dit(*, patch_size: int, duration_alignment: bool = False) -> EchoDiT:
    model_size = 4 if patch_size == 2 else 6
    return EchoDiT(
        latent_size=2,
        model_size=model_size,
        num_layers=0,
        num_heads=1,
        intermediate_size=model_size * 2,
        text_vocab_size=8,
        text_model_size=2,
        text_num_layers=0,
        text_num_heads=1,
        text_intermediate_size=4,
        speaker_patch_size=1,
        speaker_model_size=2,
        speaker_num_layers=0,
        speaker_num_heads=1,
        speaker_intermediate_size=4,
        target_patch_size=patch_size,
        timestep_embed_size=4,
        adaln_rank=2,
        use_duration_alignment=duration_alignment,
    )


class TargetPatchingTest(unittest.TestCase):
    @staticmethod
    def run_dit(
        model: EchoDiT,
        latents: torch.Tensor,
        *,
        latent_mask: torch.Tensor | None = None,
        frame_text_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return model(
            x=latents,
            t=torch.zeros(latents.shape[0]),
            text_mask=None,
            speaker_mask=None,
            kv_cache_text=[],
            kv_cache_speaker=[],
            latent_mask=latent_mask,
            frame_text_state=frame_text_state,
        )

    def test_patch_projection_retains_every_latent_coordinate_and_odd_length(self):
        model = tiny_dit(patch_size=3)
        model.out_norm = torch.nn.Identity()
        with torch.no_grad():
            model.in_proj.weight.copy_(torch.eye(6))
            model.in_proj.bias.zero_()
            model.out_proj.weight.copy_(torch.eye(6))
            model.out_proj.bias.zero_()

        latents = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]])
        output = self.run_dit(model, latents)

        torch.testing.assert_close(output, latents)

    def test_duration_state_uses_the_same_exact_pack_and_unpack_layout(self):
        model = tiny_dit(patch_size=2, duration_alignment=True)
        model.out_norm = torch.nn.Identity()
        with torch.no_grad():
            model.in_proj.weight.zero_()
            model.in_proj.bias.zero_()
            model.frame_text_proj.weight.copy_(torch.eye(4))
            model.out_proj.weight.copy_(torch.eye(4))
            model.out_proj.bias.zero_()

        latents = torch.zeros(1, 2, 3)
        frame_text = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]])
        output = self.run_dit(model, latents, frame_text_state=frame_text)

        torch.testing.assert_close(output, frame_text.transpose(1, 2))

    def test_patch_size_one_preserves_the_unpatched_projection_shapes(self):
        model = EchoDiT(
            latent_size=2,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=8,
            text_vocab_size=8,
            text_model_size=2,
            text_num_layers=0,
            text_num_heads=1,
            text_intermediate_size=4,
            target_patch_size=1,
            timestep_embed_size=4,
            adaln_rank=2,
        )

        self.assertEqual(model.in_proj.in_features, 2)
        self.assertEqual(model.out_proj.out_features, 2)
        self.assertEqual(self.run_dit(model, torch.randn(2, 2, 5)).shape, (2, 2, 5))

    def test_packed_padding_mask_uses_any_valid_frame_and_zeros_invalid_coordinates(self):
        class RecordingBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.state = None
                self.mask = None

            def forward(self, state, **kwargs):
                self.state = state.detach().clone()
                self.mask = kwargs["latent_mask"].detach().clone()
                return state

        model = tiny_dit(patch_size=2)
        recorder = RecordingBlock()
        model.blocks = torch.nn.ModuleList((recorder,))
        with torch.no_grad():
            model.in_proj.weight.copy_(torch.eye(4))
            model.in_proj.bias.zero_()
        latents = torch.tensor([[[1.0, 2.0, 3.0, 99.0, 99.0], [4.0, 5.0, 6.0, 99.0, 99.0]]])

        self.run_dit(
            model,
            latents,
            latent_mask=torch.tensor([[True, True, True, False, False]]),
        )

        torch.testing.assert_close(recorder.mask, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(recorder.state[0, 1], torch.tensor([3.0, 6.0, 0.0, 0.0]))
        torch.testing.assert_close(recorder.state[0, 2], torch.zeros(4))

    def test_target_patch_size_must_be_a_positive_integer(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    tiny_dit(patch_size=invalid)


if __name__ == "__main__":
    unittest.main()
