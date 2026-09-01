"""Speaker encoder patch-layout regression tests."""

import unittest
from unittest.mock import patch

import torch

import nar_vae.models.dit as dit_module
from nar_vae.models.dit import SpeakerEncoder


class SpeakerEncoderTest(unittest.TestCase):
    def test_patches_group_adjacent_frames_for_every_channel(self):
        encoder = SpeakerEncoder(
            latent_size=2,
            patch_size=2,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=4,
            norm_eps=1e-6,
        )
        with torch.no_grad():
            encoder.in_proj.weight.copy_(torch.eye(4))
            encoder.in_proj.bias.zero_()

        latent = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]])
        encoded = encoder(latent) * 6.0

        expected = torch.tensor([[[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]])
        torch.testing.assert_close(encoded, expected)

    def test_frame_count_must_align_to_patch_size(self):
        encoder = SpeakerEncoder(
            latent_size=2,
            patch_size=4,
            model_size=8,
            num_layers=0,
            num_heads=1,
            intermediate_size=8,
            norm_eps=1e-6,
        )

        with self.assertRaisesRegex(ValueError, "multiple of patch size"):
            encoder(torch.zeros(1, 2, 5))

    def test_masked_patch_cannot_influence_a_later_valid_patch(self):
        torch.manual_seed(1234)
        encoder = SpeakerEncoder(
            latent_size=1,
            patch_size=1,
            model_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=8,
            norm_eps=1e-6,
        )
        mask = torch.tensor([[False, True]])
        original = encoder(torch.tensor([[[1.0, 2.0]]]), mask)
        changed_masked_patch = encoder(torch.tensor([[[1000.0, 2.0]]]), mask)

        torch.testing.assert_close(original[:, 1], changed_masked_patch[:, 1])

    def test_causal_and_padding_masks_are_composed_for_minimum_torch_support(self):
        encoder = SpeakerEncoder(
            latent_size=1,
            patch_size=1,
            model_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=8,
            norm_eps=1e-6,
        )
        real_sdpa = torch.nn.functional.scaled_dot_product_attention
        calls = []

        def strict_minimum_version_sdpa(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            **kwargs,
        ):
            if attn_mask is not None and is_causal:
                raise RuntimeError("Explicit attn_mask should not be set when is_causal=True")
            calls.append((attn_mask.detach().clone(), is_causal))
            return real_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )

        with patch.object(
            dit_module.F,
            "scaled_dot_product_attention",
            side_effect=strict_minimum_version_sdpa,
        ):
            encoder(
                torch.tensor([[[1.0, 2.0, 1000.0]]]),
                torch.tensor([[True, True, False]]),
            )

        self.assertEqual(len(calls), 1)
        composed_mask, is_causal = calls[0]
        self.assertFalse(is_causal)
        torch.testing.assert_close(
            composed_mask,
            torch.tensor([[[[True, False, False], [True, True, False], [True, True, False]]]]),
        )

    def test_masked_causal_path_cannot_see_a_future_valid_patch(self):
        torch.manual_seed(5678)
        encoder = SpeakerEncoder(
            latent_size=1,
            patch_size=1,
            model_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=8,
            norm_eps=1e-6,
        )
        mask = torch.tensor([[True, True, False]])

        original = encoder(torch.tensor([[[1.0, 2.0, 0.0]]]), mask)
        changed_future = encoder(torch.tensor([[[1.0, 1000.0, 0.0]]]), mask)

        torch.testing.assert_close(original[:, 0], changed_future[:, 0])

    def test_mask_shape_must_match_patch_count(self):
        encoder = SpeakerEncoder(
            latent_size=1,
            patch_size=1,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=4,
            norm_eps=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "Speaker mask must have shape"):
            encoder(torch.ones(1, 1, 2), torch.ones(1, 1, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
