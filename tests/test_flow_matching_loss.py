"""Flow-matching timestep and padded-batch regression tests."""

import unittest
from unittest.mock import patch

import torch

from vyvotts.losses.flow_matching_loss import (
    FlowMatchingLoss,
    sample_logit_normal,
    sample_stratified_logit_normal,
)
from vyvotts.models.dit import JointAttention, RMSNorm, precompute_freqs_cis


class RecordingVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return torch.zeros_like(kwargs["latents"])


class RecordingDurationModel(RecordingVelocityModel):
    def __init__(self, duration_prediction):
        super().__init__()
        self.duration_prediction = duration_prediction

    def forward(self, **kwargs):
        self.kwargs = kwargs
        velocity = torch.zeros_like(kwargs["latents"])
        return velocity, self.duration_prediction.to(velocity.device)


class StratifiedLogitNormalTest(unittest.TestCase):
    def test_maps_stratified_probabilities_through_inverse_normal_cdf(self):
        draws = torch.tensor([0.25, 0.5, 0.75, 0.125])
        identity_permutation = torch.arange(4)
        with (
            patch(
                "vyvotts.losses.flow_matching_loss.torch.rand",
                return_value=draws,
            ),
            patch(
                "vyvotts.losses.flow_matching_loss.torch.randperm",
                return_value=identity_permutation,
            ),
        ):
            actual = sample_stratified_logit_normal(
                4,
                torch.device("cpu"),
                loc=0.3,
                scale=1.2,
                num_strata=10,
            )

        probabilities = (torch.arange(4) + draws) / 4
        normal_quantiles = (2.0**0.5) * torch.erfinv(2.0 * probabilities - 1.0)
        expected = torch.sigmoid(normal_quantiles * 1.2 + 0.3)
        torch.testing.assert_close(actual, expected)

    def test_rejects_invalid_sampling_arguments(self):
        device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            sample_stratified_logit_normal(0, device)
        with self.assertRaisesRegex(ValueError, "num_strata"):
            sample_stratified_logit_normal(1, device, num_strata=0)
        with self.assertRaisesRegex(ValueError, "scale"):
            sample_stratified_logit_normal(1, device, scale=-1.0)
        with self.assertRaisesRegex(ValueError, "scale"):
            sample_logit_normal(1, device, scale=0.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            sample_logit_normal(1, device, loc=float("nan"))

    def test_loss_rejects_malformed_sampling_configuration_at_construction(self):
        for options in (
            {"velocity_weighted": "false"},
            {"timestep_distribution": "mystery"},
            {"logit_normal_loc": float("nan")},
            {"logit_normal_loc": "0.0"},
            {"logit_normal_scale": 0.0},
            {"num_strata": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                FlowMatchingLoss(**options)

    def test_rms_norm_rejects_nonpositive_or_nonfinite_epsilon(self):
        for epsilon in (0.0, -1e-6, float("nan"), float("inf"), True):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "norm_eps"):
                RMSNorm(4, eps=epsilon)


class LatentMaskTest(unittest.TestCase):
    def test_flow_loss_passes_latent_mask_to_model(self):
        model = RecordingVelocityModel()
        latent_mask = torch.tensor([[True, True, False]])
        loss = FlowMatchingLoss(timestep_distribution="uniform")(
            model=model,
            latents=torch.ones(1, 2, 3),
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            latent_mask=latent_mask,
        )

        self.assertIsNotNone(model.kwargs)
        assert model.kwargs is not None
        torch.testing.assert_close(model.kwargs["latent_mask"], latent_mask)
        self.assertEqual(loss.ndim, 0)

    def test_flow_loss_rejects_wrong_latent_mask_shape(self):
        with self.assertRaisesRegex(ValueError, "latent_mask"):
            FlowMatchingLoss(timestep_distribution="uniform")(
                model=RecordingVelocityModel(),
                latents=torch.ones(1, 2, 3),
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                latent_mask=torch.ones(1, 2, dtype=torch.bool),
            )


class SigmaMinContractTest(unittest.TestCase):
    def test_rejects_non_default_sigma_min_instead_of_silently_ignoring_it(self):
        with self.assertRaisesRegex(ValueError, "legacy no-op"):
            FlowMatchingLoss(sigma_min=0.01)

    def test_rejects_non_finite_sigma_min(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            FlowMatchingLoss(sigma_min=float("nan"))


class DurationObjectiveTest(unittest.TestCase):
    def test_uses_valid_latent_frames_as_log_duration_target(self):
        latent_mask = torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, True],
            ]
        )
        exact_prediction = torch.log1p(torch.tensor([2.0, 4.0]))
        model = RecordingDurationModel(exact_prediction)
        objective = FlowMatchingLoss(
            timestep_distribution="uniform",
            duration_loss_weight=0.5,
        )

        torch.manual_seed(11)
        exact_loss = objective(
            model=model,
            latents=torch.ones(2, 2, 4),
            conditioning_ids=torch.ones(2, 2, dtype=torch.long),
            latent_mask=latent_mask,
        )
        torch.manual_seed(11)
        flow_only_loss = FlowMatchingLoss(timestep_distribution="uniform")(
            model=RecordingVelocityModel(),
            latents=torch.ones(2, 2, 4),
            conditioning_ids=torch.ones(2, 2, dtype=torch.long),
            latent_mask=latent_mask,
        )

        torch.testing.assert_close(exact_loss, flow_only_loss)
        self.assertTrue(model.kwargs["return_duration_prediction"])

    def test_requires_versioned_model_output_when_duration_loss_is_enabled(self):
        with self.assertRaisesRegex(RuntimeError, "EchoDiT v2"):
            FlowMatchingLoss(
                timestep_distribution="uniform",
                duration_loss_weight=0.1,
            )(
                model=RecordingVelocityModel(),
                latents=torch.ones(1, 2, 3),
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            )

    def test_rejects_invalid_duration_objective_settings(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            FlowMatchingLoss(duration_loss_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            FlowMatchingLoss(duration_huber_delta=0.0)


class LatentAttentionTest(unittest.TestCase):
    def test_joint_attention_ignores_masked_latent_keys(self):
        torch.manual_seed(7)
        attention = JointAttention(
            model_size=8,
            num_heads=2,
            text_model_size=8,
            speaker_model_size=8,
            speaker_patch_size=2,
        ).eval()
        x = torch.randn(1, 4, 8)
        changed_padding = x.clone()
        changed_padding[:, 2:] += 10.0
        empty_cache = (
            torch.empty(1, 0, 2, 4),
            torch.empty(1, 0, 2, 4),
        )
        frequencies = precompute_freqs_cis(4, 4)
        latent_mask = torch.tensor([[True, True, False, False]])

        masked = attention(
            x,
            None,
            None,
            frequencies,
            empty_cache,
            empty_cache,
            self_mask=latent_mask,
        )
        masked_changed = attention(
            changed_padding,
            None,
            None,
            frequencies,
            empty_cache,
            empty_cache,
            self_mask=latent_mask,
        )
        unmasked_changed = attention(
            changed_padding,
            None,
            None,
            frequencies,
            empty_cache,
            empty_cache,
        )

        torch.testing.assert_close(masked[:, :2], masked_changed[:, :2])
        self.assertFalse(torch.allclose(masked[:, :2], unmasked_changed[:, :2]))

    def test_joint_attention_rejects_wrong_latent_mask_shape(self):
        attention = JointAttention(
            model_size=8,
            num_heads=2,
            text_model_size=8,
            speaker_model_size=8,
            speaker_patch_size=2,
        )
        empty_cache = (
            torch.empty(1, 0, 2, 4),
            torch.empty(1, 0, 2, 4),
        )
        with self.assertRaisesRegex(ValueError, "Latent mask"):
            attention(
                torch.zeros(1, 4, 8),
                None,
                None,
                precompute_freqs_cis(4, 4),
                empty_cache,
                empty_cache,
                self_mask=torch.ones(1, 3, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
