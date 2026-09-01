import copy
import math
import unittest

import torch
import torch.nn as nn

from nar_vae.checkpoint import (
    GenerativeObjectiveCheckpointError,
    inspect_diffusion_schedule_shift,
    inspect_generative_objective,
)
from nar_vae.losses import FlowMatchingLoss
from nar_vae.models.flow_matching import create_flow_matching_echodit
from nar_vae.models.text_conditioning import FROZEN_FEATURE_TEXT_CONDITIONING
from nar_vae.objectives import (
    DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY,
    GENERATIVE_OBJECTIVE_METADATA_KEY,
    RECTIFIED_FLOW_OBJECTIVE,
    VP_DIFFUSION_OBJECTIVE,
    diffusion_probability_flow_scale,
    shifted_cosine_vp_coefficients,
)
from nar_vae.solvers.ode_solver import ODESolver


def _tiny_vp_echodit() -> nn.Module:
    """Build a real, CPU-sized frozen-text and speaker-conditioned EchoDiT."""
    return create_flow_matching_echodit(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=32,
        text_model_size=8,
        text_num_layers=0,
        text_num_heads=2,
        text_intermediate_size=16,
        text_conditioning_mode=FROZEN_FEATURE_TEXT_CONDITIONING,
        conditioning_feature_size=6,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        timestep_embed_size=8,
        adaln_rank=4,
        cfg_dropout=0.0,
        use_speaker_conditioning=True,
        generative_objective=VP_DIFFUSION_OBJECTIVE,
    ).eval()


def _cast_real_parameters(module: nn.Module, dtype: torch.dtype) -> nn.Module:
    """Cast learned state without corrupting EchoDiT's complex RoPE caches."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
    return module


def _sample_tiny_echodit(model: nn.Module) -> torch.Tensor:
    return ODESolver.sample(
        model=model,
        conditioning_ids=torch.tensor([[1, 2, 3]]),
        conditioning_mask=torch.ones(1, 3, dtype=torch.bool),
        conditioning_features=torch.randn(1, 3, 6),
        speaker_latent=torch.randn(1, 4, 4),
        num_steps=2,
        latent_shape=(1, 4, 4),
        solver="ddim",
        device=torch.device("cpu"),
    )


class _ConstantPrediction(nn.Module):
    def __init__(self, objective: str, value: float = 1.0):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.generative_objective = objective
        self.diffusion_schedule_shift = 1.0
        self.value = value

    def forward(self, latents, conditioning_ids, timesteps, attention_mask):
        del conditioning_ids, timesteps, attention_mask
        return torch.full_like(latents, self.value)


class _CleanOracle(nn.Module):
    def __init__(self, clean: torch.Tensor, shift: float):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("clean", clean)
        self.generative_objective = VP_DIFFUSION_OBJECTIVE
        self.diffusion_schedule_shift = shift

    def forward(self, latents, conditioning_ids, timesteps, attention_mask):
        del conditioning_ids, attention_mask
        alpha, sigma = shifted_cosine_vp_coefficients(
            timesteps,
            self.diffusion_schedule_shift,
        )
        alpha = alpha.to(latents).view(-1, 1, 1)
        sigma = sigma.to(latents).view(-1, 1, 1)
        return (self.clean.to(latents) - alpha * latents) / sigma


class _VPTrainingOracle(nn.Module):
    def __init__(self, clean: torch.Tensor, shift: float):
        super().__init__()
        self.register_buffer("clean", clean)
        self.shift = shift

    def forward(self, *, latents, timesteps, **kwargs):
        del kwargs
        alpha, sigma = shifted_cosine_vp_coefficients(timesteps, self.shift)
        alpha = alpha.to(latents).view(-1, 1, 1)
        sigma = sigma.to(latents).view(-1, 1, 1)
        return (self.clean.to(latents) - alpha * latents) / sigma


class _IntegerStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.int64), requires_grad=False)

    def forward(self, latents, conditioning_ids, timesteps, attention_mask):
        del conditioning_ids, timesteps, attention_mask
        return torch.zeros_like(latents)


class DiffusionObjectiveTests(unittest.TestCase):
    def test_shifted_cosine_is_variance_preserving_and_has_exact_endpoints(self):
        timesteps = torch.tensor([0.0, 0.25, 0.75, 1.0], dtype=torch.float64)
        alpha, sigma = shifted_cosine_vp_coefficients(timesteps, 0.2)
        torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(alpha))
        torch.testing.assert_close(alpha[[0, -1]], torch.tensor([0.0, 1.0], dtype=alpha.dtype))
        torch.testing.assert_close(sigma[[0, -1]], torch.tensor([1.0, 0.0], dtype=sigma.dtype))

    def test_probability_flow_chain_factor_matches_unshifted_pi_over_two(self):
        timesteps = torch.tensor([0.0, 0.2, 0.8], dtype=torch.float64)
        scale = diffusion_probability_flow_scale(timesteps, 1.0)
        assert isinstance(scale, torch.Tensor)
        torch.testing.assert_close(scale, torch.full_like(scale, math.pi / 2.0))

    def test_shifted_probability_flow_factor_matches_finite_difference(self):
        timesteps = torch.tensor([0.08, 0.29, 0.63, 0.91], dtype=torch.float64)
        shift = 0.35
        delta = 1e-6
        alpha_before, sigma_before = shifted_cosine_vp_coefficients(
            timesteps - delta,
            shift,
        )
        alpha_after, sigma_after = shifted_cosine_vp_coefficients(
            timesteps + delta,
            shift,
        )
        angle_before = torch.atan2(alpha_before, sigma_before)
        angle_after = torch.atan2(alpha_after, sigma_after)
        finite_difference = (angle_after - angle_before) / (2 * delta)
        analytical = diffusion_probability_flow_scale(timesteps, shift)
        assert isinstance(analytical, torch.Tensor)
        torch.testing.assert_close(analytical, finite_difference, atol=2e-8, rtol=2e-8)

    def test_ddim_v_oracle_reaches_clean_sample_exactly(self):
        clean = torch.tensor([[[0.25, -0.75, 1.5]]])
        model = _CleanOracle(clean, shift=0.2).eval()
        sampled = ODESolver.sample(
            model=model,
            conditioning_ids=torch.tensor([[1, 2]]),
            num_steps=8,
            latent_shape=tuple(clean.shape),
            solver="ddim",
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(sampled, clean, atol=2e-5, rtol=2e-5)

    def test_v_parameterization_reconstructs_clean_and_noise(self):
        clean = torch.randn(4, 3, 5, dtype=torch.float64)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0.07, 0.31, 0.68, 0.94], dtype=torch.float64)
        alpha, sigma = shifted_cosine_vp_coefficients(timesteps, 0.4)
        alpha = alpha[:, None, None]
        sigma = sigma[:, None, None]
        noisy = alpha * clean + sigma * noise
        velocity = sigma * clean - alpha * noise
        torch.testing.assert_close(alpha * noisy + sigma * velocity, clean)
        torch.testing.assert_close(sigma * noisy - alpha * velocity, noise)

    def test_diffusion_loss_matches_the_v_prediction_oracle(self):
        clean = torch.randn(3, 2, 7)
        objective = FlowMatchingLoss(
            generative_objective=VP_DIFFUSION_OBJECTIVE,
            diffusion_schedule_shift=0.4,
            timestep_distribution="uniform",
        )
        loss = objective(
            _VPTrainingOracle(clean, shift=0.4),
            clean,
            torch.tensor([[1, 2], [1, 2], [1, 2]]),
        )
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=2e-12, rtol=0)

    def test_euler_applies_pi_over_two_to_v_prediction(self):
        model = _ConstantPrediction(VP_DIFFUSION_OBJECTIVE).eval()
        torch.manual_seed(123)
        initial = torch.randn((1, 1, 3))
        torch.manual_seed(123)
        sampled = ODESolver.sample(
            model=model,
            conditioning_ids=torch.tensor([[1]]),
            num_steps=1,
            latent_shape=(1, 1, 3),
            solver="euler",
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(sampled, initial + math.pi / 2.0)

    def test_real_echodit_ddim_keeps_fp32_shape_and_finite_state(self):
        model = _tiny_vp_echodit()

        torch.manual_seed(17)
        sampled = _sample_tiny_echodit(model)

        self.assertEqual(sampled.dtype, torch.float32)
        self.assertEqual(tuple(sampled.shape), (1, 4, 4))
        self.assertTrue(bool(torch.isfinite(sampled).all()))

    def test_real_echodit_ddim_preserves_cpu_bfloat16_state(self):
        try:
            probe = torch.ones(1, 2, dtype=torch.bfloat16)
            torch.nn.functional.linear(probe, torch.ones(2, 2, dtype=torch.bfloat16))
            attention = torch.ones(1, 1, 2, 4, dtype=torch.bfloat16)
            torch.nn.functional.scaled_dot_product_attention(attention, attention, attention)
        except (RuntimeError, NotImplementedError) as error:
            self.skipTest(f"PyTorch CPU bfloat16 kernels are unavailable: {error}")

        fp32_model = _tiny_vp_echodit()
        bf16_model = _cast_real_parameters(copy.deepcopy(fp32_model), torch.bfloat16)

        torch.manual_seed(29)
        fp32_sample = _sample_tiny_echodit(fp32_model)
        torch.manual_seed(29)
        bf16_sample = _sample_tiny_echodit(bf16_model)

        self.assertEqual(bf16_sample.dtype, torch.bfloat16)
        self.assertEqual(tuple(bf16_sample.shape), tuple(fp32_sample.shape))
        self.assertTrue(bool(torch.isfinite(bf16_sample).all()))
        torch.testing.assert_close(
            bf16_sample.float(),
            fp32_sample,
            atol=2e-2,
            rtol=2e-2,
        )

    def test_solver_rejects_a_nonfloating_model_state(self):
        with self.assertRaisesRegex(ValueError, "state parameter must use"):
            ODESolver.sample(
                model=_IntegerStateModel(),
                conditioning_ids=torch.tensor([[1]]),
                num_steps=1,
                latent_shape=(1, 1, 1),
                solver="euler",
                device=torch.device("cpu"),
            )

    def test_checkpoint_objective_and_schedule_are_an_atomic_contract(self):
        self.assertEqual(inspect_generative_objective({}), RECTIFIED_FLOW_OBJECTIVE)
        self.assertEqual(inspect_diffusion_schedule_shift({}), 1.0)
        state = {
            GENERATIVE_OBJECTIVE_METADATA_KEY: torch.tensor(1, dtype=torch.int32),
            DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY: torch.tensor(0.2, dtype=torch.float64),
        }
        self.assertEqual(inspect_generative_objective(state), VP_DIFFUSION_OBJECTIVE)
        self.assertAlmostEqual(inspect_diffusion_schedule_shift(state), 0.2)
        with self.assertRaises(GenerativeObjectiveCheckpointError):
            inspect_generative_objective(
                {GENERATIVE_OBJECTIVE_METADATA_KEY: torch.tensor(1, dtype=torch.int32)}
            )
        with self.assertRaises(GenerativeObjectiveCheckpointError):
            inspect_generative_objective(
                {
                    GENERATIVE_OBJECTIVE_METADATA_KEY: torch.tensor(1.5),
                    DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY: torch.tensor(1.0),
                }
            )
        with self.assertRaises(GenerativeObjectiveCheckpointError):
            inspect_diffusion_schedule_shift(
                {
                    GENERATIVE_OBJECTIVE_METADATA_KEY: torch.tensor(1.5),
                    DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY: torch.tensor(1.0),
                }
            )

    def test_ddim_rejects_sampling_knobs_that_change_the_vp_contract(self):
        model = _ConstantPrediction(VP_DIFFUSION_OBJECTIVE).eval()
        common = {
            "model": model,
            "conditioning_ids": torch.tensor([[1]]),
            "num_steps": 1,
            "latent_shape": (1, 1, 1),
            "solver": "ddim",
            "device": torch.device("cpu"),
        }
        with self.assertRaisesRegex(ValueError, "initial_noise_scale=1.0"):
            ODESolver.sample(**common, initial_noise_scale=0.8)
        with self.assertRaisesRegex(ValueError, "post-hoc"):
            ODESolver.sample(**common, target_latent_std=1.0)

    def test_requested_objective_cannot_override_model_identity(self):
        model = _ConstantPrediction(VP_DIFFUSION_OBJECTIVE).eval()
        with self.assertRaisesRegex(ValueError, "does not match"):
            ODESolver.sample(
                model=model,
                conditioning_ids=torch.tensor([[1]]),
                num_steps=1,
                latent_shape=(1, 1, 1),
                solver="euler",
                generative_objective=RECTIFIED_FLOW_OBJECTIVE,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
