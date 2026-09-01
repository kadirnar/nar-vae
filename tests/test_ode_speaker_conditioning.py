"""Speaker-reference propagation tests for every ODE solver."""

import unittest
from unittest.mock import patch

import torch

from nar_vae.solvers.ode_solver import ODESolver


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.references = []
        self.masks = []
        self.cfg_calls = 0

    def forward(
        self,
        latents,
        conditioning_ids,
        timesteps,
        attention_mask=None,
        speaker_latent=None,
        use_cfg_dropout=False,
        speaker_mask=None,
    ):
        del conditioning_ids, timesteps, attention_mask, use_cfg_dropout
        self.references.append(speaker_latent)
        self.masks.append(speaker_mask)
        return torch.zeros_like(latents)

    def forward_with_cfg(
        self,
        latents,
        conditioning_ids,
        timesteps,
        *,
        speaker_latent=None,
        speaker_mask=None,
        **kwargs,
    ):
        del conditioning_ids, timesteps, kwargs
        self.cfg_calls += 1
        self.references.append(speaker_latent)
        self.masks.append(speaker_mask)
        return torch.zeros_like(latents)


class PreparedSlice:
    def __init__(self, values):
        self.values = values

    def slice_batch(self, start, stop):
        return PreparedSlice(self.values[start:stop])


class PreparedRecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.prepare_cfg_calls = 0
        self.prepare_conditioning_calls = 0
        self.prepared_forward_calls = 0
        self.cfg_fallback_calls = 0
        self.ordinary_forward_calls = 0
        self.prepare_cfg_kwargs = None
        self.prepare_conditioning_kwargs = None

    def prepare_fused_cfg_conditioning(self, *args, cfg_mode, **kwargs):
        del args
        self.prepare_cfg_kwargs = kwargs
        self.prepare_cfg_calls += 1
        if cfg_mode == "joint":
            variants = (PreparedSlice(torch.tensor([3.0, 1.0])),)
            branch_count = 2
        elif cfg_mode == "independent":
            variants = (PreparedSlice(torch.tensor([3.0, 1.0, 2.0])),)
            branch_count = 3
        elif cfg_mode == "alternating":
            variants = (
                PreparedSlice(torch.tensor([3.0, 1.0])),
                PreparedSlice(torch.tensor([3.0, 2.0])),
            )
            branch_count = 2
        else:
            raise ValueError(cfg_mode)
        return type(
            "PreparedCFG",
            (),
            {
                "mode": cfg_mode,
                "branch_count": branch_count,
                "variants": variants,
                "conditional": variants[0].slice_batch(0, 1),
            },
        )()

    def prepare_inference_conditioning(self, *args, **kwargs):
        del args
        self.prepare_conditioning_kwargs = kwargs
        self.prepare_conditioning_calls += 1
        return PreparedSlice(torch.tensor([4.0]))

    def forward_prepared(self, latents, timesteps, conditioning):
        del timesteps
        self.prepared_forward_calls += 1
        return conditioning.values[:, None, None].expand_as(latents)

    def forward_with_cfg(self, latents, *args, **kwargs):
        del args, kwargs
        self.cfg_fallback_calls += 1
        return torch.full_like(latents, 7.0)

    def forward(self, latents, conditioning_ids, timesteps, attention_mask=None):
        del conditioning_ids, timesteps, attention_mask
        self.ordinary_forward_calls += 1
        return torch.zeros_like(latents)


class ODESpeakerConditioningTest(unittest.TestCase):
    def test_invalid_numerical_options_fail_before_noise_allocation(self):
        common = {
            "model": RecordingModel(),
            "conditioning_ids": torch.ones(1, 2, dtype=torch.long),
            "latent_shape": (1, 2, 2),
            "device": torch.device("cpu"),
        }
        invalid = (
            {"num_steps": 0},
            {"num_steps": True},
            {"num_steps": 1.5},
            {"num_steps": 1, "cfg_scale": float("nan")},
            {"num_steps": 1, "cfg_scale_text": float("inf")},
            {"num_steps": 1, "cfg_min_t": float("nan")},
            {"num_steps": 1, "initial_noise_scale": 0.0},
            {"num_steps": 1, "temporal_rescale_k": 0.0},
            {"num_steps": 1, "temporal_rescale_sigma": float("inf")},
            {"num_steps": 1, "target_latent_std": float("nan")},
        )
        with patch.object(
            torch, "randn", side_effect=AssertionError("must fail before allocation")
        ):
            for options in invalid:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    ODESolver.sample(**common, **options)

    def test_reference_reaches_every_solver_without_cfg(self):
        for solver, expected_calls in (
            ("euler", 1),
            ("midpoint", 2),
            ("heun", 2),
            ("rk4", 4),
        ):
            with self.subTest(solver=solver):
                model = RecordingModel()
                reference = torch.ones(1, 2, 4)
                ODESolver.sample(
                    model=model,
                    conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                    num_steps=1,
                    latent_shape=(1, 2, 2),
                    solver=solver,
                    cfg_scale=1.0,
                    speaker_latent=reference,
                    device=torch.device("cpu"),
                )

                self.assertEqual(len(model.references), expected_calls)
                self.assertTrue(all(item is reference for item in model.references))

    def test_reference_reaches_cfg_path(self):
        model = RecordingModel()
        reference = torch.ones(1, 2, 4)
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=1,
            latent_shape=(1, 2, 2),
            solver="euler",
            cfg_scale=2.0,
            cfg_min_t=0.0,
            cfg_max_t=1.0,
            speaker_latent=reference,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.references, [reference])

    def test_invalid_reference_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "speaker_latent must have shape"):
            ODESolver.sample(
                model=RecordingModel(),
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                num_steps=1,
                latent_shape=(1, 2, 2),
                speaker_latent=torch.ones(2, 4),
                device=torch.device("cpu"),
            )

    def test_empty_and_nonfinite_references_are_rejected(self):
        common = {
            "model": RecordingModel(),
            "conditioning_ids": torch.ones(1, 2, dtype=torch.long),
            "num_steps": 1,
            "latent_shape": (1, 2, 2),
            "device": torch.device("cpu"),
        }
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            ODESolver.sample(
                **common,
                speaker_latent=torch.empty(1, 2, 0),
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ODESolver.sample(
                **common,
                speaker_latent=torch.full((1, 2, 1), torch.nan),
            )

    def test_speaker_mask_reaches_model(self):
        model = RecordingModel()
        reference = torch.ones(1, 2, 4)
        mask = torch.tensor([[True, False]])
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=1,
            latent_shape=(1, 2, 2),
            cfg_scale=1.0,
            speaker_latent=reference,
            speaker_mask=mask,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.masks, [mask])

    def test_rk4_applies_cfg_per_substage_time(self):
        model = RecordingModel()
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=1,
            latent_shape=(1, 2, 2),
            solver="rk4",
            cfg_scale=2.0,
            cfg_min_t=0.4,
            cfg_max_t=0.6,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.cfg_calls, 2)

    def test_midpoint_applies_cfg_at_the_half_step(self):
        model = RecordingModel()
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=1,
            latent_shape=(1, 2, 2),
            solver="midpoint",
            cfg_scale=2.0,
            cfg_min_t=0.4,
            cfg_max_t=0.6,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.cfg_calls, 1)

    def test_euler_does_not_extract_tensor_scalars_on_the_host(self):
        with patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("Euler must not synchronize through Tensor.item()"),
        ):
            result = ODESolver.sample(
                model=RecordingModel(),
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                num_steps=2,
                latent_shape=(1, 2, 2),
                solver="euler",
                cfg_scale=1.0,
                target_latent_std=1.0,
                device=torch.device("cpu"),
            )

        self.assertEqual(tuple(result.shape), (1, 2, 2))

    def test_nonfused_cfg_prepares_encoders_once_and_keeps_branches_sequential(self):
        model = PreparedRecordingModel().eval()
        with patch.object(torch, "randn", return_value=torch.zeros(1, 2, 2)):
            result = ODESolver.sample(
                model=model,
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                num_steps=3,
                latent_shape=(1, 2, 2),
                solver="euler",
                cfg_scale=2.0,
                cfg_mode="independent",
                cfg_scale_text=2.5,
                cfg_scale_speaker=2.0,
                fuse_cfg_branches=False,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(result, torch.full_like(result, 10.0))
        self.assertEqual(model.prepare_cfg_calls, 1)
        self.assertEqual(model.prepare_conditioning_calls, 0)
        self.assertEqual(model.prepared_forward_calls, 9)
        self.assertEqual(model.cfg_fallback_calls, 0)

    def test_explicit_independent_scale_activates_cfg_when_joint_scale_is_neutral(self):
        model = PreparedRecordingModel().eval()
        with patch.object(torch, "randn", return_value=torch.zeros(1, 2, 2)):
            result = ODESolver.sample(
                model=model,
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                num_steps=1,
                latent_shape=(1, 2, 2),
                solver="euler",
                cfg_scale=1.0,
                cfg_mode="independent",
                cfg_scale_text=2.5,
                cfg_scale_speaker=0.0,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(result, torch.full_like(result, 8.0))
        self.assertEqual(model.prepare_cfg_calls, 1)
        self.assertEqual(model.prepare_conditioning_calls, 0)
        self.assertEqual(model.prepared_forward_calls, 3)

    def test_explicit_alternating_scale_activates_cfg_when_joint_scale_is_neutral(self):
        model = RecordingModel()
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=2,
            latent_shape=(1, 2, 2),
            solver="euler",
            cfg_scale=1.0,
            cfg_mode="alternating",
            cfg_scale_text=2.0,
            cfg_scale_speaker=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.cfg_calls, 2)

    def test_alternating_cfg_keeps_one_branch_for_every_substage_in_a_solver_step(self):
        class StepRecordingModel(RecordingModel):
            def __init__(self):
                super().__init__()
                self.step_indices = []

            def forward_with_cfg(
                self,
                latents,
                conditioning_ids,
                timesteps,
                *,
                step_idx,
                **kwargs,
            ):
                del conditioning_ids, timesteps, kwargs
                self.step_indices.append(step_idx)
                return torch.zeros_like(latents)

        expected_indices = {
            "euler": [0, 1],
            "midpoint": [0, 0, 1, 1],
            "heun": [0, 0, 1, 1],
            "rk4": [0, 0, 0, 0, 1, 1, 1, 1],
        }
        for solver, expected in expected_indices.items():
            with self.subTest(solver=solver):
                model = StepRecordingModel()
                ODESolver.sample(
                    model=model,
                    conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                    num_steps=2,
                    latent_shape=(1, 2, 2),
                    solver=solver,
                    cfg_scale=2.0,
                    cfg_mode="alternating",
                    device=torch.device("cpu"),
                )

                self.assertEqual(model.step_indices, expected)

    def test_prepared_conditional_state_is_reused_without_cfg(self):
        model = PreparedRecordingModel().eval()
        result = ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=2,
            latent_shape=(1, 2, 2),
            solver="rk4",
            cfg_scale=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(result.shape), (1, 2, 2))
        self.assertEqual(model.prepare_conditioning_calls, 1)
        self.assertEqual(model.prepared_forward_calls, 8)

    def test_integration_boundary_follows_invariant_conditioning_and_precedes_forward(self):
        model = PreparedRecordingModel().eval()
        events = []
        original_prepare = model.prepare_inference_conditioning
        original_forward = model.forward_prepared

        def record_prepare(*args, **kwargs):
            events.append("prepare")
            return original_prepare(*args, **kwargs)

        def record_forward(*args, **kwargs):
            events.append("forward")
            return original_forward(*args, **kwargs)

        model.prepare_inference_conditioning = record_prepare
        model.forward_prepared = record_forward
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=1,
            latent_shape=(1, 2, 2),
            solver="euler",
            cfg_scale=1.0,
            device=torch.device("cpu"),
            integration_start_callback=lambda: events.append("boundary"),
        )

        self.assertEqual(events, ["prepare", "boundary", "forward"])

    def test_exact_token_durations_reach_prepared_cfg_and_conditional_paths(self):
        durations = torch.tensor([[1, 1]])
        for cfg_scale, expected_attribute in (
            (1.0, "prepare_conditioning_kwargs"),
            (2.0, "prepare_cfg_kwargs"),
        ):
            with self.subTest(cfg_scale=cfg_scale):
                model = PreparedRecordingModel().eval()
                ODESolver.sample(
                    model=model,
                    conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                    token_durations=durations,
                    num_steps=1,
                    latent_shape=(1, 2, 2),
                    cfg_scale=cfg_scale,
                    device=torch.device("cpu"),
                )

                forwarded = getattr(model, expected_attribute)["token_durations"]
                torch.testing.assert_close(forwarded, durations)

    def test_invalid_token_duration_contract_is_rejected_before_sampling(self):
        common = {
            "model": PreparedRecordingModel().eval(),
            "conditioning_ids": torch.ones(1, 2, dtype=torch.long),
            "num_steps": 1,
            "latent_shape": (1, 2, 3),
            "device": torch.device("cpu"),
        }
        with self.assertRaisesRegex(ValueError, "sum to"):
            ODESolver.sample(**common, token_durations=torch.tensor([[1, 1]]))
        with self.assertRaisesRegex(ValueError, "Padded"):
            ODESolver.sample(
                **common,
                token_durations=torch.tensor([[1, 2]]),
                conditioning_mask=torch.tensor([[True, False]]),
            )

    def test_nonfused_alternating_cfg_uses_the_matching_prepared_variant(self):
        model = PreparedRecordingModel().eval()
        with patch.object(torch, "randn", return_value=torch.zeros(1, 2, 2)):
            result = ODESolver.sample(
                model=model,
                conditioning_ids=torch.ones(1, 2, dtype=torch.long),
                num_steps=2,
                latent_shape=(1, 2, 2),
                solver="euler",
                cfg_scale=2.0,
                cfg_mode="alternating",
                cfg_scale_text=2.5,
                cfg_scale_speaker=2.0,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(result, torch.full_like(result, 5.0))
        self.assertEqual(model.prepare_cfg_calls, 1)
        self.assertEqual(model.prepared_forward_calls, 4)

    def test_conditional_only_preparation_does_not_bypass_cfg(self):
        model = PreparedRecordingModel().eval()
        model.prepare_fused_cfg_conditioning = None
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=2,
            latent_shape=(1, 2, 2),
            solver="euler",
            cfg_scale=2.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.prepare_conditioning_calls, 1)
        self.assertEqual(model.prepared_forward_calls, 0)
        self.assertEqual(model.cfg_fallback_calls, 2)

    def test_implicit_preparation_does_not_change_training_mode_sampling(self):
        model = PreparedRecordingModel()
        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            num_steps=1,
            latent_shape=(1, 2, 2),
            solver="euler",
            cfg_scale=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(model.prepare_conditioning_calls, 0)
        self.assertEqual(model.prepared_forward_calls, 0)
        self.assertEqual(model.ordinary_forward_calls, 1)

    def test_target_latent_std_is_independent_for_each_batched_utterance(self):
        initial = torch.tensor([[[-1.0, 1.0]], [[-10.0, 10.0]]])
        with patch.object(torch, "randn", return_value=initial.clone()):
            result = ODESolver.sample(
                model=RecordingModel(),
                conditioning_ids=torch.ones(2, 2, dtype=torch.long),
                num_steps=1,
                latent_shape=(2, 1, 2),
                solver="euler",
                cfg_scale=1.0,
                target_latent_std=2.0,
                device=torch.device("cpu"),
            )

        torch.testing.assert_close(result[0].std(), torch.tensor(2.0))
        torch.testing.assert_close(result[1].std(), torch.tensor(2.0))


if __name__ == "__main__":
    unittest.main()
