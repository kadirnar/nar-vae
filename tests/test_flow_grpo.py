"""CPU-safe tests for flow-native GRPO post-training primitives."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from nar_vae.post_training import (
    FlowGRPOConfig,
    FlowGRPOTrainer,
    clipped_grpo_loss,
    combine_reward_components,
    diagonal_gaussian_log_prob,
    flow_sde_transition,
    group_relative_advantages,
    recompute_transition_statistics,
    same_variance_gaussian_kl,
    sample_flow_grpo_trajectory,
)


class ConstantVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, time, conditioning):
        del time, conditioning
        return torch.ones_like(state) * self.scale


class ModeSensitiveVelocity(ConstantVelocity):
    def __init__(self):
        super().__init__()
        self.observed_modes = []

    def forward(self, state, time, conditioning):
        self.observed_modes.append(self.training)
        velocity = super().forward(state, time, conditioning)
        return velocity + float(self.training)


def velocity_adapter(model, state, time, conditioning):
    return model(state, time, conditioning)


class FlowGRPOTest(unittest.TestCase):
    def test_config_rejects_singular_or_empty_sde_windows(self):
        with self.assertRaisesRegex(ValueError, "t=0"):
            FlowGRPOConfig(sde_window_start=0)
        with self.assertRaisesRegex(ValueError, "fit"):
            FlowGRPOConfig(num_steps=4, sde_window_start=2, sde_window_size=3)
        with self.assertRaisesRegex(ValueError, "two samples"):
            FlowGRPOConfig(group_size=1)
        with self.assertRaisesRegex(ValueError, "event_reduction"):
            FlowGRPOConfig(event_reduction="none")
        with self.assertRaisesRegex(ValueError, "noise_level"):
            FlowGRPOConfig(noise_level=float("nan"))
        with self.assertRaisesRegex(ValueError, "integers"):
            FlowGRPOConfig(num_steps=4.5)

    def test_group_advantages_are_normalized_and_constant_groups_are_zero(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]])
        advantages = group_relative_advantages(rewards)

        torch.testing.assert_close(advantages[0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            advantages[0].std(unbiased=False), torch.tensor(1.0), atol=1e-6, rtol=0
        )
        torch.testing.assert_close(advantages[1], torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "finite"):
            group_relative_advantages(torch.tensor([[0.0, float("nan")]]))

    def test_reward_components_require_exact_weights_and_finite_values(self):
        components = {
            "asr": torch.tensor([[0.5, 0.8]]),
            "speaker": torch.tensor([[0.2, 0.4]]),
        }
        combined = combine_reward_components(components, {"asr": 1.0, "speaker": 0.5})
        self.assertEqual(combined.shape, (1, 2))
        with self.assertRaisesRegex(ValueError, "match exactly"):
            combine_reward_components(components, {"asr": 1.0})

    def test_weighted_reward_matches_batch_standard_deviation_equation(self):
        components = {
            "asr": torch.tensor([[1.0, 3.0]]),
            "speaker": torch.tensor([[2.0, 6.0]]),
        }
        combined = combine_reward_components(components, {"asr": 2.0, "speaker": 0.5})

        torch.testing.assert_close(combined, torch.tensor([[2.5, 7.5]]))

    def test_gaussian_log_prob_has_explicit_event_scaling_and_gradients(self):
        value = torch.tensor([[[0.0, 1.0, -1.0, 2.0]]], requires_grad=True)
        mean = torch.zeros_like(value)
        standard_deviation = torch.tensor([[2.0]])

        mean_log_prob = diagonal_gaussian_log_prob(value, mean, standard_deviation)
        sum_log_prob = diagonal_gaussian_log_prob(
            value,
            mean,
            standard_deviation,
            event_reduction="sum",
        )

        torch.testing.assert_close(sum_log_prob, mean_log_prob * value.shape[-1])
        mean_log_prob.sum().backward()
        expected_gradient = -value.detach() / (standard_deviation.item() ** 2 * value.shape[-1])
        torch.testing.assert_close(value.grad, expected_gradient)

    def test_gaussian_statistics_promote_low_precision_for_stability(self):
        value = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
        result = diagonal_gaussian_log_prob(value, value, torch.tensor(0.1))

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())

    def test_gaussian_log_prob_excludes_padded_event_elements(self):
        value = torch.tensor([[[0.0, 1.0, 1000.0, -1000.0]]])
        mean = torch.zeros_like(value)
        mask = torch.tensor([[[True, True, False, False]]])

        masked = diagonal_gaussian_log_prob(
            value,
            mean,
            torch.tensor(1.0),
            event_mask=mask,
        )
        expected = diagonal_gaussian_log_prob(
            value[..., :2],
            mean[..., :2],
            torch.tensor(1.0),
        )

        torch.testing.assert_close(masked, expected)
        with self.assertRaisesRegex(ValueError, "at least one"):
            diagonal_gaussian_log_prob(
                value,
                mean,
                torch.tensor(1.0),
                event_mask=torch.zeros_like(value, dtype=torch.bool),
            )

    def test_sde_transition_is_finite_inside_open_time_interval(self):
        state = torch.zeros(2, 4, 3, 2)
        velocity = torch.ones_like(state)
        mean, standard_deviation = flow_sde_transition(
            state,
            velocity,
            torch.tensor(0.25),
            torch.tensor(0.1),
            noise_level=0.5,
        )
        self.assertEqual(mean.shape, state.shape)
        self.assertTrue(torch.isfinite(mean).all())
        self.assertGreater(float(standard_deviation), 0)

    def test_sde_transition_promotes_low_precision_work_state(self):
        state = torch.full((1, 2, 4), 1000.0, dtype=torch.bfloat16)
        velocity = torch.full_like(state, 0.25)

        mean, standard_deviation = flow_sde_transition(
            state,
            velocity,
            torch.tensor(0.25, dtype=torch.bfloat16),
            torch.tensor(0.1, dtype=torch.bfloat16),
            noise_level=0.5,
        )

        self.assertEqual(mean.dtype, torch.float32)
        self.assertEqual(standard_deviation.dtype, torch.float32)
        self.assertTrue(torch.isfinite(mean).all())

    def test_sde_transition_matches_equation_seven_and_broadcasts_group_times(self):
        state = torch.tensor([[[1.0, -1.0], [0.5, 2.0]]])
        velocity = torch.tensor([[[0.25, 0.5], [-0.5, 1.0]]])
        time = torch.tensor([[0.25, 0.5]])
        step_size = torch.tensor(0.1)
        noise_level = 0.5

        mean, standard_deviation = flow_sde_transition(
            state,
            velocity,
            time,
            step_size,
            noise_level=noise_level,
        )
        expanded_time = time.unsqueeze(-1)
        sigma = noise_level * torch.sqrt((1 - expanded_time) / expanded_time)
        expected_mean = (
            state
            + (
                velocity
                + sigma.square() / (2 * (1 - expanded_time)) * (-state + expanded_time * velocity)
            )
            * step_size
        )

        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(standard_deviation, sigma * torch.sqrt(step_size))

    def test_rollout_and_recomputation_have_consistent_log_probabilities(self):
        config = FlowGRPOConfig(num_steps=6, group_size=3, sde_window_start=1, sde_window_size=2)
        policy = ConstantVelocity()
        initial = torch.randn(2, 3, 4, 2)
        generator = torch.Generator().manual_seed(123)

        trajectory = sample_flow_grpo_trajectory(
            policy,
            velocity_adapter,
            initial,
            None,
            config,
            generator=generator,
        )
        log_probs, _ = recompute_transition_statistics(
            policy,
            velocity_adapter,
            trajectory,
            None,
            config,
        )

        self.assertEqual(trajectory.old_log_probs.shape, (2, 3, 2))
        torch.testing.assert_close(log_probs, trajectory.old_log_probs)
        expected_times = torch.tensor([1 / 6, 2 / 6])
        torch.testing.assert_close(torch.stack(trajectory.times), expected_times)

    def test_rollout_is_fully_detached_and_does_not_accumulate_policy_gradients(self):
        config = FlowGRPOConfig(
            num_steps=5,
            group_size=2,
            sde_window_start=1,
            sde_window_size=2,
        )
        policy = ConstantVelocity()
        initial = torch.randn(1, 2, 3, requires_grad=True)

        trajectory = sample_flow_grpo_trajectory(
            policy,
            velocity_adapter,
            initial,
            None,
            config,
            generator=torch.Generator().manual_seed(9),
        )

        self.assertFalse(trajectory.final_state.requires_grad)
        self.assertFalse(trajectory.old_log_probs.requires_grad)
        self.assertTrue(all(not state.requires_grad for state in trajectory.states))
        self.assertIsNone(policy.scale.grad)

    def test_low_precision_rollout_uses_float32_trajectory_state(self):
        config = FlowGRPOConfig(
            num_steps=4,
            group_size=2,
            sde_window_start=1,
            sde_window_size=2,
        )
        trajectory = sample_flow_grpo_trajectory(
            ConstantVelocity(),
            velocity_adapter,
            torch.randn(1, 2, 3, dtype=torch.bfloat16),
            None,
            config,
            generator=torch.Generator().manual_seed(31),
        )

        self.assertEqual(trajectory.final_state.dtype, torch.float32)
        self.assertEqual(trajectory.old_log_probs.dtype, torch.float32)
        self.assertTrue(all(state.dtype == torch.float32 for state in trajectory.states))

    def test_recomputation_rejects_event_reduction_mismatch(self):
        rollout_config = FlowGRPOConfig(
            num_steps=4,
            group_size=2,
            sde_window_start=1,
            sde_window_size=1,
        )
        trajectory = sample_flow_grpo_trajectory(
            ConstantVelocity(),
            velocity_adapter,
            torch.randn(1, 2, 3),
            None,
            rollout_config,
        )
        update_config = FlowGRPOConfig(
            num_steps=4,
            group_size=2,
            sde_window_start=1,
            sde_window_size=1,
            event_reduction="sum",
        )

        with self.assertRaisesRegex(ValueError, "same event"):
            recompute_transition_statistics(
                ConstantVelocity(),
                velocity_adapter,
                trajectory,
                None,
                update_config,
            )

    def test_same_variance_kl_matches_closed_form_and_detaches_reference(self):
        policy_mean = torch.ones(1, 2, 4, requires_grad=True)
        reference_mean = torch.zeros(1, 2, 4, requires_grad=True)
        deviation = torch.tensor(2.0)

        mean_kl = same_variance_gaussian_kl(
            (policy_mean,),
            (reference_mean,),
            (deviation,),
        )
        sum_kl = same_variance_gaussian_kl(
            (policy_mean,),
            (reference_mean,),
            (deviation,),
            event_reduction="sum",
        )

        torch.testing.assert_close(mean_kl, torch.full((1, 2, 1), 0.125))
        torch.testing.assert_close(sum_kl, mean_kl * 4)
        mean_kl.sum().backward()
        self.assertTrue(torch.all(policy_mean.grad > 0))
        self.assertIsNone(reference_mean.grad)

    def test_same_variance_kl_masks_padded_gradients(self):
        policy_mean = torch.ones(1, 2, 4, requires_grad=True)
        reference_mean = torch.zeros_like(policy_mean)
        mask = torch.tensor([[[True, True, False, False]]])

        kl = same_variance_gaussian_kl(
            (policy_mean,),
            (reference_mean,),
            (torch.tensor(1.0),),
            event_mask=mask,
        )
        kl.sum().backward()

        self.assertTrue(torch.all(policy_mean.grad[..., :2] > 0))
        torch.testing.assert_close(policy_mean.grad[..., 2:], torch.zeros(1, 2, 2))

    def test_clipped_loss_backpropagates(self):
        new = torch.zeros(2, 3, 2, requires_grad=True)
        old = torch.zeros_like(new)
        advantages = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 0.25, 0.75]])
        kl = torch.full_like(new, 0.1)
        loss, _ = clipped_grpo_loss(
            new,
            old,
            advantages,
            kl,
            clip_ratio=0.2,
            kl_beta=0.01,
        )
        loss.backward()
        self.assertIsNotNone(new.grad)
        self.assertTrue(torch.isfinite(new.grad).all())

    def test_clipping_has_correct_sign_and_keeps_extreme_corrective_gradient(self):
        new = torch.tensor([[[100.0], [-100.0], [100.0], [-100.0]]], requires_grad=True)
        old = torch.zeros_like(new, requires_grad=True)
        advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0]], requires_grad=True)
        kl = torch.zeros_like(new, requires_grad=True)

        loss, _ = clipped_grpo_loss(
            new,
            old,
            advantages,
            kl,
            clip_ratio=0.2,
            kl_beta=0.1,
        )
        loss.backward()

        self.assertEqual(float(new.grad[0, 0, 0]), 0.0)
        self.assertLess(float(new.grad[0, 1, 0]), 0.0)
        self.assertGreater(float(new.grad[0, 2, 0]), 0.0)
        self.assertEqual(float(new.grad[0, 3, 0]), 0.0)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantages.grad)
        self.assertTrue(torch.all(kl.grad > 0))

    def test_trainer_performs_one_finite_update(self):
        config = FlowGRPOConfig(
            num_steps=5,
            group_size=3,
            sde_window_start=1,
            sde_window_size=2,
            supervised_replay_weight=0.0,
        )
        policy = ConstantVelocity()
        reference = copy.deepcopy(policy)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)

        def decode(latents, batch):
            del batch
            return latents

        def reward(audio, batch):
            del batch
            scores = -audio.square().mean(dim=(2, 3))
            return {"quality": scores}

        trainer = FlowGRPOTrainer(
            policy=policy,
            reference_policy=reference,
            optimizer=optimizer,
            velocity_adapter=velocity_adapter,
            decode=decode,
            reward=reward,
            reward_weights={"quality": 1.0},
            config=config,
        )
        before = policy.scale.detach().clone()
        metrics = trainer.step(
            initial_state=torch.randn(2, 3, 4, 2),
            conditioning=None,
            batch=None,
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(math_is_finite(metrics.loss))
        self.assertFalse(torch.equal(before, policy.scale.detach()))
        self.assertFalse(any(parameter.requires_grad for parameter in reference.parameters()))

    def test_trainer_uses_eval_mode_for_both_rollout_and_policy_recomputation(self):
        config = FlowGRPOConfig(
            num_steps=4,
            group_size=2,
            sde_window_start=1,
            sde_window_size=1,
            supervised_replay_weight=0.0,
        )
        policy = ModeSensitiveVelocity()
        reference = copy.deepcopy(policy)
        trainer = FlowGRPOTrainer(
            policy=policy,
            reference_policy=reference,
            optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
            velocity_adapter=velocity_adapter,
            decode=lambda latents, batch: latents,
            reward=lambda audio, batch: {"quality": -audio.square().mean(dim=2)},
            reward_weights={"quality": 1.0},
            config=config,
        )

        trainer.step(
            initial_state=torch.randn(1, 2, 3),
            conditioning=None,
            batch=None,
            generator=torch.Generator().manual_seed(11),
        )

        self.assertTrue(policy.training)
        self.assertTrue(policy.observed_modes)
        self.assertFalse(any(policy.observed_modes))
        self.assertFalse(any(reference.observed_modes))

    def test_trainer_restores_mode_and_clears_gradients_after_failure(self):
        config = FlowGRPOConfig(
            num_steps=4,
            group_size=2,
            sde_window_start=1,
            sde_window_size=1,
        )
        policy = ConstantVelocity()

        def failing_reward(audio, batch):
            del audio, batch
            raise RuntimeError("reward failed")

        trainer = FlowGRPOTrainer(
            policy=policy,
            reference_policy=copy.deepcopy(policy),
            optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
            velocity_adapter=velocity_adapter,
            decode=lambda latents, batch: latents,
            reward=failing_reward,
            reward_weights={"quality": 1.0},
            config=config,
        )

        with self.assertRaisesRegex(RuntimeError, "reward failed"):
            trainer.step(
                initial_state=torch.randn(1, 2, 3),
                conditioning=None,
                batch=None,
            )
        self.assertTrue(policy.training)
        self.assertIsNone(policy.scale.grad)

    def test_trainer_rejects_a_reference_that_shares_policy_parameters(self):
        policy = ConstantVelocity()
        with self.assertRaisesRegex(ValueError, "must not share"):
            FlowGRPOTrainer(
                policy=policy,
                reference_policy=policy,
                optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
                velocity_adapter=velocity_adapter,
                decode=lambda latents, batch: latents,
                reward=lambda audio, batch: {"quality": audio.mean(dim=2)},
                reward_weights={"quality": 1.0},
                config=FlowGRPOConfig(),
            )

    def test_trainer_rejects_unrelated_reference_state_without_lineage(self):
        policy = ConstantVelocity()
        reference = copy.deepcopy(policy)
        reference.scale.data.add_(1)

        with self.assertRaisesRegex(ValueError, "identical state"):
            FlowGRPOTrainer(
                policy=policy,
                reference_policy=reference,
                optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
                velocity_adapter=velocity_adapter,
                decode=lambda latents, batch: latents,
                reward=lambda audio, batch: {"quality": audio.mean(dim=2)},
                reward_weights={"quality": 1.0},
                config=FlowGRPOConfig(),
            )

    def test_resumed_trainer_requires_matching_reference_manifest_lineage(self):
        policy = ConstantVelocity()
        reference = copy.deepcopy(policy)
        policy.scale.data.add_(0.5)
        manifest_hash = "a" * 64

        trainer = FlowGRPOTrainer(
            policy=policy,
            reference_policy=reference,
            optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
            velocity_adapter=velocity_adapter,
            decode=lambda latents, batch: latents,
            reward=lambda audio, batch: {"quality": audio.mean(dim=2)},
            reward_weights={"quality": 1.0},
            config=FlowGRPOConfig(),
            policy_reference_manifest_sha256=manifest_hash,
            reference_manifest_sha256=manifest_hash,
        )
        self.assertIs(trainer.policy, policy)

        with self.assertRaisesRegex(ValueError, "does not match"):
            FlowGRPOTrainer(
                policy=policy,
                reference_policy=copy.deepcopy(reference),
                optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
                velocity_adapter=velocity_adapter,
                decode=lambda latents, batch: latents,
                reward=lambda audio, batch: {"quality": audio.mean(dim=2)},
                reward_weights={"quality": 1.0},
                config=FlowGRPOConfig(),
                policy_reference_manifest_sha256="a" * 64,
                reference_manifest_sha256="b" * 64,
            )

    def test_trainer_rejects_optimizer_without_policy_parameters(self):
        policy = ConstantVelocity()
        unrelated = nn.Parameter(torch.tensor(0.0))

        with self.assertRaisesRegex(ValueError, "optimizer must contain exactly"):
            FlowGRPOTrainer(
                policy=policy,
                reference_policy=copy.deepcopy(policy),
                optimizer=torch.optim.SGD([unrelated], lr=1e-3),
                velocity_adapter=velocity_adapter,
                decode=lambda latents, batch: latents,
                reward=lambda audio, batch: {"quality": audio.mean(dim=2)},
                reward_weights={"quality": 1.0},
                config=FlowGRPOConfig(),
            )

    @unittest.skipUnless(dist.is_available(), "torch.distributed is unavailable")
    def test_trainer_accepts_a_cpu_distributed_data_parallel_policy(self):
        if dist.is_initialized():
            self.skipTest("An external process group is already active")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory, "gloo-rendezvous").resolve()
            dist.init_process_group(
                "gloo",
                init_method=rendezvous.as_uri(),
                rank=0,
                world_size=1,
            )
            try:
                raw_policy = ConstantVelocity()
                reference = copy.deepcopy(raw_policy)
                policy = DistributedDataParallel(raw_policy)
                trainer = FlowGRPOTrainer(
                    policy=policy,
                    reference_policy=reference,
                    optimizer=torch.optim.SGD(policy.parameters(), lr=1e-3),
                    velocity_adapter=velocity_adapter,
                    decode=lambda latents, batch: latents,
                    reward=lambda audio, batch: {"quality": -audio.square().mean(dim=2)},
                    reward_weights={"quality": 1.0},
                    config=FlowGRPOConfig(
                        num_steps=4,
                        group_size=2,
                        sde_window_start=1,
                        sde_window_size=2,
                        supervised_replay_weight=0.0,
                    ),
                )

                metrics = trainer.step(
                    initial_state=torch.randn(1, 2, 3),
                    conditioning=None,
                    batch=None,
                    generator=torch.Generator().manual_seed(21),
                )
            finally:
                dist.destroy_process_group()

        self.assertTrue(math_is_finite(metrics.loss))


def math_is_finite(value):
    return value == value and abs(value) != float("inf")


if __name__ == "__main__":
    unittest.main()
