"""Flow-specific Group Relative Policy Optimization primitives.

This module follows the mixed ODE/SDE formulation used by FlowTTS-GRPO. It deliberately does not
bundle an ASR model, speaker verifier, or perceptual model: those evaluators are expensive,
language-specific training inputs whose versions must be selected and recorded by the server run.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn

Tensor = torch.Tensor
VelocityAdapter = Callable[[nn.Module, Tensor, Tensor, Any], Tensor]
DecodeFunction = Callable[[Tensor, Any], Tensor]
RewardFunction = Callable[[Tensor, Any], Mapping[str, Tensor]]
SupervisedLossFunction = Callable[[nn.Module, Any], Tensor]
OptimizerStepCallback = Callable[[], None]
DistributedErrorSynchronizer = Callable[[Exception | None, str], None]
EventReduction = Literal["mean", "sum"]
_MANIFEST_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FlowGRPOConfig:
    """Validated numerical contract for one mixed ODE/SDE GRPO update."""

    num_steps: int = 16
    group_size: int = 4
    sde_window_start: int = 1
    sde_window_size: int = 4
    noise_level: float = 0.7
    clip_ratio: float = 0.2
    kl_beta: float = 0.01
    supervised_replay_weight: float = 0.1
    max_grad_norm: float = 1.0
    advantage_epsilon: float = 1e-6
    log_ratio_clip: float = 20.0
    event_reduction: EventReduction = "mean"
    policy_update_epochs: int = 2

    def __post_init__(self) -> None:
        integer_fields = {
            "num_steps": self.num_steps,
            "group_size": self.group_size,
            "sde_window_start": self.sde_window_start,
            "sde_window_size": self.sde_window_size,
            "policy_update_epochs": self.policy_update_epochs,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields.values()
        ):
            raise ValueError("Flow GRPO step and group settings must be integers.")
        if self.num_steps < 3:
            raise ValueError("Flow GRPO requires at least three integration steps.")
        if self.group_size < 2:
            raise ValueError("Flow GRPO requires at least two samples per prompt group.")
        if self.policy_update_epochs < 1:
            raise ValueError("policy_update_epochs must be positive.")
        if self.sde_window_start < 1:
            raise ValueError(
                "The SDE window cannot include singular t=0; start at step 1 or later."
            )
        if self.sde_window_size <= 0:
            raise ValueError("sde_window_size must be positive.")
        if self.sde_window_start + self.sde_window_size > self.num_steps:
            raise ValueError("The SDE training window must fit within num_steps.")
        if not math.isfinite(self.noise_level) or self.noise_level <= 0:
            raise ValueError("noise_level must be positive.")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be between zero and one.")
        if (
            not math.isfinite(self.kl_beta)
            or not math.isfinite(self.supervised_replay_weight)
            or self.kl_beta < 0
            or self.supervised_replay_weight < 0
        ):
            raise ValueError("KL and supervised replay weights must be nonnegative.")
        if (
            not math.isfinite(self.max_grad_norm)
            or not math.isfinite(self.advantage_epsilon)
            or self.max_grad_norm <= 0
            or self.advantage_epsilon <= 0
        ):
            raise ValueError("Gradient norm and advantage epsilon must be positive.")
        if not math.isfinite(self.log_ratio_clip) or not 0 < self.log_ratio_clip <= 80:
            raise ValueError("log_ratio_clip must be finite and in (0, 80].")
        minimum_ratio_range = max(
            math.log1p(self.clip_ratio),
            -math.log1p(-self.clip_ratio),
        )
        if self.log_ratio_clip <= minimum_ratio_range:
            raise ValueError("log_ratio_clip must contain both PPO clipping boundaries.")
        _validate_event_reduction(self.event_reduction)

    @property
    def stochastic_steps(self) -> range:
        """Return the discrete integration steps optimized by GRPO."""
        return range(self.sde_window_start, self.sde_window_start + self.sde_window_size)


@dataclass(frozen=True)
class FlowGRPOTrajectory:
    """Detached rollout states needed to recompute stochastic transition likelihoods."""

    final_state: Tensor
    states: tuple[Tensor, ...]
    next_states: tuple[Tensor, ...]
    times: tuple[Tensor, ...]
    step_sizes: tuple[Tensor, ...]
    standard_deviations: tuple[Tensor, ...]
    old_log_probs: Tensor
    event_reduction: EventReduction = "mean"
    event_mask: Tensor | None = None

    def __post_init__(self) -> None:
        count = len(self.states)
        if not count:
            raise ValueError("A Flow GRPO trajectory must contain at least one stochastic step.")
        if not all(
            len(values) == count
            for values in (
                self.next_states,
                self.times,
                self.step_sizes,
                self.standard_deviations,
            )
        ):
            raise ValueError("Trajectory fields must contain the same number of stochastic steps.")
        if self.final_state.ndim < 3:
            raise ValueError("Trajectory states must have [batch, group, ...event] shape.")
        if self.final_state.shape[0] == 0 or self.final_state.shape[1] < 2:
            raise ValueError(
                "Trajectory states require nonempty groups of at least two candidates."
            )
        if self.old_log_probs.ndim != 3 or self.old_log_probs.shape[-1] != count:
            raise ValueError("old_log_probs must end with the stochastic-step dimension.")
        expected_state_shape = self.final_state.shape
        expected_log_prob_shape = (*expected_state_shape[:2], count)
        if self.old_log_probs.shape != expected_log_prob_shape:
            raise ValueError("old_log_probs must have shape [batch, group, stochastic_step].")
        if any(state.shape != expected_state_shape for state in self.states):
            raise ValueError("Every trajectory state must have the final-state shape.")
        if any(state.shape != expected_state_shape for state in self.next_states):
            raise ValueError("Every next trajectory state must have the final-state shape.")
        detached_tensors = (
            self.final_state,
            self.old_log_probs,
            *self.states,
            *self.next_states,
            *self.times,
            *self.step_sizes,
            *self.standard_deviations,
        )
        if any(tensor.requires_grad for tensor in detached_tensors):
            raise ValueError("Rollout trajectories must be detached from autograd.")
        if any(not torch.isfinite(tensor).all() for tensor in detached_tensors):
            raise ValueError("Rollout trajectories must contain only finite values.")
        if any(torch.any(time <= 0) or torch.any(time >= 1) for time in self.times):
            raise ValueError("Stochastic trajectory times must be in the open interval (0, 1).")
        if any(torch.any(step_size <= 0) for step_size in self.step_sizes):
            raise ValueError("Trajectory step sizes must be positive.")
        if any(torch.any(value <= 0) for value in self.standard_deviations):
            raise ValueError("Trajectory standard deviations must be positive.")
        if self.event_mask is not None:
            _broadcast_event_mask(self.event_mask, self.final_state)
        _validate_event_reduction(self.event_reduction)


@dataclass(frozen=True)
class FlowGRPOMetrics:
    """Detached scalars returned by one optimizer update."""

    loss: float
    policy_loss: float
    kl: float
    supervised_loss: float
    reward: float
    reward_std: float
    grad_norm: float
    mean_abs_log_ratio: float
    clip_fraction: float


def _validate_event_reduction(reduction: str) -> None:
    if reduction not in {"mean", "sum"}:
        raise ValueError("event_reduction must be either 'mean' or 'sum'.")


def _complete_collective_phase(
    synchronizer: DistributedErrorSynchronizer | None,
    error: Exception | None,
    description: str,
) -> None:
    """Stop every rank at the same boundary before another model collective."""
    if synchronizer is not None:
        synchronizer(error, description)
        if error is not None:
            raise RuntimeError(
                "The distributed error synchronizer returned after a local failure."
            ) from error
    elif error is not None:
        raise error


def _event_dimensions(value: Tensor) -> tuple[int, ...]:
    if value.ndim < 3:
        raise ValueError("Flow GRPO tensors must have shape [batch, group, ...event dimensions].")
    return tuple(range(2, value.ndim))


def _floating_work_tensor(value: Tensor) -> Tensor:
    if not value.is_floating_point():
        raise ValueError("Flow GRPO numerical tensors must use a floating-point dtype.")
    if value.dtype in {torch.float16, torch.bfloat16}:
        return value.float()
    return value


def _broadcast_parameter(value: Tensor, reference: Tensor, name: str) -> Tensor:
    value = value.to(device=reference.device, dtype=reference.dtype)
    event_rank = reference.ndim - 2
    if value.ndim == 1 and value.shape == reference.shape[:1]:
        value = value.reshape(reference.shape[0], 1, *(1 for _ in range(event_rank)))
    elif value.ndim == 2 and value.shape == reference.shape[:2]:
        value = value.reshape(*reference.shape[:2], *(1 for _ in range(event_rank)))
    try:
        result_shape = torch.broadcast_shapes(value.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(f"{name} is not broadcastable to the flow state shape.") from error
    if result_shape != reference.shape:
        raise ValueError(f"{name} must not add dimensions to the flow state shape.")
    return value


def _broadcast_event_mask(mask: Tensor, reference: Tensor) -> Tensor:
    """Validate and expand a [batch, group, ...event] validity mask."""
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("event_mask must be a boolean tensor.")
    if mask.requires_grad:
        raise ValueError("event_mask must be detached from autograd.")
    if mask.ndim != reference.ndim or mask.shape[0] != reference.shape[0]:
        raise ValueError("event_mask must have the state rank and batch dimension.")
    if mask.shape[1] not in {1, reference.shape[1]}:
        raise ValueError("event_mask group dimension must be one or match the state.")
    if any(
        mask_size not in {1, state_size}
        for mask_size, state_size in zip(mask.shape, reference.shape)
    ):
        raise ValueError("event_mask dimensions must be singleton or match the state shape.")
    expanded = torch.broadcast_to(mask.to(device=reference.device), reference.shape)
    valid_per_candidate = expanded.flatten(start_dim=2).sum(dim=2)
    if bool((valid_per_candidate <= 0).any()):
        raise ValueError("event_mask must retain at least one event element per candidate.")
    return expanded


def _reduce_event(
    value: Tensor,
    reduction: EventReduction,
    event_mask: Tensor | None = None,
) -> Tensor:
    _validate_event_reduction(reduction)
    dimensions = _event_dimensions(value)
    if event_mask is not None:
        mask = _broadcast_event_mask(event_mask, value)
        masked = value.masked_fill(~mask, 0)
        reduced = masked.sum(dim=dimensions)
        if reduction == "sum":
            return reduced
        return reduced / mask.sum(dim=dimensions).to(dtype=value.dtype)
    if reduction == "sum":
        return value.sum(dim=dimensions)
    return value.mean(dim=dimensions)


def group_relative_advantages(rewards: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Standardize rewards independently inside each prompt group."""
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, group].")
    if rewards.shape[0] == 0:
        raise ValueError("rewards must contain at least one prompt group.")
    if rewards.shape[1] < 2:
        raise ValueError("Each reward group must contain at least two candidates.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    rewards = _floating_work_tensor(rewards)
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must contain only finite values.")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scale = rewards.std(dim=1, unbiased=False, keepdim=True)
    return torch.where(
        scale > epsilon,
        centered / scale.clamp_min(epsilon),
        torch.zeros_like(rewards),
    )


def combine_reward_components(
    components: Mapping[str, Tensor],
    weights: Mapping[str, float],
    *,
    epsilon: float = 1e-6,
) -> Tensor:
    """Combine batch-standardized reward components without hiding missing objectives."""
    if not components:
        raise ValueError("At least one reward component is required.")
    if set(components) != set(weights):
        missing_weights = sorted(set(components) - set(weights))
        missing_components = sorted(set(weights) - set(components))
        raise ValueError(
            "Reward components and weights must match exactly; "
            f"missing weights={missing_weights}, missing components={missing_components}."
        )
    shape = next(iter(components.values())).shape
    if len(shape) != 2:
        raise ValueError("Reward components must have shape [batch, group].")
    if shape[0] == 0 or shape[1] < 2:
        raise ValueError("Reward components require nonempty groups of at least two candidates.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")

    first_component = next(iter(components.values()))
    component_dtype = first_component.dtype
    first = _floating_work_tensor(first_component)
    combined = torch.zeros_like(first)
    for name in sorted(components):
        values = components[name]
        if values.shape != shape:
            raise ValueError("Every reward component must use the same [batch, group] shape.")
        if values.device != first.device or values.dtype != component_dtype:
            raise ValueError("Every reward component must use the same device and dtype.")
        values = _floating_work_tensor(values)
        if not torch.isfinite(values).all():
            raise ValueError(f"Reward component {name!r} contains nonfinite values.")
        weight = float(weights[name])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Reward weight {name!r} must be finite and nonnegative.")
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            statistics = torch.stack(
                (
                    values.sum(),
                    values.square().sum(),
                    torch.tensor(values.numel(), device=values.device, dtype=values.dtype),
                )
            )
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            total, total_square, count = statistics
            mean = total / count
            variance = (total_square / count - mean.square()).clamp_min(0)
            scale = variance.sqrt()
        else:
            scale = values.std(unbiased=False)
        normalized = torch.where(
            scale > epsilon,
            values / scale.clamp_min(epsilon),
            torch.zeros_like(values),
        )
        combined = combined + weight * normalized
    return combined


def _validate_reward_components(
    components: Any,
    weights: Mapping[str, float],
    *,
    expected_shape: torch.Size,
) -> None:
    """Validate rank-local reward values before any global statistics collective."""
    if not isinstance(components, Mapping) or set(components) != set(weights):
        raise ValueError("The reward function must return every weighted component exactly.")
    for name in sorted(weights):
        values = components[name]
        if not isinstance(values, torch.Tensor) or values.shape != expected_shape:
            raise ValueError(f"Reward component {name!r} must have shape {tuple(expected_shape)}.")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError(
                f"Reward component {name!r} must contain finite floating-point values."
            )
        weight = weights[name]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"Reward weight {name!r} must be a finite nonnegative number.")
        if not math.isfinite(float(weight)) or float(weight) < 0:
            raise ValueError(f"Reward weight {name!r} must be finite and nonnegative.")


def diagonal_gaussian_log_prob(
    value: Tensor,
    mean: Tensor,
    standard_deviation: Tensor,
    *,
    event_reduction: EventReduction = "mean",
    event_mask: Tensor | None = None,
) -> Tensor:
    """Return a diagonal-Gaussian transition log probability per candidate.

    ``mean`` matches established Flow-GRPO implementations and keeps PPO ratios comparable across
    differently sized latent buckets. ``sum`` is the exact joint event log density in Equation 11.
    """
    if value.shape != mean.shape:
        raise ValueError("value and mean must have identical shapes.")
    if value.device != mean.device:
        raise ValueError("value and mean must use the same device.")
    _event_dimensions(value)
    value = _floating_work_tensor(value)
    mean = _floating_work_tensor(mean).to(dtype=value.dtype)
    if not torch.isfinite(value).all() or not torch.isfinite(mean).all():
        raise ValueError("value and mean must contain only finite values.")
    standard_deviation = _broadcast_parameter(
        standard_deviation,
        value,
        "standard_deviation",
    )
    if not torch.isfinite(standard_deviation).all():
        raise ValueError("standard_deviation must contain only finite values.")
    if torch.any(standard_deviation <= 0):
        raise ValueError("standard_deviation must be positive.")
    variance = standard_deviation.square()
    if torch.any(variance == 0) or not torch.isfinite(variance).all():
        raise ValueError("standard_deviation is outside the numerically representable range.")
    log_prob = -0.5 * ((value - mean).square() / variance + torch.log(2 * math.pi * variance))
    return _reduce_event(log_prob, event_reduction, event_mask)


def flow_sde_transition(
    state: Tensor,
    velocity: Tensor,
    time: Tensor,
    step_size: Tensor,
    *,
    noise_level: float,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """Return Equation 7's 0-to-1 FlowTTS-GRPO transition parameters."""
    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have identical shapes.")
    _event_dimensions(state)
    state = _floating_work_tensor(state)
    velocity = _floating_work_tensor(velocity).to(device=state.device, dtype=state.dtype)
    if not torch.isfinite(state).all() or not torch.isfinite(velocity).all():
        raise ValueError("state and velocity must contain only finite values.")
    if not math.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("noise_level must be finite and positive.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    time = _broadcast_parameter(time, state, "time")
    step_size = _broadcast_parameter(step_size, state, "step_size")
    if not torch.isfinite(time).all() or not torch.isfinite(step_size).all():
        raise ValueError("time and step_size must contain only finite values.")
    if torch.any(time <= 0) or torch.any(time >= 1):
        raise ValueError("Stochastic flow transitions require 0 < time < 1.")
    if torch.any(step_size <= 0):
        raise ValueError("step_size must be positive.")

    safe_time = time.clamp_min(epsilon)
    sigma = noise_level * torch.sqrt((1 - time).clamp_min(epsilon) / safe_time)
    # Algebraically equal to sigma^2 / (2 * (1 - t)), without a near-t=1 cancellation.
    correction = noise_level**2 / (2 * safe_time)
    drift = velocity + correction * (-state + time * velocity)
    mean = state + drift * step_size
    standard_deviation = sigma * torch.sqrt(step_size)
    if not torch.isfinite(mean).all() or not torch.isfinite(standard_deviation).all():
        raise FloatingPointError("Nonfinite FlowTTS-GRPO transition parameters.")
    return mean, standard_deviation


@torch.no_grad()
def sample_flow_grpo_trajectory(
    policy: nn.Module,
    velocity_adapter: VelocityAdapter,
    initial_state: Tensor,
    conditioning: Any,
    config: FlowGRPOConfig,
    *,
    generator: torch.Generator | None = None,
    event_mask: Tensor | None = None,
    distributed_error_synchronizer: DistributedErrorSynchronizer | None = None,
) -> FlowGRPOTrajectory:
    """Sample a mixed deterministic/stochastic trajectory from noise to speech latents."""
    preflight_error: Exception | None = None
    try:
        if initial_state.ndim < 3:
            raise ValueError("initial_state must have shape [batch, group, ...event dimensions].")
        if initial_state.shape[0] == 0:
            raise ValueError("initial_state must contain at least one prompt group.")
        if initial_state.shape[1] != config.group_size:
            raise ValueError(
                f"initial_state group dimension {initial_state.shape[1]} does not match "
                f"configured group_size {config.group_size}."
            )
        initial_state = _floating_work_tensor(initial_state)
        if not torch.isfinite(initial_state).all():
            raise ValueError("initial_state must contain only finite values.")
        if event_mask is not None:
            # Preserve singleton dimensions in the stored mask while validating its
            # complete broadcast and every candidate's nonempty support now.
            _broadcast_event_mask(event_mask, initial_state)
            event_mask = event_mask.to(device=initial_state.device).detach()
    except Exception as exc:
        preflight_error = exc
    _complete_collective_phase(
        distributed_error_synchronizer,
        preflight_error,
        "GRPO rollout input validation",
    )

    times = torch.linspace(
        0,
        1,
        config.num_steps + 1,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    state = initial_state
    states: list[Tensor] = []
    next_states: list[Tensor] = []
    stochastic_times: list[Tensor] = []
    step_sizes: list[Tensor] = []
    standard_deviations: list[Tensor] = []
    old_log_probs: list[Tensor] = []
    stochastic_steps = set(config.stochastic_steps)

    for step in range(config.num_steps):
        time = times[step]
        step_size = times[step + 1] - time
        time_batch = time.expand(initial_state.shape[:2])
        velocity = None
        velocity_error: Exception | None = None
        try:
            velocity = velocity_adapter(policy, state, time_batch, conditioning)
            if not isinstance(velocity, torch.Tensor) or velocity.shape != state.shape:
                raise ValueError("velocity_adapter must return a tensor with the state shape.")
            if not torch.isfinite(velocity).all():
                raise FloatingPointError("velocity_adapter returned nonfinite values.")
        except Exception as exc:
            velocity_error = exc
        _complete_collective_phase(
            distributed_error_synchronizer,
            velocity_error,
            f"GRPO rollout velocity step {step}",
        )
        assert velocity is not None

        next_state = None
        mean = None
        standard_deviation = None
        old_log_prob = None
        transition_error: Exception | None = None
        try:
            if step in stochastic_steps:
                mean, standard_deviation = flow_sde_transition(
                    state,
                    velocity,
                    time,
                    step_size,
                    noise_level=config.noise_level,
                    epsilon=config.advantage_epsilon,
                )
                noise = torch.randn(
                    state.shape,
                    generator=generator,
                    device=state.device,
                    dtype=state.dtype,
                )
                next_state = mean + standard_deviation * noise
                old_log_prob = diagonal_gaussian_log_prob(
                    next_state,
                    mean,
                    standard_deviation,
                    event_reduction=config.event_reduction,
                    event_mask=event_mask,
                ).detach()
            else:
                next_state = state + velocity * step_size
            if not torch.isfinite(next_state).all():
                raise FloatingPointError("Flow GRPO rollout produced a nonfinite state.")
        except Exception as exc:
            transition_error = exc
        _complete_collective_phase(
            distributed_error_synchronizer,
            transition_error,
            f"GRPO rollout transition step {step}",
        )
        assert next_state is not None

        if step in stochastic_steps:
            assert standard_deviation is not None
            assert old_log_prob is not None
            states.append(state.detach())
            next_states.append(next_state.detach())
            stochastic_times.append(time.detach())
            step_sizes.append(step_size.detach())
            standard_deviations.append(standard_deviation.detach())
            old_log_probs.append(old_log_prob)
        state = next_state

    return FlowGRPOTrajectory(
        final_state=state.detach(),
        states=tuple(states),
        next_states=tuple(next_states),
        times=tuple(stochastic_times),
        step_sizes=tuple(step_sizes),
        standard_deviations=tuple(standard_deviations),
        old_log_probs=torch.stack(old_log_probs, dim=-1),
        event_reduction=config.event_reduction,
        event_mask=event_mask,
    )


def recompute_transition_statistics(
    policy: nn.Module,
    velocity_adapter: VelocityAdapter,
    trajectory: FlowGRPOTrajectory,
    conditioning: Any,
    config: FlowGRPOConfig,
    *,
    distributed_error_synchronizer: DistributedErrorSynchronizer | None = None,
    phase_name: str = "GRPO policy recomputation",
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Recompute transition log probabilities and means under a policy."""
    preflight_error: Exception | None = None
    try:
        if trajectory.old_log_probs.shape[1] != config.group_size:
            raise ValueError("Trajectory group size does not match the GRPO configuration.")
        if len(trajectory.states) != config.sde_window_size:
            raise ValueError("Trajectory step count does not match the configured SDE window.")
        if trajectory.event_reduction != config.event_reduction:
            raise ValueError(
                "Rollout and update must use the same event log-probability reduction."
            )
    except Exception as exc:
        preflight_error = exc
    _complete_collective_phase(
        distributed_error_synchronizer,
        preflight_error,
        f"{phase_name} input validation",
    )
    log_probs: list[Tensor] = []
    means: list[Tensor] = []
    for step_index, (state, next_state, time, step_size, standard_deviation) in enumerate(
        zip(
            trajectory.states,
            trajectory.next_states,
            trajectory.times,
            trajectory.step_sizes,
            trajectory.standard_deviations,
            strict=True,
        )
    ):
        time_batch = time.expand(state.shape[:2])
        velocity = None
        velocity_error: Exception | None = None
        try:
            velocity = velocity_adapter(policy, state, time_batch, conditioning)
            if not isinstance(velocity, torch.Tensor) or velocity.shape != state.shape:
                raise ValueError("velocity_adapter must return a tensor with the state shape.")
            if not torch.isfinite(velocity).all():
                raise FloatingPointError("velocity_adapter returned nonfinite values.")
        except Exception as exc:
            velocity_error = exc
        _complete_collective_phase(
            distributed_error_synchronizer,
            velocity_error,
            f"{phase_name} velocity step {step_index}",
        )
        assert velocity is not None
        mean = None
        log_prob = None
        transition_error: Exception | None = None
        try:
            mean, recomputed_standard_deviation = flow_sde_transition(
                state,
                velocity,
                time,
                step_size,
                noise_level=config.noise_level,
                epsilon=config.advantage_epsilon,
            )
            if not torch.allclose(recomputed_standard_deviation, standard_deviation):
                raise RuntimeError("The GRPO rollout and update transition variances do not match.")
            log_prob = diagonal_gaussian_log_prob(
                next_state,
                mean,
                standard_deviation,
                event_reduction=config.event_reduction,
                event_mask=trajectory.event_mask,
            )
        except Exception as exc:
            transition_error = exc
        _complete_collective_phase(
            distributed_error_synchronizer,
            transition_error,
            f"{phase_name} transition step {step_index}",
        )
        assert mean is not None
        assert log_prob is not None
        means.append(mean)
        log_probs.append(log_prob)
    return torch.stack(log_probs, dim=-1), tuple(means)


def same_variance_gaussian_kl(
    policy_means: tuple[Tensor, ...],
    reference_means: tuple[Tensor, ...],
    standard_deviations: tuple[Tensor, ...],
    *,
    event_reduction: EventReduction = "mean",
    event_mask: Tensor | None = None,
) -> Tensor:
    """Return analytic KL terms for Gaussian transitions with the same variance."""
    if not policy_means:
        raise ValueError("At least one stochastic transition is required for KL computation.")
    if len(policy_means) != len(reference_means) or len(policy_means) != len(standard_deviations):
        raise ValueError("Policy means, reference means, and deviations must have equal lengths.")
    _validate_event_reduction(event_reduction)
    penalties: list[Tensor] = []
    for policy_mean, reference_mean, standard_deviation in zip(
        policy_means,
        reference_means,
        standard_deviations,
        strict=True,
    ):
        if policy_mean.shape != reference_mean.shape:
            raise ValueError("Policy and reference transition means must have identical shapes.")
        policy_mean = _floating_work_tensor(policy_mean)
        reference_mean = (
            _floating_work_tensor(reference_mean)
            .to(device=policy_mean.device, dtype=policy_mean.dtype)
            .detach()
        )
        if not torch.isfinite(policy_mean).all() or not torch.isfinite(reference_mean).all():
            raise ValueError("Transition means must contain only finite values.")
        standard_deviation = _broadcast_parameter(
            standard_deviation,
            policy_mean,
            "standard_deviation",
        )
        if not torch.isfinite(standard_deviation).all() or torch.any(standard_deviation <= 0):
            raise ValueError("Transition standard deviations must be finite and positive.")
        variance = standard_deviation.square()
        if torch.any(variance == 0) or not torch.isfinite(variance).all():
            raise ValueError("Transition variance is outside the numerically representable range.")
        elementwise_kl = (policy_mean - reference_mean).square() / (2 * variance)
        if not torch.isfinite(elementwise_kl).all():
            raise FloatingPointError("Nonfinite Gaussian KL penalty.")
        penalties.append(_reduce_event(elementwise_kl, event_reduction, event_mask))
    return torch.stack(penalties, dim=-1)


def clipped_grpo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    kl_penalty: Tensor,
    *,
    clip_ratio: float,
    kl_beta: float,
    log_ratio_clip: float = 20.0,
) -> tuple[Tensor, Tensor]:
    """Return the minimized clipped GRPO loss and its clipped policy component."""
    if new_log_probs.shape != old_log_probs.shape or new_log_probs.shape != kl_penalty.shape:
        raise ValueError("New, old, and KL tensors must share [batch, group, step] shape.")
    if new_log_probs.ndim != 3 or any(size == 0 for size in new_log_probs.shape):
        raise ValueError("New, old, and KL tensors must have nonempty [batch, group, step] shape.")
    if advantages.shape != new_log_probs.shape[:2]:
        raise ValueError("advantages must have shape [batch, group].")
    if (
        not math.isfinite(clip_ratio)
        or not math.isfinite(kl_beta)
        or not math.isfinite(log_ratio_clip)
        or not 0 < clip_ratio < 1
        or kl_beta < 0
        or not 0 < log_ratio_clip <= 80
    ):
        raise ValueError("Invalid GRPO clipping or KL parameters.")
    minimum_ratio_range = max(math.log1p(clip_ratio), -math.log1p(-clip_ratio))
    if log_ratio_clip <= minimum_ratio_range:
        raise ValueError("log_ratio_clip must contain both PPO clipping boundaries.")

    new_log_probs = _floating_work_tensor(new_log_probs)
    old_log_probs = _floating_work_tensor(old_log_probs).to(
        device=new_log_probs.device,
        dtype=new_log_probs.dtype,
    )
    advantages = _floating_work_tensor(advantages).to(
        device=new_log_probs.device,
        dtype=new_log_probs.dtype,
    )
    kl_penalty = _floating_work_tensor(kl_penalty).to(
        device=new_log_probs.device,
        dtype=new_log_probs.dtype,
    )
    if not all(
        torch.isfinite(value).all()
        for value in (new_log_probs, old_log_probs, advantages, kl_penalty)
    ):
        raise ValueError("GRPO loss inputs must contain only finite values.")
    if torch.any(kl_penalty < 0):
        raise ValueError("kl_penalty must be nonnegative.")

    raw_log_ratio = new_log_probs - old_log_probs.detach()
    bounded_log_ratio = raw_log_ratio.clamp(-log_ratio_clip, log_ratio_clip)
    # Bound the forward exponential while retaining the corrective gradient on the PPO-unclipped
    # branch. A plain clamp incorrectly gives zero gradient for large ratios with negative A.
    log_ratio = raw_log_ratio + (bounded_log_ratio - raw_log_ratio).detach()
    ratio = log_ratio.exp()
    expanded_advantages = advantages.detach().unsqueeze(-1)
    unclipped = ratio * expanded_advantages
    clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * expanded_advantages
    policy_objective = torch.minimum(unclipped, clipped).mean()
    objective = policy_objective - kl_beta * kl_penalty.mean()
    if not torch.isfinite(objective):
        raise FloatingPointError("Nonfinite Flow GRPO objective.")
    return -objective, -policy_objective


class FlowGRPOTrainer:
    """One-step optimizer for flow-native GRPO with injected speech reward evaluators.

    The policy may be wrapped by DDP. All samples for one prompt group must stay on the same rank;
    callers are responsible for distributed sampling and for reducing rank-local metrics before
    rank-zero logging.
    """

    def __init__(
        self,
        *,
        policy: nn.Module,
        reference_policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        velocity_adapter: VelocityAdapter,
        decode: DecodeFunction,
        reward: RewardFunction,
        reward_weights: Mapping[str, float],
        config: FlowGRPOConfig,
        supervised_loss: SupervisedLossFunction | None = None,
        policy_reference_manifest_sha256: str | None = None,
        reference_manifest_sha256: str | None = None,
        optimizer_step_callback: OptimizerStepCallback | None = None,
        distributed_error_synchronizer: DistributedErrorSynchronizer | None = None,
    ) -> None:
        self.policy = policy
        self.reference_policy = reference_policy
        self.optimizer = optimizer
        self.velocity_adapter = velocity_adapter
        self.decode = decode
        self.reward = reward
        self.reward_weights = dict(reward_weights)
        self.config = config
        self.supervised_loss = supervised_loss
        self.optimizer_step_callback = optimizer_step_callback
        self.distributed_error_synchronizer = distributed_error_synchronizer

        policy_parameter_ids = {id(parameter) for parameter in self.policy.parameters()}
        reference_parameter_ids = {
            id(parameter) for parameter in self.reference_policy.parameters()
        }
        if policy is reference_policy or policy_parameter_ids & reference_parameter_ids:
            raise ValueError("policy and reference_policy must not share trainable parameters.")

        policy_module = self._unwrap_module(self.policy)
        reference_module = self._unwrap_module(self.reference_policy)
        if type(policy_module) is not type(reference_module):
            raise ValueError("policy and reference_policy must use the same module type.")
        policy_state = policy_module.state_dict()
        reference_state = reference_module.state_dict()
        if set(policy_state) != set(reference_state):
            raise ValueError("policy and reference_policy must have identical state keys.")
        incompatible = sorted(
            key
            for key in policy_state
            if policy_state[key].shape != reference_state[key].shape
            or policy_state[key].dtype != reference_state[key].dtype
        )
        if incompatible:
            raise ValueError(
                f"policy and reference_policy state shapes/dtypes differ for keys: {incompatible}."
            )

        lineage_supplied = (
            policy_reference_manifest_sha256 is not None or reference_manifest_sha256 is not None
        )
        if lineage_supplied:
            hashes = {
                "policy_reference_manifest_sha256": policy_reference_manifest_sha256,
                "reference_manifest_sha256": reference_manifest_sha256,
            }
            invalid = [
                name
                for name, value in hashes.items()
                if not isinstance(value, str) or not _MANIFEST_SHA256.fullmatch(value)
            ]
            if invalid:
                raise ValueError(
                    "Both GRPO reference-lineage fields must be lowercase manifest SHA-256 "
                    f"values; invalid: {invalid}."
                )
            if policy_reference_manifest_sha256 != reference_manifest_sha256:
                raise ValueError(
                    "The policy's recorded reference manifest does not match the frozen "
                    "reference policy manifest."
                )
        else:
            unequal = [
                key
                for key in policy_state
                if not torch.equal(
                    policy_state[key].detach().to(device="cpu"),
                    reference_state[key].detach().to(device="cpu"),
                )
            ]
            if unequal:
                raise ValueError(
                    "A fresh GRPO policy and reference must start with identical state. For a "
                    "resumed, already-updated policy, provide matching validated manifest "
                    f"lineage hashes. Unequal keys: {unequal}."
                )

        trainable_parameters = [
            parameter for parameter in self.policy.parameters() if parameter.requires_grad
        ]
        trainable_ids = {id(parameter) for parameter in trainable_parameters}
        if not trainable_ids:
            raise ValueError("GRPO policy must expose at least one trainable parameter.")
        optimizer_parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group.get("params", ())
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise ValueError("GRPO optimizer parameter groups contain duplicate parameters.")
        if set(optimizer_ids) != trainable_ids:
            missing_count = len(trainable_ids - set(optimizer_ids))
            extra_count = len(set(optimizer_ids) - trainable_ids)
            raise ValueError(
                "GRPO optimizer must contain exactly the policy parameters with "
                "requires_grad=True; freeze intentional exclusions before constructing it. "
                f"Missing={missing_count}, extra={extra_count}."
            )
        self._optimized_parameters = tuple(trainable_parameters)
        self.reference_policy.eval()
        for parameter in self.reference_policy.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        """Remove common DDP/DataParallel/compile wrappers for reference validation."""
        current = module
        visited: set[int] = set()
        while id(current) not in visited:
            visited.add(id(current))
            wrapped = getattr(current, "module", None)
            if isinstance(wrapped, nn.Module) and wrapped is not current:
                current = wrapped
                continue
            original = getattr(current, "_orig_mod", None)
            if isinstance(original, nn.Module) and original is not current:
                current = original
                continue
            break
        return current

    def _complete_distributed_phase(
        self,
        error: Exception | None,
        description: str,
    ) -> None:
        """Reach one rank-consistent verdict before the next DDP collective phase."""
        if self.distributed_error_synchronizer is not None:
            self.distributed_error_synchronizer(error, description)
            if error is not None:
                raise RuntimeError(
                    "The distributed error synchronizer returned after a local failure."
                ) from error
        elif error is not None:
            raise error

    def step(
        self,
        *,
        initial_state: Tensor,
        conditioning: Any,
        batch: Any,
        generator: torch.Generator | None = None,
        event_mask: Tensor | None = None,
    ) -> FlowGRPOMetrics:
        """Roll out candidates, evaluate rewards, and perform one policy update."""
        was_training = self.policy.training
        self.policy.eval()
        try:
            trajectory = None
            rollout_error: Exception | None = None
            try:
                trajectory = sample_flow_grpo_trajectory(
                    self.policy,
                    self.velocity_adapter,
                    initial_state,
                    conditioning,
                    self.config,
                    generator=generator,
                    event_mask=event_mask,
                    distributed_error_synchronizer=self.distributed_error_synchronizer,
                )
            except Exception as exc:
                rollout_error = exc
            self._complete_distributed_phase(rollout_error, "GRPO policy rollout")
            assert trajectory is not None

            reward_components = None
            evaluation_error: Exception | None = None
            try:
                with torch.no_grad():
                    audio = self.decode(trajectory.final_state, batch)
                    reward_components = self.reward(audio, batch)
                    _validate_reward_components(
                        reward_components,
                        self.reward_weights,
                        expected_shape=initial_state.shape[:2],
                    )
                    # Canonical device/dtype makes every rank enter reward-statistics
                    # collectives with the same tensor schema.
                    reward_components = {
                        name: reward_components[name].to(
                            device=initial_state.device,
                            dtype=torch.float32,
                        )
                        for name in sorted(self.reward_weights)
                    }
            except Exception as exc:
                evaluation_error = exc
            self._complete_distributed_phase(
                evaluation_error,
                "GRPO decode/reward validation",
            )
            assert reward_components is not None

            rewards = None
            combination_error: Exception | None = None
            try:
                with torch.no_grad():
                    rewards = combine_reward_components(reward_components, self.reward_weights)
                    advantages = group_relative_advantages(
                        rewards,
                        epsilon=self.config.advantage_epsilon,
                    )
            except Exception as exc:
                combination_error = exc
            self._complete_distributed_phase(
                combination_error,
                "GRPO global reward normalization",
            )
            assert rewards is not None

            reference_means = None
            reference_error: Exception | None = None
            try:
                with torch.no_grad():
                    _, reference_means = recompute_transition_statistics(
                        self.reference_policy,
                        self.velocity_adapter,
                        trajectory,
                        conditioning,
                        self.config,
                        distributed_error_synchronizer=self.distributed_error_synchronizer,
                        phase_name="GRPO frozen-reference recomputation",
                    )
            except Exception as exc:
                reference_error = exc
            self._complete_distributed_phase(
                reference_error,
                "GRPO frozen-reference evaluation",
            )
            assert reference_means is not None
            totals = {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl": 0.0,
                "supervised": 0.0,
                "grad_norm": 0.0,
                "mean_abs_log_ratio": 0.0,
                "clip_fraction": 0.0,
            }
            for update_epoch in range(self.config.policy_update_epochs):
                # Reuse one detached rollout for multiple policy epochs. The first ratio is
                # exactly one by construction; later epochs compare the updated policy against
                # the fixed rollout policy, which makes PPO clipping operational.
                self.policy.eval()
                self.optimizer.zero_grad(set_to_none=True)
                new_log_probs = None
                kl_penalty = None
                grpo_loss = None
                policy_loss = None
                objective_error: Exception | None = None
                try:
                    new_log_probs, policy_means = recompute_transition_statistics(
                        self.policy,
                        self.velocity_adapter,
                        trajectory,
                        conditioning,
                        self.config,
                        distributed_error_synchronizer=self.distributed_error_synchronizer,
                        phase_name=f"GRPO policy epoch {update_epoch + 1} recomputation",
                    )
                    kl_penalty = same_variance_gaussian_kl(
                        policy_means,
                        reference_means,
                        trajectory.standard_deviations,
                        event_reduction=self.config.event_reduction,
                        event_mask=trajectory.event_mask,
                    )
                    grpo_loss, policy_loss = clipped_grpo_loss(
                        new_log_probs,
                        trajectory.old_log_probs,
                        advantages,
                        kl_penalty,
                        clip_ratio=self.config.clip_ratio,
                        kl_beta=self.config.kl_beta,
                        log_ratio_clip=self.config.log_ratio_clip,
                    )
                except Exception as exc:
                    objective_error = exc
                self._complete_distributed_phase(
                    objective_error,
                    f"GRPO policy objective epoch {update_epoch + 1}",
                )
                assert new_log_probs is not None
                assert kl_penalty is not None
                assert grpo_loss is not None
                assert policy_loss is not None

                supervised = torch.zeros(
                    (),
                    device=grpo_loss.device,
                    dtype=grpo_loss.dtype,
                )
                replay_error: Exception | None = None
                try:
                    if self.supervised_loss is not None and self.config.supervised_replay_weight:
                        self.policy.train()
                        supervised = self.supervised_loss(self.policy, batch)
                        if supervised.ndim != 0 or not torch.isfinite(supervised):
                            raise ValueError(
                                "supervised_loss must return one finite scalar tensor."
                            )
                except Exception as exc:
                    replay_error = exc
                self._complete_distributed_phase(
                    replay_error,
                    f"GRPO supervised replay epoch {update_epoch + 1}",
                )

                loss = None
                combined_error: Exception | None = None
                try:
                    loss = grpo_loss + self.config.supervised_replay_weight * supervised
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite combined Flow GRPO loss.")
                except Exception as exc:
                    combined_error = exc
                self._complete_distributed_phase(
                    combined_error,
                    f"GRPO combined objective epoch {update_epoch + 1}",
                )
                assert loss is not None

                backward_error: Exception | None = None
                try:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self._optimized_parameters,
                        self.config.max_grad_norm,
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("Nonfinite gradient norm in Flow GRPO update.")
                except Exception as exc:
                    backward_error = exc
                self._complete_distributed_phase(
                    backward_error,
                    f"GRPO backward epoch {update_epoch + 1}",
                )

                update_error: Exception | None = None
                try:
                    self.optimizer.step()
                    if self.optimizer_step_callback is not None:
                        self.optimizer_step_callback()
                except Exception as exc:
                    update_error = exc
                self._complete_distributed_phase(
                    update_error,
                    f"GRPO optimizer epoch {update_epoch + 1}",
                )

                with torch.no_grad():
                    log_ratio = new_log_probs - trajectory.old_log_probs
                    clipped = (log_ratio < math.log1p(-self.config.clip_ratio)) | (
                        log_ratio > math.log1p(self.config.clip_ratio)
                    )
                    totals["loss"] += float(loss.detach())
                    totals["policy_loss"] += float(policy_loss.detach())
                    totals["kl"] += float(kl_penalty.mean().detach())
                    totals["supervised"] += float(supervised.detach())
                    totals["grad_norm"] += float(grad_norm.detach())
                    totals["mean_abs_log_ratio"] += float(log_ratio.abs().mean().detach())
                    totals["clip_fraction"] += float(clipped.float().mean().detach())

            updates = float(self.config.policy_update_epochs)

            return FlowGRPOMetrics(
                loss=totals["loss"] / updates,
                policy_loss=totals["policy_loss"] / updates,
                kl=totals["kl"] / updates,
                supervised_loss=totals["supervised"] / updates,
                reward=float(rewards.mean().detach()),
                reward_std=float(rewards.std(unbiased=False).detach()),
                grad_norm=totals["grad_norm"] / updates,
                mean_abs_log_ratio=totals["mean_abs_log_ratio"] / updates,
                clip_fraction=totals["clip_fraction"] / updates,
            )
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            self.policy.train(was_training)


__all__ = [
    "EventReduction",
    "DistributedErrorSynchronizer",
    "FlowGRPOConfig",
    "FlowGRPOMetrics",
    "FlowGRPOTrainer",
    "FlowGRPOTrajectory",
    "OptimizerStepCallback",
    "clipped_grpo_loss",
    "combine_reward_components",
    "diagonal_gaussian_log_prob",
    "flow_sde_transition",
    "group_relative_advantages",
    "recompute_transition_statistics",
    "same_variance_gaussian_kl",
    "sample_flow_grpo_trajectory",
]
