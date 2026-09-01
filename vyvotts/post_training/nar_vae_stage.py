"""Production NAR-VAE model and data adapters for the executable GRPO stage."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vyvotts.checkpoint import CheckpointProvenance, FlowCheckpoint
from vyvotts.dacvae import HubDACVAESource, load_dacvae
from vyvotts.dataset.data_collator import FlowMatchingDataCollator
from vyvotts.dataset.identity import resolve_local_prepared_dataset_identity
from vyvotts.distributed import (
    distributed_cleanup_guard,
    initialize_distributed,
    propagate_distributed_error,
    resolve_node_consistent_value,
)
from vyvotts.losses.flow_matching_loss import FlowMatchingLoss
from vyvotts.model_manifest import (
    ModelManifest,
    validate_grpo_parent_manifest,
    validate_loaded_codec,
    validate_manifest_weight,
)
from vyvotts.models.flow_matching import create_flow_matching_echodit
from vyvotts.post_training.stage import (
    GRPOPreparedBatch,
    GRPOStageConfig,
    GRPOStageRuntime,
    grpo_reference_identity,
    load_grpo_stage_config,
    run_grpo_stage,
)
from vyvotts.training_utils import freeze_layers, validate_tts_dataset

SpeechRewardFunction = Callable[[torch.Tensor, Any], Mapping[str, torch.Tensor]]
DEFAULT_GRPO_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs/grpo_config.yaml",
)


def bind_reward_evaluator_manifest(
    reward: SpeechRewardFunction,
    evaluators: Mapping[str, Mapping[str, str]],
) -> SpeechRewardFunction:
    """Bind a callable to the evaluator identities recorded by the GRPO config."""
    if not callable(reward):
        raise TypeError("reward must be callable.")
    setattr(
        reward,
        "nar_vae_reward_evaluators",
        {name: dict(identity) for name, identity in evaluators.items()},
    )
    return reward


def model_export_config_from_manifest(manifest: ModelManifest) -> dict[str, Any]:
    """Reconstruct only the immutable model/export fields from a validated manifest."""
    architecture = dict(manifest.architecture)
    capabilities = dict(manifest.capabilities)
    representation = dict(manifest.representation)
    config = {
        **{
            name: architecture[name]
            for name in architecture
            if name
            not in {
                "latent_size",
                "text_vocab_size",
                "speaker_patch_size",
                "use_speaker_conditioning",
                "use_mas_duration",
                "norm_eps",
            }
        },
        "dacvae_latent_dim": architecture["latent_size"],
        "text_vocab_size": architecture["text_vocab_size"],
        "speaker_patch_size": architecture["speaker_patch_size"],
        "use_speaker_conditioning": capabilities["speaker_conditioning"],
        "use_language_conditioning": capabilities["language_conditioning"],
        "supported_languages": list(capabilities["supported_languages"]),
        "supported_reference_languages": (
            list(capabilities["supported_reference_languages"])
            if capabilities["supported_reference_languages"]
            else None
        ),
        "use_duration_predictor": capabilities["duration_predictor"],
        "duration_predictor_hidden_size": capabilities["duration_predictor_hidden_size"],
        "duration_predictor_num_layers": capabilities["duration_predictor_num_layers"],
        "duration_predictor_use_speaker": capabilities["duration_predictor_use_speaker"],
        "use_mas_duration": capabilities["monotonic_alignment"],
        "duration_alignment_hidden_size": capabilities["duration_alignment_hidden_size"],
        "norm_eps": architecture["norm_eps"],
        "dacvae_model": representation["codec_source"],
        "dacvae_backend": representation["codec_backend"],
        "dacvae_revision": representation["codec_revision"],
        "dacvae_filename": representation["codec_filename"],
        "dacvae_sha256": representation["codec_sha256"],
        "dacvae_sample_rate": representation["sample_rate"],
        "dacvae_hop_length": representation["hop_length"],
    }
    return config


def _new_model_from_manifest(
    manifest: ModelManifest,
    *,
    cfg_dropout: float = 0.1,
) -> nn.Module:
    architecture = manifest.architecture
    capabilities = manifest.capabilities
    model_kwargs = {
        name: architecture[name]
        for name in (
            "model_size",
            "num_layers",
            "num_heads",
            "intermediate_size",
            "text_model_size",
            "text_num_layers",
            "text_num_heads",
            "text_intermediate_size",
            "speaker_model_size",
            "speaker_num_layers",
            "speaker_num_heads",
            "speaker_intermediate_size",
            "timestep_embed_size",
            "adaln_rank",
        )
    }
    return create_flow_matching_echodit(
        latent_size=architecture["latent_size"],
        text_vocab_size=architecture["text_vocab_size"],
        speaker_patch_size=architecture["speaker_patch_size"],
        norm_eps=architecture["norm_eps"],
        cfg_dropout=cfg_dropout,
        use_speaker_conditioning=capabilities["speaker_conditioning"],
        use_language_conditioning=capabilities["language_conditioning"],
        supported_languages=capabilities["supported_languages"],
        supported_reference_languages=capabilities["supported_reference_languages"],
        use_duration_predictor=capabilities["duration_predictor"],
        duration_predictor_hidden_size=capabilities["duration_predictor_hidden_size"] or 256,
        duration_predictor_num_layers=capabilities["duration_predictor_num_layers"] or 2,
        duration_predictor_use_speaker=capabilities["duration_predictor_use_speaker"],
        use_mas_duration=capabilities["monotonic_alignment"],
        duration_alignment_hidden_size=capabilities["duration_alignment_hidden_size"] or 64,
        **model_kwargs,
    )


def _preserve_flow_only_trainability(model: nn.Module) -> None:
    # GRPO optimizes the token-duration-conditioned velocity graph. Duration prediction and MAS
    # alignment are selected once by the frozen SFT reference and are not part of that graph;
    # leaving these policy-only branches trainable makes default DDP hang on unused gradients.
    for branch_name in ("duration_predictor", "duration_alignment"):
        branch = getattr(model, branch_name, None)
        if isinstance(branch, nn.Module):
            branch.requires_grad_(False)
    setter = getattr(getattr(model, "dit", None), "set_latent_prefix_trainable", None)
    if callable(setter):
        setter(False)


def _validate_grpo_parent_before_deserialization(
    parent_manifest: ModelManifest,
    provenance: CheckpointProvenance,
) -> None:
    """Re-authenticate every SFT tensor artifact at the ``torch.load`` boundary."""
    validate_manifest_weight(
        parent_manifest,
        provenance.path,
        selected_filename=provenance.selected_filename,
    )
    if provenance.ema_filename is None:
        return
    if provenance.base_path is None:
        raise FileNotFoundError(
            "A partial EMA checkpoint requires its full manifest-bound SFT base "
            f"checkpoint {provenance.base_filename!r}."
        )
    validate_manifest_weight(
        parent_manifest,
        provenance.base_path,
        selected_filename=provenance.base_filename,
    )


class NARVAEGRPOCollator:
    """Collate one row per prompt while retaining metadata for external rewards."""

    def __init__(self, *, pad_token: int, speaker_patch_size: int, prompt_id_column: str) -> None:
        self.base = FlowMatchingDataCollator(
            pad_token=pad_token,
            speaker_patch_size=speaker_patch_size,
        )
        self.prompt_id_column = prompt_id_column

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        model_inputs = self.base([dict(feature) for feature in features])
        prompt_metadata = tuple(
            (
                feature[self.prompt_id_column],
                {
                    name: value
                    for name, value in feature.items()
                    if name not in {"latents", "speaker_latents"}
                },
            )
            for feature in features
        )
        # Derive both sequences from the same pairs so prompt identifiers cannot drift from the
        # reward metadata consumed by per-prompt evaluators.
        prompt_ids = tuple(prompt_id for prompt_id, _ in prompt_metadata)
        reward_rows = tuple(reward_row for _, reward_row in prompt_metadata)
        latent_lengths = model_inputs["latent_mask"].sum(dim=1, dtype=torch.long)
        return {
            "model_inputs": model_inputs,
            "prompt_ids": prompt_ids,
            "reward_rows": reward_rows,
            "latent_lengths": latent_lengths,
        }


def _move_tensor_mapping(value: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: item.to(device, non_blocking=device.type == "cuda")
        if isinstance(item, torch.Tensor)
        else item
        for name, item in value.items()
    }


def _expand_prompt_tensor(value: torch.Tensor | None, *, group_size: int) -> torch.Tensor | None:
    if value is None:
        return None
    batch_size = value.shape[0]
    expanded = value.unsqueeze(1).expand(batch_size, group_size, *value.shape[1:])
    return expanded.reshape(batch_size * group_size, *value.shape[1:])


def _pad_token_durations_to_batch_frames(
    token_durations: torch.Tensor,
    token_mask: torch.Tensor | None,
    true_lengths: torch.Tensor,
    *,
    padded_frames: int,
) -> torch.Tensor:
    """Assign only padded tail frames to each row's final valid text token."""
    batch_size = token_durations.shape[0]
    padding = padded_frames - true_lengths.to(token_durations.device)
    if torch.any(padding < 0):
        raise ValueError("Latent lengths cannot exceed the collated frame dimension.")
    if token_mask is None:
        last_tokens = torch.full(
            (batch_size,),
            token_durations.shape[1] - 1,
            device=token_durations.device,
            dtype=torch.long,
        )
    else:
        token_counts = token_mask.sum(dim=1, dtype=torch.long)
        if torch.any(token_counts <= 0):
            raise ValueError("Every MAS GRPO prompt must contain a valid text token.")
        last_tokens = token_counts - 1
    padded = token_durations.clone()
    padded[torch.arange(batch_size, device=token_durations.device), last_tokens] += padding
    if not torch.equal(
        padded.sum(dim=1),
        torch.full((batch_size,), padded_frames, device=padded.device, dtype=padded.dtype),
    ):
        raise ValueError("Every padded MAS duration row must span the collated frame count.")
    return padded


def _velocity_adapter(
    model: nn.Module,
    state: torch.Tensor,
    time: torch.Tensor,
    conditioning: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    batch_size, group_size = state.shape[:2]
    flat_state = state.reshape(batch_size * group_size, *state.shape[2:])
    flat_time = time.reshape(batch_size * group_size)
    conditioning_ids = _expand_prompt_tensor(
        conditioning["conditioning_ids"], group_size=group_size
    )
    if conditioning_ids is None:  # pragma: no cover - protected by collator contract
        raise ValueError("GRPO conditioning_ids cannot be absent.")
    prediction = model(
        latents=flat_state,
        conditioning_ids=conditioning_ids,
        timesteps=flat_time,
        attention_mask=_expand_prompt_tensor(
            conditioning.get("conditioning_mask"), group_size=group_size
        ),
        speaker_latent=_expand_prompt_tensor(
            conditioning.get("speaker_latents"), group_size=group_size
        ),
        speaker_mask=_expand_prompt_tensor(conditioning.get("speaker_mask"), group_size=group_size),
        language_ids=_expand_prompt_tensor(conditioning.get("language_ids"), group_size=group_size),
        latent_mask=_expand_prompt_tensor(conditioning.get("latent_mask"), group_size=group_size),
        token_durations=_expand_prompt_tensor(
            conditioning.get("token_durations"), group_size=group_size
        ),
        use_cfg_dropout=False,
    )
    if not isinstance(prediction, torch.Tensor):
        raise RuntimeError("The GRPO velocity policy must return one tensor.")
    return prediction.reshape_as(state)


def _codec_source(manifest: ModelManifest) -> str | HubDACVAESource:
    representation = manifest.representation
    if representation["codec_revision"] is None:
        return str(representation["codec_source"])
    return HubDACVAESource(
        repo_id=str(representation["codec_source"]),
        revision=str(representation["codec_revision"]),
        filename=str(representation["codec_filename"]),
    )


def _decode_exact_latent_lengths(
    codec: nn.Module,
    final_state: torch.Tensor,
    latent_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode each prompt before padding so codec context never includes latent padding."""
    batch_size, group_size = final_state.shape[:2]
    if tuple(latent_lengths.shape) != (batch_size,):
        raise ValueError("latent_lengths must contain one value per prompt.")
    decoded: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    for index, latent_length in enumerate(latent_lengths.tolist()):
        length = int(latent_length)
        if length <= 0 or length > final_state.shape[-1]:
            raise ValueError("Every latent length must select a nonempty unpadded prefix.")
        prompt_audio = codec.decode(final_state[index, :, :, :length])
        if not isinstance(prompt_audio, torch.Tensor) or prompt_audio.shape[0] != group_size:
            raise RuntimeError("DACVAE decode must return one waveform per candidate.")
        decoded.append(prompt_audio)
        sample_lengths.append(prompt_audio.shape[-1])
    max_samples = max(sample_lengths)
    audio = torch.stack(
        [F.pad(item, (0, max_samples - item.shape[-1])) for item in decoded],
        dim=0,
    )
    return audio, torch.tensor(sample_lengths, device=audio.device, dtype=torch.long)


def _evaluate_exact_audio_lengths(
    reward: SpeechRewardFunction,
    audio: torch.Tensor,
    batch: Mapping[str, Any],
    *,
    component_names: Sequence[str],
    group_size: int,
) -> dict[str, torch.Tensor]:
    """Call evaluators only with each prompt's exact decoded waveform prefix."""
    components: dict[str, list[torch.Tensor]] = {name: [] for name in component_names}
    for index, sample_length in enumerate(batch["sample_lengths"].tolist()):
        single_batch = {
            "prompt_ids": (batch["prompt_ids"][index],),
            "reward_rows": (batch["reward_rows"][index],),
            "latent_lengths": batch["latent_lengths"][index : index + 1],
            "sample_lengths": batch["sample_lengths"][index : index + 1],
            "model_inputs": {
                name: value[index : index + 1] if isinstance(value, torch.Tensor) else value
                for name, value in batch["model_inputs"].items()
            },
        }
        result = reward(audio[index : index + 1, ..., : int(sample_length)], single_batch)
        if not isinstance(result, Mapping) or set(result) != set(components):
            raise ValueError(
                "The versioned reward callable must return every configured component exactly."
            )
        for name, value in result.items():
            if not isinstance(value, torch.Tensor) or value.shape != (1, group_size):
                raise ValueError(
                    f"Reward component {name!r} must have shape [1, group_size] per prompt."
                )
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(
                    f"Reward component {name!r} must contain finite floating-point values."
                )
            components[name].append(value)
    return {name: torch.cat(values, dim=0) for name, values in components.items()}


def build_nar_vae_grpo_runtime(
    config: GRPOStageConfig,
    *,
    parent_manifest: ModelManifest,
    reward: SpeechRewardFunction,
    device: torch.device,
    codec: nn.Module | None = None,
    pad_token: int = 100286,
) -> GRPOStageRuntime:
    """Build independent policy/reference copies and concrete speech adapters."""
    if not callable(reward):
        raise TypeError("reward must be a callable speech evaluator.")
    declared_evaluators = getattr(reward, "nar_vae_reward_evaluators", None)
    expected_evaluators = {
        name: dict(identity) for name, identity in config.reward_evaluators.items()
    }
    if declared_evaluators != expected_evaluators:
        raise ValueError(
            "The reward callable must bind the exact reward_evaluators configuration with "
            "bind_reward_evaluator_manifest()."
        )
    policy = _new_model_from_manifest(parent_manifest)
    reference = _new_model_from_manifest(parent_manifest)
    # The config/manifest identity selects one exact artifact; never auto-switch a base path to a
    # sibling EMA checkpoint merely because it exists.
    parent_weights = FlowCheckpoint.load(
        config.parent_checkpoint,
        prefer_ema=False,
        preload_validator=lambda provenance: _validate_grpo_parent_before_deserialization(
            parent_manifest,
            provenance,
        ),
    )
    parent_weights.load_into(policy)
    parent_weights.load_into(reference)
    freeze_layers(
        policy,
        {
            "freeze_text_encoder": config.freeze_text_encoder,
            "freeze_speaker_encoder": config.freeze_speaker_encoder,
            "freeze_language_embedding": config.freeze_language_embedding,
            "freeze_first_n_layers": config.freeze_first_n_layers,
        },
    )
    _preserve_flow_only_trainability(policy)

    if codec is None:
        representation = parent_manifest.representation
        codec = load_dacvae(
            _codec_source(parent_manifest),
            backend=str(representation["codec_backend"]),
            device=device,
            freeze=True,
            expected_latent_size=int(representation["latent_width"]),
            expected_sha256=str(representation["codec_sha256"]),
        )
    validate_loaded_codec(parent_manifest, codec)
    codec.eval()
    codec.requires_grad_(False)

    collator = NARVAEGRPOCollator(
        pad_token=pad_token,
        speaker_patch_size=int(parent_manifest.architecture["speaker_patch_size"]),
        prompt_id_column=config.prompt_id_column,
    )

    def prepare_batch(
        batch: Any,
        selected_device: torch.device,
        group_size: int,
        generator: torch.Generator,
    ) -> GRPOPreparedBatch:
        if not isinstance(batch, Mapping) or "model_inputs" not in batch:
            raise ValueError("NAR-VAE GRPO batches must come from NARVAEGRPOCollator.")
        model_inputs = _move_tensor_mapping(batch["model_inputs"], selected_device)
        latents = model_inputs["latents"]
        batch_size, latent_size, frames = latents.shape
        if parent_manifest.capabilities["monotonic_alignment"]:
            # Keep one fixed allocation across rollout, frozen-reference scoring, and every
            # repeated policy epoch. Target lengths come from the prompt row; allocation comes
            # from the immutable SFT reference rather than the changing policy.
            with torch.no_grad():
                token_durations = reference.predict_token_duration_frames(
                    model_inputs["conditioning_ids"],
                    model_inputs.get("conditioning_mask"),
                    model_inputs.get("speaker_latents"),
                    model_inputs.get("speaker_mask"),
                    model_inputs.get("language_ids"),
                    total_frames=batch["latent_lengths"].to(selected_device),
                ).detach()
                # expand_text_by_durations requires one tensor-wide frame count. Allocate only
                # the padded tail to the final valid token; every true frame keeps the exact
                # reference allocation, and event_mask removes the tail from GRPO statistics.
                model_inputs["token_durations"] = _pad_token_durations_to_batch_frames(
                    token_durations,
                    model_inputs.get("conditioning_mask"),
                    batch["latent_lengths"],
                    padded_frames=frames,
                )
        initial_state = torch.randn(
            (batch_size, group_size, latent_size, frames),
            device=selected_device,
            dtype=torch.float32,
            generator=generator,
        )
        event_mask = model_inputs["latent_mask"][:, None, None, :]
        trainer_batch = {
            **dict(batch),
            "model_inputs": model_inputs,
            "latent_lengths": batch["latent_lengths"].to(selected_device),
            "sample_lengths": batch["latent_lengths"].to(selected_device)
            * int(parent_manifest.representation["hop_length"]),
        }
        return GRPOPreparedBatch(
            initial_state=initial_state,
            conditioning=model_inputs,
            trainer_batch=trainer_batch,
            event_mask=event_mask,
        )

    def decode(final_state: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
        audio, sample_lengths = _decode_exact_latent_lengths(
            codec,
            final_state,
            batch["latent_lengths"],
        )
        # The reward wrapper below slices every prompt back to this exact decoded length before
        # invoking any ASR, verifier, or perceptual evaluator.
        batch["sample_lengths"] = sample_lengths
        return audio

    def exact_length_reward(audio: torch.Tensor, batch: Mapping[str, Any]):
        return _evaluate_exact_audio_lengths(
            reward,
            audio,
            batch,
            component_names=tuple(config.reward_weights),
            group_size=config.group_size,
        )

    supervised_loss = None
    if config.supervised_replay_weight > 0:
        flow_loss = FlowMatchingLoss()

        def supervised_loss(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
            inputs = batch["model_inputs"]
            return flow_loss(
                model=model,
                latents=inputs["latents"],
                conditioning_ids=inputs["conditioning_ids"],
                conditioning_mask=inputs.get("conditioning_mask"),
                latent_mask=inputs.get("latent_mask"),
                speaker_latent=inputs.get("speaker_latents"),
                speaker_mask=inputs.get("speaker_mask"),
                language_ids=inputs.get("language_ids"),
                token_durations=inputs.get("token_durations"),
            )

    return GRPOStageRuntime(
        policy=policy,
        reference_policy=reference,
        collate_fn=collator,
        prepare_batch=prepare_batch,
        velocity_adapter=_velocity_adapter,
        decode=decode,
        reward=exact_length_reward,
        supervised_loss=supervised_loss,
        model_export_config=model_export_config_from_manifest(parent_manifest),
        parent_model_manifest=parent_manifest,
    )


def _grpo_post_train(
    config_path: str | os.PathLike[str] = DEFAULT_GRPO_CONFIG_PATH,
    *,
    reward: SpeechRewardFunction,
) -> Path:
    """Run NAR-VAE flow-GRPO from a local, lineage-bearing SFT checkpoint.

    This is the public executable entry function.  It has no implicit evaluator downloads and no
    console-script wrapper.  For multiple GPUs, call it from a short Python module launched with
    ``torchrun``; each process reads the same immutable config and prepared dataset.
    """
    context = initialize_distributed()
    device = None
    device_error: Exception | None = None
    try:
        device = context.device()
    except Exception as exc:
        device_error = exc
    propagate_distributed_error(
        context,
        device_error,
        description="GRPO device initialization",
    )
    assert device is not None

    config = None
    parent_manifest = None
    reference_identity = None
    load_from_disk = None
    startup_error: Exception | None = None
    try:
        try:
            from datasets import load_from_disk as datasets_load_from_disk
        except (ImportError, RuntimeError) as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "GRPO training requires the bounded `nar-vae[train]` stack."
            ) from exc
        load_from_disk = datasets_load_from_disk
        config = load_grpo_stage_config(config_path)
        if not config.parent_checkpoint.is_file():
            raise FileNotFoundError(
                f"GRPO parent checkpoint does not exist: {config.parent_checkpoint}."
            )
        if not config.prompt_dataset_local.is_dir():
            raise FileNotFoundError(
                f"GRPO prompt dataset does not exist: {config.prompt_dataset_local}."
            )
        parent_manifest = validate_grpo_parent_manifest(config.parent_checkpoint)
        reference_identity = grpo_reference_identity(
            config.parent_checkpoint,
            parent_manifest,
        )
    except Exception as exc:
        startup_error = exc
    propagate_distributed_error(context, startup_error, description="GRPO startup validation")
    assert config is not None
    assert parent_manifest is not None
    assert reference_identity is not None
    assert load_from_disk is not None
    node_reference_identity = resolve_node_consistent_value(
        context,
        lambda: reference_identity,
        description="GRPO SFT reference identity",
    )
    identity_error = (
        None
        if reference_identity == node_reference_identity
        else ValueError("GRPO SFT reference identity differs within this node.")
    )
    propagate_distributed_error(
        context,
        identity_error,
        description="GRPO SFT reference identity agreement",
    )
    reference_identity = node_reference_identity

    dataset = None
    dataset_load_error: Exception | None = None
    try:
        dataset = load_from_disk(str(config.prompt_dataset_local))
    except Exception as exc:
        dataset_load_error = exc
    propagate_distributed_error(context, dataset_load_error, description="GRPO dataset loading")
    assert dataset is not None
    dataset_identity = resolve_node_consistent_value(
        context,
        lambda: resolve_local_prepared_dataset_identity(
            dataset,
            config.prompt_dataset_local,
        ),
        description="GRPO prompt dataset identity",
    )
    validation_error: Exception | None = None
    try:
        validate_tts_dataset(
            dataset,
            latent_size=int(parent_manifest.architecture["latent_size"]),
            use_speaker_conditioning=parent_manifest.capabilities["speaker_conditioning"],
            use_language_conditioning=parent_manifest.capabilities["language_conditioning"],
            supported_languages=tuple(parent_manifest.capabilities["supported_languages"]),
            supported_reference_languages=tuple(
                parent_manifest.capabilities["supported_reference_languages"]
            ),
            require_language_coverage=True,
            use_mas_duration=parent_manifest.capabilities["monotonic_alignment"],
            allow_legacy_representation=False,
            expected_codec_source=parent_manifest.representation["codec_source"],
            expected_codec_backend=parent_manifest.representation["codec_backend"],
            expected_codec_revision=parent_manifest.representation["codec_revision"],
            expected_codec_filename=parent_manifest.representation["codec_filename"],
            expected_codec_sha256=parent_manifest.representation["codec_sha256"],
            expected_sample_rate=parent_manifest.representation["sample_rate"],
            expected_hop_length=parent_manifest.representation["hop_length"],
        )
    except Exception as exc:
        validation_error = exc
    propagate_distributed_error(
        context,
        validation_error,
        description="GRPO prompt dataset validation",
    )

    runtime = None
    runtime_error: Exception | None = None
    try:
        runtime = build_nar_vae_grpo_runtime(
            config,
            parent_manifest=parent_manifest,
            reward=reward,
            device=device,
        )
    except Exception as exc:
        runtime_error = exc
    propagate_distributed_error(context, runtime_error, description="GRPO runtime construction")
    assert runtime is not None
    return run_grpo_stage(
        config,
        runtime=runtime,
        dataset=dataset,
        dataset_identity=dataset_identity,
        reference_identity=reference_identity,
        context=context,
        device=device,
    )


def grpo_post_train(
    config_path: str | os.PathLike[str] = DEFAULT_GRPO_CONFIG_PATH,
    *,
    reward: SpeechRewardFunction,
) -> Path:
    """Run GRPO with failure-safe teardown for process groups created by this call."""
    with distributed_cleanup_guard():
        return _grpo_post_train(config_path, reward=reward)


__all__ = [
    "DEFAULT_GRPO_CONFIG_PATH",
    "NARVAEGRPOCollator",
    "SpeechRewardFunction",
    "bind_reward_evaluator_manifest",
    "build_nar_vae_grpo_runtime",
    "grpo_post_train",
    "model_export_config_from_manifest",
]
