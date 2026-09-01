import os
from collections.abc import Mapping
from importlib import import_module

import torch
import torch.distributed as dist
import torch.nn as nn

from vyvotts.checkpoint import load_pretrained_checkpoint
from vyvotts.configuration import (
    bind_training_dataset_identity,
    build_training_argument_overrides,
    initialize_training_run,
    invalidate_training_checkpoint_manifest,
    load_training_lineage,
    load_training_run_manifest,
    resolve_same_run_resume,
    validate_sft_config,
    validate_training_checkpoint_manifest,
    write_training_checkpoint_manifest,
    write_training_lineage,
)
from vyvotts.dataset.data_collator import FlowMatchingDataCollator
from vyvotts.dataset.identity import resolve_local_prepared_dataset_identity
from vyvotts.distributed import (
    distributed_cleanup_guard,
    initialize_distributed,
    propagate_distributed_error,
    propagate_process_group_error,
    require_process_group_consistent_value,
    resolve_node_consistent_value,
    run_distributed_operation,
)
from vyvotts.losses.flow_matching_loss import FlowMatchingLoss
from vyvotts.model_manifest import (
    ModelManifest,
    load_model_manifest,
    validate_manifest_weight,
    validate_sft_parent_manifest,
    validate_sft_resume_manifest,
    write_model_manifest,
)
from vyvotts.model_presets import resolve_model_architecture
from vyvotts.models.flow_matching import create_flow_matching_echodit
from vyvotts.training_data import FrameBudgetTrainerMixin
from vyvotts.training_utils import (
    freeze_layers,
    resolve_duration_training_options,
    resolve_language_training_options,
    resolve_reference_language_training_options,
    resolve_speaker_training_options,
    unwrap_training_model,
    validate_tts_dataset,
)

_TRAINING_IMPORT_ERROR: Exception | None = None
yaml = None
try:
    import yaml
    from datasets import load_from_disk
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
except (ImportError, RuntimeError) as exc:
    # Keep the module importable without the optional training stack. Calling
    # the fine-tuning API still fails early with a useful dependency hint.
    _TRAINING_IMPORT_ERROR = exc
    load_from_disk = None
    TrainingArguments = None
    Trainer = object
    TrainerCallback = object
    set_seed = None


def _require_wandb() -> None:
    """Import the optional reporter only when the selected run requests it."""
    try:
        import_module("wandb")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "W&B reporting was requested but wandb is unavailable. Install "
            "'nar-vae[wandb]' or set report_to: none."
        ) from exc


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that are updated
    with exponential moving average during training.

    Each training process maintains an identical EMA copy. Updates happen only
    after completed optimizer steps through :class:`EMACallback`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = "cpu"):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1.")
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0
        self.last_update_step = 0

        model = unwrap_training_model(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().to(device=device, dtype=torch.float32).clone()

    @torch.no_grad()
    def update(self, model: nn.Module, *, step: int | None = None) -> bool:
        """Update once for a completed optimizer step; return whether it ran."""
        if step is not None and step <= self.last_update_step:
            return False
        model = unwrap_training_model(model)
        parameters = dict(model.named_parameters())
        missing = set(self.shadow) - set(parameters)
        if missing:
            raise RuntimeError(
                "EMA update model does not match the initialized parameter topology. "
                f"Missing: {sorted(missing)}."
            )
        for name, shadow in self.shadow.items():
            param = parameters[name]
            if shadow.device != param.device:
                shadow = shadow.to(param.device)
                self.shadow[name] = shadow
            shadow.mul_(self.decay).add_(
                param.detach().to(dtype=shadow.dtype),
                alpha=1.0 - self.decay,
            )
        self.num_updates += 1
        if step is not None:
            self.last_update_step = step
        return True

    def apply_shadow(self, model: nn.Module):
        """Apply shadow parameters to model (for inference)."""
        model = unwrap_training_model(model)
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    def restore(self, model: nn.Module):
        """Restore original parameters from backup."""
        model = unwrap_training_model(model)
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
            "decay": self.decay,
            "num_updates": self.num_updates,
            "last_update_step": self.last_update_step,
        }

    def load_state_dict(self, state_dict):
        self.shadow = {
            name: value.to(device=self.device, dtype=torch.float32)
            for name, value in state_dict["shadow"].items()
        }
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict.get("num_updates", 0))
        self.last_update_step = int(state_dict.get("last_update_step", 0))


class EMACallback(TrainerCallback):
    """Update EMA after optimizer steps, never during forward/evaluation calls."""

    def __init__(self, ema_model: EMAModel, update_every: int):
        if update_every <= 0:
            raise ValueError("ema_update_every must be positive.")
        self.ema_model = ema_model
        self.update_every = update_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        del args, kwargs
        if model is None or state.global_step <= 0:
            return control
        if state.global_step % self.update_every:
            return control
        self.ema_model.update(model, step=state.global_step)
        return control


class EchoDiTFineTuner(FrameBudgetTrainerMixin, Trainer):
    """
    Trainer for fine-tuning EchoDiT flow matching TTS.

    Features:
    - Pretrained model loading
    - Layer freezing
    - EMA support
    """

    def __init__(
        self,
        config: dict,
        ema_model=None,
        parent_lineage=None,
        parent_model_manifest: ModelManifest | Mapping | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.config = config
        self.training_config = config
        self.flow_model = self.model
        self.ema_model = ema_model
        self.parent_lineage = parent_lineage
        self.parent_model_manifest = parent_model_manifest
        # compute_loss consumes our per-objective accumulation denominators.
        self.model_accepts_loss_kwargs = True

        # Flow matching loss with stratified logit-normal timestep distribution
        self.flow_loss_fn = FlowMatchingLoss(
            sigma_min=config.get("flow_sigma_min", 1e-4),
            velocity_weighted=config.get("flow_velocity_weighted", False),
            timestep_distribution=config.get("timestep_distribution", "stratified_logit_normal"),
            logit_normal_loc=config.get("logit_normal_loc", 0.0),
            logit_normal_scale=config.get("logit_normal_scale", 1.0),
            num_strata=config.get("num_strata", 10),
            duration_loss_weight=config.get("duration_loss_weight", 0.0),
            duration_huber_delta=config.get("duration_huber_delta", 1.0),
            mas_duration_loss_weight=config.get("mas_duration_loss_weight", 0.0),
            mas_alignment_loss_weight=config.get("mas_alignment_loss_weight", 0.0),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute flow matching loss."""
        latents = inputs["latents"]
        conditioning_ids = inputs["conditioning_ids"]
        conditioning_mask = inputs.get("conditioning_mask", None)
        latent_mask = inputs.get("latent_mask", None)
        speaker_latent = inputs.get("speaker_latents", None)
        speaker_mask = inputs.get("speaker_mask", None)
        language_ids = inputs.get("language_ids", None)

        loss = self.flow_loss_fn(
            model=model,
            latents=latents,
            conditioning_ids=conditioning_ids,
            conditioning_mask=conditioning_mask,
            latent_mask=latent_mask,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
            accumulation_normalization=num_items_in_batch,
        )

        return (loss, None) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Save a Trainer-resumable checkpoint plus compatibility/EMA exports."""
        if output_dir is None:
            output_dir = self.args.output_dir

        super().save_model(output_dir, _internal_call=_internal_call)

        if self.is_world_process_zero():
            flow_model_dir = os.path.join(output_dir, "flow_model")
            os.makedirs(flow_model_dir, exist_ok=True)
            # Get the underlying model
            model_to_save = unwrap_training_model(self.flow_model)

            # Save main model
            cpu_state_dict = {
                key: value.detach().cpu() for key, value in model_to_save.state_dict().items()
            }
            torch.save(cpu_state_dict, os.path.join(flow_model_dir, "pytorch_model.bin"))

            # Save EMA model if available
            if self.ema_model is not None:
                ema_state = self.ema_model.state_dict()
                torch.save(ema_state, os.path.join(flow_model_dir, "ema_model.bin"))

                # Also save EMA weights as a separate model for easy loading
                ema_weights = {k: v.cpu() for k, v in self.ema_model.shadow.items()}
                torch.save(ema_weights, os.path.join(flow_model_dir, "pytorch_model_ema.bin"))

            # Save config
            with open(os.path.join(flow_model_dir, "config.yaml"), "w", encoding="utf-8") as file:
                yaml.safe_dump(self.config, file, sort_keys=True)
            write_training_lineage(
                flow_model_dir,
                self.config,
                stage="sft",
                checkpoint_file="pytorch_model.bin",
                parent_lineage=self.parent_lineage,
            )
            checkpoint_files = ["pytorch_model.bin"]
            if self.ema_model is not None:
                checkpoint_files.extend(("ema_model.bin", "pytorch_model_ema.bin"))
            write_model_manifest(
                flow_model_dir,
                self.config,
                stage="sft",
                checkpoint_files=checkpoint_files,
                parent_manifest=self.parent_model_manifest,
            )
            checkpoint_name = os.path.basename(os.path.normpath(output_dir))
            is_trainer_checkpoint = (
                checkpoint_name.startswith("checkpoint-")
                and checkpoint_name[len("checkpoint-") :].isdigit()
            )
            if not is_trainer_checkpoint:
                write_training_checkpoint_manifest(
                    output_dir,
                    self.config,
                    stage="sft",
                    kind="export",
                )

            print(f"NAR-VAE SFT model saved to {flow_model_dir}")

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        """Bind a checkpoint only after Trainer has saved all resume state."""
        distributed = dist.is_available() and dist.is_initialized()
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_dir = os.path.join(run_dir, f"checkpoint-{self.state.global_step}")
        invalidation_error = None
        if self.is_world_process_zero():
            try:
                invalidate_training_checkpoint_manifest(checkpoint_dir)
            except Exception as exc:  # pragma: no cover - distributed propagation path
                invalidation_error = exc
        if distributed:
            errors = [None if invalidation_error is None else repr(invalidation_error)]
            dist.broadcast_object_list(errors, src=0)
            if errors[0] is not None:
                if invalidation_error is not None:
                    raise invalidation_error
                raise RuntimeError(
                    f"Rank zero could not invalidate the old checkpoint seal: {errors[0]}"
                )
            dist.barrier()
        elif invalidation_error is not None:
            raise invalidation_error

        result = None
        checkpoint_error = None
        try:
            result = super()._save_checkpoint(model, trial, *args, **kwargs)
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            checkpoint_error = exc
        propagate_process_group_error(
            checkpoint_error,
            description="Trainer SFT checkpoint save",
        )
        if distributed:
            # Every rank writes its own RNG state. Seal only the complete,
            # shared artifact set so later DDP recovery is deterministic.
            dist.barrier()
        manifest_error = None
        if self.is_world_process_zero():
            try:
                write_training_checkpoint_manifest(
                    checkpoint_dir,
                    self.config,
                    stage="sft",
                    kind="trainer_checkpoint",
                )
            except Exception as exc:  # pragma: no cover - distributed propagation path
                manifest_error = exc
        if distributed:
            errors = [None if manifest_error is None else repr(manifest_error)]
            dist.broadcast_object_list(errors, src=0)
            if errors[0] is not None:
                if manifest_error is not None:
                    raise manifest_error
                raise RuntimeError(f"Rank zero could not bind the Trainer checkpoint: {errors[0]}")
        elif manifest_error is not None:
            raise manifest_error
        return result

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Restore model and EMA state from the same Trainer checkpoint."""
        checkpoint_manifest = None
        validation_error = None
        try:
            checkpoint_manifest = validate_training_checkpoint_manifest(
                resume_from_checkpoint,
                self.config,
                stage="sft",
                expected_kind="trainer_checkpoint",
            )
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            validation_error = exc
        propagate_process_group_error(
            validation_error,
            description="SFT resume-checkpoint validation",
        )
        require_process_group_consistent_value(
            checkpoint_manifest,
            description="SFT resume-checkpoint identity",
        )

        result = None
        load_error = None
        try:
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            load_error = exc
        propagate_process_group_error(
            load_error,
            description="SFT resume-checkpoint deserialization",
        )

        metadata_error = None
        resumed_lineage = None
        resumed_model_manifest = None
        try:
            resumed_flow_dir = os.path.join(resume_from_checkpoint, "flow_model")
            resumed_lineage = load_training_lineage(resumed_flow_dir)
            resumed_model_manifest = load_model_manifest(
                os.path.join(resumed_flow_dir, "nar_vae_manifest.json")
            )
            validate_manifest_weight(
                resumed_model_manifest,
                os.path.join(resumed_flow_dir, "pytorch_model.bin"),
            )
            validate_sft_resume_manifest(resumed_model_manifest, self.config)
            if self.ema_model is not None:
                ema_path = os.path.join(resumed_flow_dir, "ema_model.bin")
                if not os.path.isfile(ema_path):
                    raise FileNotFoundError(
                        "EMA is enabled, but its state is missing from the resume checkpoint: "
                        f"{ema_path}"
                    )
                ema_state = torch.load(ema_path, map_location="cpu", weights_only=True)
                self.ema_model.load_state_dict(ema_state)
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            metadata_error = exc
        propagate_process_group_error(
            metadata_error,
            description="SFT resume metadata restoration",
        )
        assert resumed_lineage is not None and resumed_model_manifest is not None
        require_process_group_consistent_value(
            {
                "lineage": resumed_lineage,
                "model_manifest_sha256": resumed_model_manifest.sha256,
            },
            description="SFT resumed model identity",
        )
        # Preserve the original pretraining parent across any number of SFT
        # interruption/recovery cycles instead of replacing it with an SFT child.
        self.parent_lineage = resumed_lineage["parent"]
        self.parent_model_manifest = resumed_model_manifest.parent
        return result


DEFAULT_FINETUNE_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "configs/finetune_config.yaml",
)


def _preserve_flow_only_trainability(model, freeze_info: dict[str, float | int]):
    """Keep legacy latent-prefix weights frozen after generic layer selection."""
    setter = getattr(getattr(model, "dit", None), "set_latent_prefix_trainable", None)
    if callable(setter):
        setter(False)
        frozen = sum(
            parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
        )
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total = frozen + trainable
        return {
            "frozen_params": frozen,
            "trainable_params": trainable,
            "frozen_ratio": frozen / total if total else 0.0,
        }
    return freeze_info


def _freeze_sft_layers(model, config: Mapping) -> dict[str, float | int]:
    freeze_info = freeze_layers(model, config)
    return _preserve_flow_only_trainability(model, freeze_info)


def _configure_sft_wandb_environment(config: Mapping, *, enabled: bool) -> None:
    if not enabled:
        return
    os.environ["WANDB_PROJECT"] = config.get(
        "wandb_project",
        config.get("project_name", "nar-vae-sft"),
    )
    os.environ["WANDB_RUN_NAME"] = config.get(
        "wandb_run_name",
        config.get("run_name", "nar-vae-sft"),
    )
    os.environ["WANDB_LOG_MODEL"] = str(config.get("wandb_log_model", False)).lower()


def _finetune(
    config_path: str | os.PathLike[str] = DEFAULT_FINETUNE_CONFIG_PATH,
) -> None:
    """Run supervised fine-tuning from a pretrained or same-run NAR-VAE checkpoint."""
    if _TRAINING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Fine-tuning dependencies are unavailable. Install the bounded training "
            "stack with `pip install 'nar-vae[train]'`."
        ) from _TRAINING_IMPORT_ERROR

    # Select LOCAL_RANK before reading node-local model/data paths. Propagate a
    # startup failure collectively so peers never enter a later DDP operation.
    process = initialize_distributed()
    startup_error = None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        parent_lineage = validate_sft_config(config)
        training_overrides = build_training_argument_overrides(config)
        if "wandb" in training_overrides["report_to"]:
            _require_wandb()

        tts_dataset_local = config.get("TTS_dataset_local")
        if not tts_dataset_local:
            raise ValueError("TTS_dataset_local must be specified in config")
        if not os.path.exists(tts_dataset_local):
            raise FileNotFoundError(
                f"Dataset not found: {tts_dataset_local}\n"
                "Create it with "
                "`nar_vae.dataset.prepare_dataset.prepare_from_local_folder("
                f"input_dir='path/to/audio', output_dir={tts_dataset_local!r})`."
            )
    except Exception as exc:  # pragma: no cover - distributed startup failure
        startup_error = exc
        config = None
        parent_lineage = None
        training_overrides = None
        tts_dataset_local = None
    propagate_distributed_error(process, startup_error, description="SFT startup")
    assert config is not None and training_overrides is not None and tts_dataset_local is not None

    local_rank = process.local_rank
    device = process.device()
    is_main = process.is_main_process

    if is_main:
        print(f"\nLoading prepared dataset from: {tts_dataset_local}")
    ds_tts = None
    dataset_load_error = None
    try:
        ds_tts = load_from_disk(tts_dataset_local)
    except Exception as exc:  # pragma: no cover - distributed filesystem failure
        dataset_load_error = exc
    propagate_distributed_error(process, dataset_load_error, description="SFT dataset loading")
    dataset_identity = resolve_node_consistent_value(
        process,
        lambda: resolve_local_prepared_dataset_identity(ds_tts, tts_dataset_local),
        description="SFT dataset identity",
    )
    bind_training_dataset_identity(config, dataset_identity)

    if parent_lineage is not None:
        resolve_node_consistent_value(
            process,
            lambda: parent_lineage,
            description="SFT pretraining lineage",
        )

    resume_payload = [None, None]
    initialization_error = None
    if is_main:
        try:
            initialize_training_run(
                config,
                stage="sft",
                is_world_process_zero=True,
            )
            resume_payload[0] = resolve_same_run_resume(config)
        except Exception as exc:  # pragma: no cover - distributed propagation path
            initialization_error = exc
            resume_payload[1] = repr(exc)
    if process.is_distributed:
        dist.broadcast_object_list(resume_payload, src=0)
    if resume_payload[1] is not None:
        if initialization_error is not None:
            raise initialization_error
        raise RuntimeError(f"Rank zero could not initialize the SFT run: {resume_payload[1]}")

    run_manifest = None
    run_manifest_error = None
    try:
        run_manifest = load_training_run_manifest(config, stage="sft")
    except Exception as exc:  # pragma: no cover - distributed filesystem failure
        run_manifest_error = exc
    propagate_distributed_error(
        process,
        run_manifest_error,
        description="SFT run-manifest loading",
    )
    assert run_manifest is not None
    resolve_node_consistent_value(
        process,
        lambda: run_manifest,
        description="SFT run-manifest identity",
    )

    resume_from_checkpoint = resume_payload[0]
    parent_model_manifest = None
    if resume_from_checkpoint is None:
        parent_manifest_error = None
        try:
            parent_model_manifest = validate_sft_parent_manifest(
                config["pretrained_checkpoint"],
                config,
            )
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            parent_manifest_error = exc
        propagate_distributed_error(
            process,
            parent_manifest_error,
            description="SFT parent-model validation",
        )
        assert parent_model_manifest is not None
        parent_identity = {
            "manifest_sha256": parent_model_manifest.sha256,
            "weights": dict(parent_model_manifest.weights),
            "representation": dict(parent_model_manifest.representation),
        }
        resolve_node_consistent_value(
            process,
            lambda: parent_identity,
            description="SFT parent-model identity",
        )

    if is_main:
        print("=" * 80)
        print("NAR-VAE supervised fine-tuning")
        print("=" * 80)
        print(f"World size: {process.world_size}")

    print(f"[Rank {local_rank}] Device: {device}")

    # Create model on correct device from the start
    if is_main:
        print("\nCreating the NAR-VAE acoustic model...")

    # Create model on CPU first, then move to the selected device.
    pretrained_path = config.get("pretrained_checkpoint")
    checkpoint_context = pretrained_path or (
        str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    run_distributed_operation(
        process,
        lambda: set_seed(config.get("seed", 1337)),
        description="SFT random-seed initialization",
    )
    (
        use_speaker_conditioning,
        initialize_speaker_conditioning,
        speaker_patch_size,
    ) = run_distributed_operation(
        process,
        lambda: resolve_speaker_training_options(
            config,
            pretrained_checkpoint=checkpoint_context,
        ),
        description="SFT speaker-topology resolution",
    )
    duration_options = run_distributed_operation(
        process,
        lambda: resolve_duration_training_options(
            config,
            use_speaker_conditioning=use_speaker_conditioning,
            pretrained_checkpoint=checkpoint_context,
        ),
        description="SFT duration-topology resolution",
    )
    (
        use_language_conditioning,
        initialize_language_conditioning,
        supported_languages,
    ) = run_distributed_operation(
        process,
        lambda: resolve_language_training_options(
            config,
            pretrained_checkpoint=checkpoint_context,
        ),
        description="SFT language-topology resolution",
    )
    (
        supported_reference_languages,
        initialize_cross_lingual_capability,
    ) = run_distributed_operation(
        process,
        lambda: resolve_reference_language_training_options(
            config,
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=use_language_conditioning,
            pretrained_checkpoint=checkpoint_context,
        ),
        description="SFT reference-language resolution",
    )
    architecture = run_distributed_operation(
        process,
        lambda: resolve_model_architecture(config),
        description="SFT architecture resolution",
    )
    flow_model = run_distributed_operation(
        process,
        lambda: create_flow_matching_echodit(
            latent_size=config["dacvae_latent_dim"],
            text_vocab_size=config["text_vocab_size"],
            speaker_patch_size=speaker_patch_size,
            **architecture.model_kwargs(),
            norm_eps=config.get("norm_eps", 1e-6),
            cfg_dropout=config.get("cfg_dropout", 0.1),
            cfg_dropout_text=config.get("cfg_dropout_text"),
            cfg_dropout_speaker=config.get("cfg_dropout_speaker"),
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=use_language_conditioning,
            supported_languages=supported_languages,
            supported_reference_languages=supported_reference_languages,
            use_duration_predictor=duration_options.enabled,
            duration_predictor_hidden_size=duration_options.hidden_size,
            duration_predictor_num_layers=duration_options.num_layers,
            duration_predictor_use_speaker=duration_options.uses_speaker,
            use_mas_duration=duration_options.uses_mas,
            duration_alignment_hidden_size=duration_options.alignment_hidden_size,
        ),
        description="SFT model construction",
    )
    if is_main:
        print(f"Speaker conditioning: {'ENABLED' if use_speaker_conditioning else 'DISABLED'}")
        print(
            "Language conditioning: "
            f"{'ENABLED' if use_language_conditioning else 'DISABLED'} "
            f"({', '.join(supported_languages)})"
        )
        print(f"Learned duration: {'ENABLED' if duration_options.enabled else 'DISABLED'}")

    # A same-run resume is restored by Trainer, including optimizer/scheduler/RNG.
    # Only a fresh SFT run loads the explicitly selected pretrained model here.
    if resume_from_checkpoint:
        if is_main:
            print(f"Resuming the existing SFT run: {resume_from_checkpoint}")
    elif pretrained_path:
        print(f"Loading pretrained checkpoint: {pretrained_path}")
        run_distributed_operation(
            process,
            lambda: load_pretrained_checkpoint(
                flow_model,
                pretrained_path,
                strict=not initialize_speaker_conditioning,
                initialize_speaker_conditioning=initialize_speaker_conditioning,
                initialize_language_conditioning=initialize_language_conditioning,
                initialize_cross_lingual_capability=initialize_cross_lingual_capability,
                initialize_duration_predictor=duration_options.initialize,
                preload_validator=lambda selected_path: validate_manifest_weight(
                    parent_model_manifest,
                    selected_path,
                ),
            ),
            description="SFT parent-checkpoint deserialization",
        )
        if is_main:
            print("Pretrained checkpoint loaded successfully!")
    # Move model to correct GPU AFTER loading checkpoint
    flow_model = run_distributed_operation(
        process,
        lambda: flow_model.to(device),
        description="SFT model device placement",
    )
    print(f"[Rank {local_rank}] Model moved to {device}")

    # Freeze layers
    if is_main:
        print("\nApplying layer freezing...")
    freeze_info = run_distributed_operation(
        process,
        lambda: _freeze_sft_layers(flow_model, config),
        description="SFT layer freezing",
    )
    params_flow = run_distributed_operation(
        process,
        flow_model.get_num_params,
        description="SFT model topology inspection",
    )
    if is_main:
        print(f"  Frozen parameters: {freeze_info['frozen_params'] / 1e6:.2f}M")
        print(f"  Trainable parameters: {freeze_info['trainable_params'] / 1e6:.2f}M")
        print(f"  Frozen ratio: {freeze_info['frozen_ratio']:.1%}")

        # Print model info
        print("\nNAR-VAE model parameters:")
        for k, v in params_flow.items():
            print(f"  {k}: {v / 1e6:.2f}M")

    # Initialize EMA if enabled (once per training process).
    ema_model = None
    if config.get("use_ema", False):
        if is_main:
            print("\nInitializing optimizer-step EMA...")
        ema_model = run_distributed_operation(
            process,
            lambda: EMAModel(
                flow_model,
                decay=config.get("ema_decay", 0.9999),
                device=str(device),
            ),
            description="SFT EMA initialization",
        )
        if is_main:
            print(f"  EMA decay: {ema_model.decay}")

    run_distributed_operation(
        process,
        lambda: validate_tts_dataset(
            ds_tts,
            latent_size=config["dacvae_latent_dim"],
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=use_language_conditioning,
            supported_languages=supported_languages,
            supported_reference_languages=supported_reference_languages,
            require_language_coverage=True,
            use_mas_duration=duration_options.uses_mas,
            allow_legacy_representation=config.get("allow_legacy_representation", False),
            expected_codec_source=config.get("dacvae_model"),
            expected_codec_backend=config.get("dacvae_backend"),
            expected_codec_revision=config.get("dacvae_revision"),
            expected_codec_filename=config.get("dacvae_filename"),
            expected_codec_sha256=config.get("dacvae_sha256"),
            expected_sample_rate=config.get("dacvae_sample_rate"),
            expected_hop_length=config.get("dacvae_hop_length"),
        ),
        description="SFT dataset preflight",
    )

    if is_main:
        print(f"Dataset size: {len(ds_tts)} samples")

    # Data collator
    data_collator = run_distributed_operation(
        process,
        lambda: FlowMatchingDataCollator(
            pad_token=config["pad_token"],
            speaker_patch_size=speaker_patch_size,
        ),
        description="SFT data-collator construction",
    )

    # Trainer's W&B integration is optional and owns rank-zero-safe logging.
    run_distributed_operation(
        process,
        lambda: _configure_sft_wandb_environment(
            config,
            enabled=is_main and "wandb" in training_overrides["report_to"],
        ),
        description="SFT W&B environment setup",
    )

    # Training arguments
    training_args = run_distributed_operation(
        process,
        lambda: TrainingArguments(
            output_dir=config["save_folder"],
            num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 1000),
            remove_unused_columns=False,
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            dataloader_num_workers=config.get("dataloader_num_workers", 4),
            dataloader_pin_memory=config.get("dataloader_pin_memory", True),
            dataloader_drop_last=config.get("dataloader_drop_last", True),
            save_total_limit=config.get("save_total_limit", 3),
            local_rank=process.trainer_local_rank,
            **training_overrides,
        ),
        description="SFT TrainingArguments construction",
    )

    # Initialize trainer
    callbacks = run_distributed_operation(
        process,
        lambda: (
            [EMACallback(ema_model, config.get("ema_update_every", 10))]
            if ema_model is not None
            else None
        ),
        description="SFT callback construction",
    )
    trainer = run_distributed_operation(
        process,
        lambda: EchoDiTFineTuner(
            config=config,
            model=flow_model,
            ema_model=ema_model,
            parent_lineage=parent_lineage,
            parent_model_manifest=parent_model_manifest,
            args=training_args,
            train_dataset=ds_tts,
            data_collator=data_collator,
            callbacks=callbacks,
        ),
        description="SFT Trainer construction",
    )

    # Train
    if is_main:
        print("\n" + "=" * 80)
        print("Starting NAR-VAE supervised fine-tuning...")
        print("=" * 80)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    if is_main:
        print("\nSaving final model...")
    final_save_error = None
    try:
        trainer.save_model(os.path.join(config["save_folder"], "final"))
    except Exception as exc:  # pragma: no cover - distributed filesystem failure
        final_save_error = exc
    propagate_distributed_error(process, final_save_error, description="final SFT export")

    if is_main:
        print("\n" + "=" * 80)
        print("NAR-VAE supervised fine-tuning complete!")
        print("=" * 80)


def finetune(
    config_path: str | os.PathLike[str] = DEFAULT_FINETUNE_CONFIG_PATH,
) -> None:
    """Run SFT with failure-safe teardown for process groups created here."""
    with distributed_cleanup_guard():
        _finetune(config_path)
