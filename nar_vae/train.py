import os
import re
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from nar_vae.configuration import (
    bind_training_dataset_identity,
    build_training_argument_overrides,
    initialize_training_run,
    invalidate_training_checkpoint_manifest,
    load_training_run_manifest,
    resolve_same_run_resume,
    validate_pretraining_config,
    validate_training_checkpoint_manifest,
    write_training_checkpoint_manifest,
    write_training_lineage,
)
from nar_vae.dataset.data_collator import FlowMatchingDataCollator
from nar_vae.dataset.identity import (
    PREPARED_DATASET_MANIFEST_FILENAME,
    DatasetIdentityError,
    resolve_hub_dataset_identity,
    resolve_local_prepared_dataset_identity,
)
from nar_vae.dataset.utterance_store import DynamicReferenceDataset
from nar_vae.distributed import (
    distributed_cleanup_guard,
    initialize_distributed,
    propagate_distributed_error,
    propagate_process_group_error,
    require_process_group_consistent_value,
    resolve_node_consistent_value,
    run_distributed_operation,
)
from nar_vae.frozen_text_provider import FrozenTextProviderSpec
from nar_vae.losses.flow_matching_loss import FlowMatchingLoss
from nar_vae.model_manifest import representation_from_config, write_model_manifest
from nar_vae.model_presets import resolve_model_architecture
from nar_vae.models.flow_matching import create_flow_matching_echodit
from nar_vae.training_data import FrameBudgetTrainerMixin
from nar_vae.training_optimizers import MuonTrainerMixin
from nar_vae.training_utils import (
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
    from transformers import Trainer, TrainingArguments, set_seed
except (ImportError, RuntimeError) as exc:
    # Keep the module importable so dependency failures remain actionable.
    _TRAINING_IMPORT_ERROR = exc
    load_from_disk = None
    TrainingArguments = None
    Trainer = object
    set_seed = None


def _require_wandb() -> None:
    """Import the mandatory reporter lazily when pretraining starts."""
    try:
        import_module("wandb")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "W&B is required for pretraining but wandb is unavailable. "
            "Install the complete nar-vae package before starting training."
        ) from exc


def _materialize_flow_model_weights(
    output_dir: str | os.PathLike[str],
    flow_model_dir: str | os.PathLike[str],
    model,
) -> Path:
    """Create the inference weight path without duplicating Trainer serialization."""
    trainer_weights = Path(output_dir) / "pytorch_model.bin"
    flow_weights = Path(flow_model_dir) / "pytorch_model.bin"
    if flow_weights.exists() or flow_weights.is_symlink():
        flow_weights.unlink()
    if trainer_weights.is_file():
        try:
            os.link(trainer_weights, flow_weights)
        except OSError:
            shutil.copyfile(trainer_weights, flow_weights)
        return flow_weights

    # Safe fallback for explicitly requested safetensors or Trainer backends
    # whose root artifact cannot be consumed by the inference checkpoint loader.
    model_to_save = unwrap_training_model(model)
    cpu_state_dict = {
        key: value.detach().cpu() for key, value in model_to_save.state_dict().items()
    }
    torch.save(cpu_state_dict, flow_weights)
    return flow_weights


class EchoDiTTrainer(MuonTrainerMixin, FrameBudgetTrainerMixin, Trainer):
    """
    Trainer for EchoDiT flow matching TTS.

    Handles:
    - TTS training with flow matching loss
    - DDP distributed training through torchrun/Transformers Trainer
    - Gradient checkpointing for memory efficiency
    """

    def __init__(
        self,
        training_config: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if training_config is None:
            raise ValueError("training_config is required")

        self.flow_model = self.model  # Get model from parent Trainer
        self.training_config = training_config
        # compute_loss consumes our per-objective accumulation denominators.
        # This flag prevents Trainer from dividing the already window-normalized
        # loss by gradient_accumulation_steps a second time.
        self.model_accepts_loss_kwargs = True

        # Flow matching loss with stratified logit-normal timestep distribution
        self.flow_loss_fn = FlowMatchingLoss(
            sigma_min=training_config.get("flow_sigma_min", 1e-4),
            generative_objective=training_config.get("generative_objective", "rectified_flow"),
            diffusion_schedule_shift=training_config.get("diffusion_schedule_shift", 1.0),
            velocity_weighted=training_config.get("flow_velocity_weighted", False),
            timestep_distribution=training_config.get(
                "timestep_distribution",
                "stratified_logit_normal",
            ),
            logit_normal_loc=training_config.get("logit_normal_loc", 0.0),
            logit_normal_scale=training_config.get("logit_normal_scale", 1.0),
            num_strata=training_config.get("num_strata", 10),
            duration_loss_weight=training_config.get("duration_loss_weight", 0.0),
            duration_huber_delta=training_config.get("duration_huber_delta", 1.0),
            mas_duration_loss_weight=training_config.get("mas_duration_loss_weight", 0.0),
            mas_alignment_loss_weight=training_config.get("mas_alignment_loss_weight", 0.0),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute flow matching loss for TTS.

        Args:
            model: Model passed by Trainer, including any distributed wrapper
            inputs: Batch inputs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (for loss scaling)
        """
        # Flow matching loss
        latents = inputs["latents"]
        conditioning_ids = inputs["conditioning_ids"]
        conditioning_mask = inputs.get("conditioning_mask", None)
        latent_mask = inputs.get("latent_mask", None)
        speaker_latent = inputs.get("speaker_latents", None)
        speaker_mask = inputs.get("speaker_mask", None)
        language_ids = inputs.get("language_ids", None)
        token_language_ids = inputs.get("token_language_ids", None)
        alignment_mask = inputs.get("alignment_mask", None)
        conditioning_features = inputs.get("conditioning_features", None)

        loss = self.flow_loss_fn(
            model=model,
            latents=latents,
            conditioning_ids=conditioning_ids,
            conditioning_mask=conditioning_mask,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            conditioning_features=conditioning_features,
            latent_mask=latent_mask,
            speaker_latent=speaker_latent,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
            accumulation_normalization=num_items_in_batch,
        )

        return (loss, None) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Save a Trainer-resumable checkpoint and the legacy flow-model export."""
        if output_dir is None:
            output_dir = self.args.output_dir

        # Trainer owns the root model artifact used by resume_from_checkpoint.
        super().save_model(output_dir, _internal_call=_internal_call)

        if self.is_world_process_zero():
            flow_model_dir = os.path.join(output_dir, "flow_model")
            os.makedirs(flow_model_dir, exist_ok=True)
            # The pretraining config defaults to the same PyTorch state-dict
            # format required by inference. A hard link avoids serializing and
            # storing an identical multi-GB checkpoint twice while retaining
            # both paths expected by exact Trainer resume and model manifests.
            _materialize_flow_model_weights(output_dir, flow_model_dir, self.flow_model)
            with open(os.path.join(flow_model_dir, "config.yaml"), "w", encoding="utf-8") as file:
                yaml.safe_dump(self.training_config, file, sort_keys=True)
            write_training_lineage(
                flow_model_dir,
                self.training_config,
                stage="pretrain",
                checkpoint_file="pytorch_model.bin",
            )
            write_model_manifest(
                flow_model_dir,
                self.training_config,
                stage="pretrain",
                checkpoint_files=("pytorch_model.bin",),
            )
            checkpoint_name = Path(output_dir).name
            is_trainer_checkpoint = (
                checkpoint_name.startswith("checkpoint-")
                and checkpoint_name[len("checkpoint-") :].isdigit()
            )
            if not is_trainer_checkpoint:
                write_training_checkpoint_manifest(
                    output_dir,
                    self.training_config,
                    stage="pretrain",
                    kind="export",
                )
            print(f"NAR-VAE model saved to {flow_model_dir}")

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
            description="Trainer pretraining checkpoint save",
        )
        if distributed:
            # Every rank writes its own RNG state. Do not seal the artifact set
            # until all of those files are visible on the shared filesystem.
            dist.barrier()
        manifest_error = None
        if self.is_world_process_zero():
            try:
                write_training_checkpoint_manifest(
                    checkpoint_dir,
                    self.training_config,
                    stage="pretrain",
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
        """Revalidate the artifact binding immediately before Trainer loads it."""
        checkpoint_manifest = None
        validation_error = None
        try:
            checkpoint_manifest = validate_training_checkpoint_manifest(
                resume_from_checkpoint,
                self.training_config,
                stage="pretrain",
                expected_kind="trainer_checkpoint",
            )
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            validation_error = exc
        propagate_process_group_error(
            validation_error,
            description="pretraining resume-checkpoint validation",
        )
        require_process_group_consistent_value(
            checkpoint_manifest,
            description="pretraining resume-checkpoint identity",
        )

        result = None
        load_error = None
        try:
            result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            load_error = exc
        propagate_process_group_error(
            load_error,
            description="pretraining resume-checkpoint deserialization",
        )
        return result


DEFAULT_TRAIN_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "configs/pretrain_config.yaml",
)
_MAX_DATASET_DOWNLOAD_WORKERS = 32
_HUB_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class _PretrainingDatasetSource:
    kind: str
    location: str
    revision: str | None
    download_workers: int


def _resolve_pretraining_dataset_source(config: dict) -> _PretrainingDatasetSource:
    """Resolve one authoritative local or reproducibly pinned remote dataset."""
    download_workers = config.get("dataset_download_workers", 8)
    if (
        isinstance(download_workers, bool)
        or not isinstance(download_workers, int)
        or not 1 <= download_workers <= _MAX_DATASET_DOWNLOAD_WORKERS
    ):
        raise ValueError(
            "dataset_download_workers must be an integer between "
            f"1 and {_MAX_DATASET_DOWNLOAD_WORKERS}."
        )

    configured_local = config.get("TTS_dataset_local")
    if not isinstance(configured_local, (str, os.PathLike, type(None))):
        raise ValueError("TTS_dataset_local must be a filesystem path, empty, or null.")
    if isinstance(configured_local, str) and not configured_local.strip():
        configured_local = None
    if configured_local is not None:
        local_path = Path(configured_local).expanduser()
        if not local_path.exists():
            raise FileNotFoundError(
                "TTS_dataset_local is configured but does not exist: "
                f"{local_path}. Set it to null to select a remote dataset explicitly."
            )
        return _PretrainingDatasetSource(
            kind="local",
            location=str(local_path),
            revision=None,
            download_workers=download_workers,
        )

    repo_id = config.get("TTS_dataset")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("Remote pretraining requires a non-empty TTS_dataset repo id.")
    repo_id = repo_id.strip()
    repo_parts = repo_id.split("/")
    if repo_id.casefold() == "owner/dataset" or len(repo_parts) != 2 or not all(repo_parts):
        raise ValueError(
            "Remote pretraining requires a non-placeholder TTS_dataset repo id "
            "in 'namespace/name' form."
        )

    revision = config.get("TTS_dataset_revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(
            "Remote pretraining requires a full 40-character TTS_dataset_revision commit."
        )
    revision = revision.strip()
    if not _HUB_COMMIT_PATTERN.fullmatch(revision):
        raise ValueError(
            "TTS_dataset_revision must be a full 40-character Hub commit SHA; mutable branches "
            "and tags are not reproducible pretraining sources."
        )
    return _PretrainingDatasetSource(
        kind="remote",
        location=repo_id,
        revision=revision,
        download_workers=download_workers,
    )


def _load_pretraining_dataset(source: _PretrainingDatasetSource, process):
    """Load only byte-manifested prepared data from local disk or a pinned Hub snapshot."""
    if source.kind == "local":
        print(f"Loading TTS dataset from local path: {source.location}")
        dataset = None
        load_error = None
        try:
            dataset = load_from_disk(source.location)
        except Exception as exc:  # pragma: no cover - distributed filesystem failure
            load_error = exc
        propagate_distributed_error(process, load_error, description="local dataset loading")
        return dataset, Path(source.location).expanduser().resolve()

    snapshot_path = None
    # Hub caches are node-local in a conventional multi-node torchrun. Have
    # each node leader materialize the exact commit, then share only that
    # leader's local path with ranks on the same node.
    if not process.is_distributed or process.local_rank == 0:
        try:
            print("Downloading TTS dataset with snapshot_download...")
            snapshot_path = snapshot_download(
                repo_id=source.location,
                repo_type="dataset",
                revision=source.revision,
                max_workers=source.download_workers,
            )
        except Exception as exc:  # pragma: no cover - distributed network/cache failure
            download_error = exc
        else:
            download_error = None
    else:
        download_error = None
    propagate_distributed_error(process, download_error, description="pinned dataset download")
    if process.is_distributed:
        snapshot_paths = [None] * process.world_size
        dist.all_gather_object(snapshot_paths, snapshot_path)
        node_leader_rank = process.rank - process.local_rank
        snapshot_path = snapshot_paths[node_leader_rank]
    if not isinstance(snapshot_path, (str, os.PathLike)):
        raise RuntimeError("The node leader did not resolve the pinned dataset snapshot path.")
    snapshot = Path(snapshot_path).expanduser().resolve()
    dataset = None
    load_error = None
    try:
        manifest_path = snapshot / PREPARED_DATASET_MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise DatasetIdentityError(
                "Remote pretraining accepts only commit-contained Dataset.save_to_disk artifacts "
                f"with {PREPARED_DATASET_MANIFEST_FILENAME}; missing: {manifest_path}. Dataset "
                "scripts that may fetch mutable external URLs are not accepted."
            )
        print(f"Loading prepared TTS snapshot: {source.location}@{source.revision}")
        dataset = load_from_disk(str(snapshot))
    except Exception as exc:  # pragma: no cover - distributed filesystem failure
        load_error = exc
    propagate_distributed_error(process, load_error, description="pinned dataset loading")
    return dataset, snapshot


def _resolve_pretraining_dataset_identity(
    dataset,
    source: _PretrainingDatasetSource,
    prepared_path: Path,
) -> dict:
    """Bind the exact local artifacts or pinned Hub split loaded for this run."""
    if source.kind == "local":
        return resolve_local_prepared_dataset_identity(dataset, source.location)
    return resolve_hub_dataset_identity(
        dataset,
        repo_id=source.location,
        revision=source.revision,
        split="train",
        snapshot_dir=prepared_path,
    )


def _resolve_distributed_pretraining_dataset_identity(
    dataset,
    source: _PretrainingDatasetSource,
    prepared_path: Path,
    process,
) -> dict:
    """Hash once per node and require identical prepared bytes across nodes."""
    return resolve_node_consistent_value(
        process,
        lambda: _resolve_pretraining_dataset_identity(
            dataset,
            source,
            prepared_path,
        ),
        description="pretraining dataset identity",
    )


def _load_pretraining_yaml(config_path: str | os.PathLike[str]) -> dict:
    """Load a YAML config and one optional relative compatibility base."""
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Pretraining configuration must be a YAML mapping.")

    extends = config.pop("extends", None)
    if extends is None:
        return config
    if not isinstance(extends, str) or not extends.strip():
        raise ValueError("extends must name one YAML file relative to the selected config.")
    base_path = (path.parent / extends).resolve()
    with base_path.open(encoding="utf-8") as file:
        base = yaml.safe_load(file)
    if not isinstance(base, dict) or "extends" in base:
        raise ValueError(
            "A pretraining base must be a YAML mapping and cannot extend another file."
        )
    base.update(config)
    return base


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


def _freeze_pretraining_layers(model, config: dict) -> dict[str, float | int]:
    freeze_info = freeze_layers(model, config)
    return _preserve_flow_only_trainability(model, freeze_info)


def _configure_pretraining_wandb_environment(config: dict, *, enabled: bool) -> None:
    if not enabled:
        return
    os.environ["WANDB_PROJECT"] = config.get(
        "wandb_project",
        config.get("project_name", "nar-vae-pretraining"),
    )
    os.environ["WANDB_RUN_NAME"] = config.get(
        "wandb_run_name",
        config.get("run_name", "nar-vae-pretraining"),
    )
    os.environ["WANDB_LOG_MODEL"] = str(config.get("wandb_log_model", False)).lower()


def _pretrain(
    config_path: str | os.PathLike[str] = DEFAULT_TRAIN_CONFIG_PATH,
) -> None:
    """Pretrain a randomly initialized NAR-VAE acoustic model."""
    if _TRAINING_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training dependencies are unavailable. Reinstall the single package with "
            "`python -m pip install -e .`."
        ) from _TRAINING_IMPORT_ERROR

    # Establish the group before touching node-local config, parent, codec, or
    # dataset paths so one host cannot fail while peers advance into a later
    # collective. No model/CUDA allocation beyond device selection happens yet.
    process = initialize_distributed()
    startup_error = None
    try:
        config = _load_pretraining_yaml(config_path)
        validate_pretraining_config(config)
        representation_from_config(config)
        training_overrides = build_training_argument_overrides(config)
        _require_wandb()
        dataset_source = _resolve_pretraining_dataset_source(config)
    except Exception as exc:  # pragma: no cover - distributed startup failure
        startup_error = exc
        config = None
        training_overrides = None
        dataset_source = None
    propagate_distributed_error(process, startup_error, description="pretraining startup")
    assert config is not None and training_overrides is not None and dataset_source is not None

    save_folder = config["save_folder"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    save_steps = config["save_steps"]
    pad_token = config["pad_token"]
    learning_rate = config["learning_rate"]

    print("=" * 80)
    print("NAR-VAE flow-matching TTS pretraining")
    print("=" * 80)

    device = process.device()

    # Resolve data before creating the immutable run identity. Local prepared
    # artifacts are hash-checked; Hub data binds both its commit and Dataset fingerprint.
    ds_tts, prepared_dataset_path = _load_pretraining_dataset(dataset_source, process)
    dataset_identity = _resolve_distributed_pretraining_dataset_identity(
        ds_tts,
        dataset_source,
        prepared_dataset_path,
        process,
    )
    bind_training_dataset_identity(config, dataset_identity)

    resume_payload = [None, None]
    initialization_error = None
    if process.is_main_process:
        try:
            initialize_training_run(
                config,
                stage="pretrain",
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
        raise RuntimeError(f"Rank zero could not initialize the training run: {resume_payload[1]}")
    run_manifest = run_distributed_operation(
        process,
        lambda: load_training_run_manifest(config, stage="pretrain"),
        description="pretraining run-manifest loading",
    )
    resolve_node_consistent_value(
        process,
        lambda: run_manifest,
        description="pretraining run-manifest identity",
    )
    resume_from_checkpoint = resume_payload[0]

    if process.is_main_process:
        print(f"Device: {device}")
        print(f"World size: {process.world_size}")

    # Seed before construction so this stage has a reproducible random initialization.
    run_distributed_operation(
        process,
        lambda: set_seed(config.get("seed", 1337)),
        description="pretraining random-seed initialization",
    )

    # This path deliberately has no pretrained-checkpoint loader. The only allowed
    # checkpoint input is a Trainer checkpoint belonging to this same output run.
    if process.is_main_process:
        print("Initializing the NAR-VAE acoustic model from random weights...")
    use_speaker_conditioning, _, speaker_patch_size = run_distributed_operation(
        process,
        lambda: resolve_speaker_training_options(config),
        description="pretraining speaker-topology resolution",
    )
    duration_options = run_distributed_operation(
        process,
        lambda: resolve_duration_training_options(
            config,
            use_speaker_conditioning=use_speaker_conditioning,
        ),
        description="pretraining duration-topology resolution",
    )
    use_language_conditioning, _, supported_languages = run_distributed_operation(
        process,
        lambda: resolve_language_training_options(config),
        description="pretraining language-topology resolution",
    )
    supported_reference_languages, supported_language_pairs, _ = run_distributed_operation(
        process,
        lambda: resolve_reference_language_training_options(
            config,
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=use_language_conditioning,
            supported_languages=supported_languages,
        ),
        description="pretraining reference-language resolution",
    )
    if use_speaker_conditioning:
        ds_tts = run_distributed_operation(
            process,
            lambda: DynamicReferenceDataset(
                ds_tts,
                supported_language_pairs=supported_language_pairs or None,
                seed=config.get(
                    "reference_seed",
                    config.get("data_seed", config.get("seed", 1337)),
                ),
                min_reference_seconds=config.get("min_reference_seconds", 3.0),
                short_reference_max_seconds=config.get(
                    "short_reference_max_seconds",
                    8.0,
                ),
                max_reference_seconds=config.get("max_reference_seconds", 12.0),
                short_reference_probability=config.get(
                    "short_reference_probability",
                    0.8,
                ),
                speaker_patch_size=speaker_patch_size,
                strict=config.get("dynamic_reference_strict", True),
            ),
            description="dynamic speaker-reference dataset construction",
        )
    architecture = run_distributed_operation(
        process,
        lambda: resolve_model_architecture(config),
        description="pretraining architecture resolution",
    )
    flow_model = run_distributed_operation(
        process,
        lambda: create_flow_matching_echodit(
            latent_size=config["dacvae_latent_dim"],
            text_vocab_size=config["text_vocab_size"],
            text_conditioning_mode=config.get("text_conditioning_mode", "scratch_tokens"),
            conditioning_feature_size=config.get("conditioning_feature_size"),
            speaker_patch_size=speaker_patch_size,
            speaker_num_summary_tokens=config.get("speaker_num_summary_tokens", 0),
            target_patch_size=config.get("target_patch_size", 1),
            **architecture.model_kwargs(),
            norm_eps=config.get("norm_eps", 1e-6),
            cfg_dropout=config.get("cfg_dropout", 0.1),
            cfg_dropout_text=config.get("cfg_dropout_text"),
            cfg_dropout_speaker=config.get("cfg_dropout_speaker"),
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=use_language_conditioning,
            supported_languages=supported_languages,
            supported_reference_languages=supported_reference_languages,
            supported_language_pairs=supported_language_pairs or None,
            use_duration_predictor=duration_options.enabled,
            duration_predictor_hidden_size=duration_options.hidden_size,
            duration_predictor_num_layers=duration_options.num_layers,
            duration_predictor_use_speaker=duration_options.uses_speaker,
            use_mas_duration=duration_options.uses_mas,
            duration_alignment_hidden_size=duration_options.alignment_hidden_size,
            generative_objective=config.get("generative_objective", "rectified_flow"),
            diffusion_schedule_shift=config.get("diffusion_schedule_shift", 1.0),
        ),
        description="pretraining model construction",
    )
    print(f"Speaker conditioning: {'ENABLED' if use_speaker_conditioning else 'DISABLED'}")
    print(
        "Language conditioning: "
        f"{'ENABLED' if use_language_conditioning else 'DISABLED'} "
        f"({', '.join(supported_languages)})"
    )
    print(f"Learned duration: {'ENABLED' if duration_options.enabled else 'DISABLED'}")
    # Note: Do NOT move model to device manually - Trainer handles this

    # Print model info
    params_flow = run_distributed_operation(
        process,
        flow_model.get_num_params,
        description="pretraining model topology inspection",
    )
    print("\nNAR-VAE model parameters:")
    for k, v in params_flow.items():
        print(f"  {k}: {v / 1e6:.2f}M")

    # Apply layer freezing based on config
    # By default both encoders remain trainable.
    print("\nApplying layer freezing configuration...")
    freeze_info = run_distributed_operation(
        process,
        lambda: _freeze_pretraining_layers(flow_model, config),
        description="pretraining layer freezing",
    )
    print(
        f"  Text encoder training: {'DISABLED' if config.get('freeze_text_encoder', False) else 'ENABLED'}"
    )
    print(
        f"  Speaker encoder training: {'DISABLED' if config.get('freeze_speaker_encoder', False) else 'ENABLED'}"
    )
    print(f"  Frozen parameters: {freeze_info['frozen_params'] / 1e6:.2f}M")
    print(f"  Trainable parameters: {freeze_info['trainable_params'] / 1e6:.2f}M")
    print(f"  Frozen ratio: {freeze_info['frozen_ratio']:.1%}")

    print(f"TTS dataset: {len(ds_tts)} samples")
    run_distributed_operation(
        process,
        lambda: validate_tts_dataset(
            ds_tts,
            latent_size=config["dacvae_latent_dim"],
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=use_language_conditioning,
            supported_languages=supported_languages,
            supported_reference_languages=supported_reference_languages,
            supported_language_pairs=supported_language_pairs or None,
            require_language_coverage=True,
            use_mas_duration=duration_options.uses_mas,
            text_conditioning_mode=config.get("text_conditioning_mode", "scratch_tokens"),
            conditioning_feature_size=config.get("conditioning_feature_size"),
            frozen_text_provider_spec=(
                FrozenTextProviderSpec.from_config(config)
                if config.get("text_conditioning_mode", "scratch_tokens") == "frozen_features"
                else None
            ),
            text_vocab_size=config.get("text_vocab_size"),
            text_pad_token=config.get("pad_token"),
            allow_legacy_representation=config.get("allow_legacy_representation", False),
            expected_codec_source=config.get("dacvae_model"),
            expected_codec_backend=config.get("dacvae_backend"),
            expected_codec_revision=config.get("dacvae_revision"),
            expected_codec_filename=config.get("dacvae_filename"),
            expected_codec_sha256=config.get("dacvae_sha256"),
            expected_sample_rate=config.get("dacvae_sample_rate"),
            expected_hop_length=config.get("dacvae_hop_length"),
        ),
        description="pretraining dataset preflight",
    )

    # Data collator
    data_collator = run_distributed_operation(
        process,
        lambda: FlowMatchingDataCollator(
            pad_token=pad_token,
            speaker_patch_size=speaker_patch_size,
        ),
        description="pretraining data-collator construction",
    )

    # Mandatory W&B reporting is owned by Trainer and runs only on world process zero.
    run_distributed_operation(
        process,
        lambda: _configure_pretraining_wandb_environment(
            config,
            enabled=process.is_main_process,
        ),
        description="pretraining W&B environment setup",
    )

    # Training arguments
    training_args = run_distributed_operation(
        process,
        lambda: TrainingArguments(
            output_dir=save_folder,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 2),
            learning_rate=learning_rate,
            weight_decay=config.get("weight_decay", 0.01),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            logging_steps=config.get("logging_steps", 10),
            save_steps=save_steps,
            remove_unused_columns=False,
            lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
            dataloader_num_workers=config.get("dataloader_num_workers", 0),
            dataloader_pin_memory=config.get("dataloader_pin_memory", True),
            dataloader_drop_last=config.get("dataloader_drop_last", True),
            save_total_limit=config.get("save_total_limit", 3),
            local_rank=process.trainer_local_rank,
            **training_overrides,
        ),
        description="pretraining TrainingArguments construction",
    )

    # Initialize trainer
    trainer = run_distributed_operation(
        process,
        lambda: EchoDiTTrainer(
            model=flow_model,
            training_config=config,
            args=training_args,
            train_dataset=ds_tts,
            data_collator=data_collator,
        ),
        description="pretraining Trainer construction",
    )

    # Train
    print("=" * 80)
    print("Starting NAR-VAE pretraining...")
    print("=" * 80)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    print("Saving final model...")
    final_save_error = None
    try:
        trainer.save_model(os.path.join(save_folder, "final"))
    except Exception as exc:  # pragma: no cover - distributed filesystem failure
        final_save_error = exc
    propagate_distributed_error(process, final_save_error, description="final pretraining export")
    print("NAR-VAE pretraining complete!")


def pretrain(
    config_path: str | os.PathLike[str] = DEFAULT_TRAIN_CONFIG_PATH,
) -> None:
    """Pretrain a randomly initialized model with failure-safe DDP teardown."""
    with distributed_cleanup_guard():
        _pretrain(config_path)


def train(
    config_path: str | os.PathLike[str] = DEFAULT_TRAIN_CONFIG_PATH,
) -> None:
    """Compatibility alias for :func:`pretrain`."""
    pretrain(config_path)


__all__ = ["DEFAULT_TRAIN_CONFIG_PATH", "EchoDiTTrainer", "pretrain", "train"]
