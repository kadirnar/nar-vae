"""Typed configuration contracts shared by inference and training."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from difflib import get_close_matches
from functools import lru_cache
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any

from nar_vae.objectives import (
    DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    RECTIFIED_FLOW_OBJECTIVE,
    normalize_generative_objective,
    validate_diffusion_schedule_shift,
)
from nar_vae.tokenization import PAD_TOKEN, TOTAL_VOCAB_SIZE

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

SOLVERS = ("ddim", "euler", "midpoint", "heun", "rk4")
SOLVER_NFE_PER_STEP = {
    "ddim": 1,
    "euler": 1,
    "midpoint": 2,
    "heun": 2,
    "rk4": 4,
}
CFG_MODES = ("joint", "independent", "alternating")
CACHE_MODES = ("none", "cache_dit")
CACHE_DIT_MIN_STEPS = 8
TRAINING_STAGES = ("pretrain", "sft")
MIXED_PRECISION_MODES = ("fp32", "fp16", "bf16")
TRAINING_REPORTERS = ("wandb",)
TRAINING_LINEAGE_SCHEMA_VERSION = 1
TRAINING_LINEAGE_FILENAME = "lineage.json"
TRAINING_LIBRARY_NAME = "nar-vae"
TRAINING_RUN_MANIFEST_SCHEMA_VERSION = 2
TRAINING_RUN_MANIFEST_FILENAME = "run_manifest.json"
TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
TRAINING_CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
RESOLVED_TRAINING_DATASET_IDENTITY_KEY = "_resolved_dataset_identity"

_PRETRAINED_INITIALIZATION_KEYS = (
    "pretrained_checkpoint",
    "pretrained_model_name_or_path",
    "model_name_or_path",
    "init_checkpoint",
    "initial_checkpoint",
)
_CHECKPOINT_DIRECTORY = re.compile(r"checkpoint-[0-9]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HUB_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_RNG_STATE_ARTIFACT = re.compile(r"rng_state(?:_([0-9]+))?\.pth")
_UNSUPPORTED_TRAINING_FIELDS = frozenset(
    {
        # Inference belongs to configs/inference.toml and checkpoint-specific evaluation.
        "cfg_max_t",
        "cfg_min_t",
        "cfg_mode",
        "cfg_scale",
        "cfg_scale_speaker",
        "cfg_scale_text",
        "initial_noise_scale",
        "ode_solver",
        "ode_steps_inference",
        "temporal_rescale_k",
        "temporal_rescale_sigma",
        # These compatibility-era fields were never wired to either Trainer.
        "eval_steps",
        "final_ratio",
        "initial_ratio",
        "lr_min_ratio",
        "ratio",
        "use_echodit",
        "validation_cfg_scale",
        "validation_ode_steps",
        "validation_samples",
        "validation_solver",
        "validation_steps",
    }
)
_TRAINING_BOOLEAN_FIELDS = frozenset(
    {
        "allow_legacy_frame_length_inference",
        "allow_legacy_representation",
        "dataloader_drop_last",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "ddp_find_unused_parameters",
        "do_validation",
        "dynamic_reference_strict",
        "duration_predictor_use_speaker",
        "flow_velocity_weighted",
        "freeze_language_embedding",
        "freeze_speaker_encoder",
        "freeze_text_encoder",
        "gradient_checkpointing",
        "initialize_cross_lingual_capability",
        "initialize_duration_predictor",
        "initialize_language_conditioning",
        "initialize_speaker_conditioning",
        "logging_first_step",
        "muon_nesterov",
        "save_safetensors",
        "tf32",
        "torch_compile",
        "use_duration_predictor",
        "use_ema",
        "use_fsdp",
        "use_flash_attention",
        "use_language_conditioning",
        "use_mas_duration",
        "use_speaker_conditioning",
        "wandb_log_model",
    }
)

# Training YAML is a public experiment contract. Keep an explicit schema so a
# misspelled optimization does not get hashed into a run while Trainer silently
# uses its default. Runtime-derived dataset identity is deliberately excluded.
_COMMON_TRAINING_FIELDS = frozenset(
    {
        "TTS_dataset_local",
        "adaln_rank",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "allow_legacy_frame_length_inference",
        "allow_legacy_representation",
        "batch_size",
        "batching_cost",
        "cfg_dropout",
        "cfg_dropout_speaker",
        "cfg_dropout_text",
        "dacvae_backend",
        "dacvae_filename",
        "dacvae_hop_length",
        "dacvae_latent_dim",
        "dacvae_model",
        "dacvae_revision",
        "dacvae_sample_rate",
        "dacvae_sha256",
        "data_seed",
        "dataloader_drop_last",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "ddp_bucket_cap_mb",
        "ddp_find_unused_parameters",
        "do_validation",
        "dynamic_reference_strict",
        "duration_alignment_hidden_size",
        "duration_huber_delta",
        "duration_loss_weight",
        "duration_predictor_hidden_size",
        "duration_predictor_num_layers",
        "duration_predictor_use_speaker",
        "epochs",
        "flow_sigma_min",
        "flow_velocity_weighted",
        "generative_objective",
        "diffusion_schedule_shift",
        "frame_bucket_size",
        "freeze_first_n_layers",
        "freeze_language_embedding",
        "freeze_speaker_encoder",
        "freeze_text_encoder",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "intermediate_size",
        "learning_rate",
        "logging_dir",
        "logging_first_step",
        "logging_steps",
        "logit_normal_loc",
        "logit_normal_scale",
        "lr_scheduler_type",
        "mas_alignment_loss_weight",
        "mas_duration_loss_weight",
        "max_examples_per_batch",
        "max_attention_cost_per_batch",
        "max_frames_per_batch",
        "max_grad_norm",
        "max_reference_seconds",
        "mixed_precision",
        "min_reference_seconds",
        "model_preset",
        "model_size",
        "muon_adjust_lr_fn",
        "muon_epsilon",
        "muon_learning_rate",
        "muon_momentum",
        "muon_nesterov",
        "muon_ns_steps",
        "muon_weight_decay",
        "norm_eps",
        "num_heads",
        "num_layers",
        "num_strata",
        "optimizer",
        "pad_token",
        "pretrained_checkpoint",
        "project_name",
        "report_to",
        "reference_seed",
        "resume_from_checkpoint",
        "run_name",
        "save_folder",
        "save_safetensors",
        "save_steps",
        "save_total_limit",
        "seed",
        "short_reference_max_seconds",
        "short_reference_probability",
        "speaker_intermediate_size",
        "speaker_model_size",
        "speaker_num_heads",
        "speaker_num_layers",
        "speaker_num_summary_tokens",
        "speaker_patch_size",
        "supported_languages",
        "supported_language_pairs",
        "supported_reference_languages",
        "target_patch_size",
        "text_intermediate_size",
        "text_conditioning_mode",
        "conditioning_feature_size",
        "conditioning_feature_dtype",
        "frozen_text_alignment",
        "frozen_text_cache_version",
        "frozen_text_config_sha256",
        "frozen_text_encoder_id",
        "frozen_text_encoder_revision",
        "frozen_text_frontend",
        "frozen_text_hidden_layer",
        "frozen_text_model_filename",
        "frozen_text_model_sha256",
        "frozen_text_tokenizer_filename",
        "frozen_text_tokenizer_id",
        "frozen_text_tokenizer_revision",
        "frozen_text_tokenizer_sha256",
        "text_model_size",
        "text_num_heads",
        "text_num_layers",
        "text_vocab_size",
        "tf32",
        "timestep_distribution",
        "timestep_embed_size",
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        "training_stage",
        "use_duration_predictor",
        "use_flash_attention",
        "use_fsdp",
        "use_language_conditioning",
        "use_mas_duration",
        "use_speaker_conditioning",
        "wandb_log_model",
        "wandb_project",
        "wandb_run_name",
        "warmup_ratio",
        "warmup_steps",
        "weight_decay",
    }
)
_CHECKPOINT_MIGRATION_FIELDS = frozenset(
    {
        "initialize_cross_lingual_capability",
        "initialize_duration_predictor",
        "initialize_language_conditioning",
        "initialize_speaker_conditioning",
    }
)
_PRETRAINING_FIELDS = (
    _COMMON_TRAINING_FIELDS
    | _CHECKPOINT_MIGRATION_FIELDS
    | frozenset(
        {
            "TTS_dataset",
            "TTS_dataset_revision",
            "dataset_download_workers",
            "initial_checkpoint",
            "init_checkpoint",
            "model_initialization",
            "model_name_or_path",
            "pretrained_model_name_or_path",
        }
    )
)
_SFT_FIELDS = (
    _COMMON_TRAINING_FIELDS
    | _CHECKPOINT_MIGRATION_FIELDS
    | frozenset(
        {
            "ema_decay",
            "ema_update_every",
            "finetune_mode",
            "use_ema",
        }
    )
)


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a checkpoint without loading the artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash every resolved training option except the resume selector itself.

    Switching ``resume_from_checkpoint`` from ``null`` to ``true`` or to the
    selected ``checkpoint-N`` path must not turn an otherwise identical run
    into a different configuration. No other option is omitted.
    """
    normalized = {
        key: value for key, value in dict(config).items() if key != "resume_from_checkpoint"
    }
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        default=os.fspath,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bind_training_dataset_identity(
    config: dict[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach a validated runtime data identity before run initialization."""
    from nar_vae.dataset.identity import validate_training_dataset_identity

    if not isinstance(config, dict):
        raise ValueError("Resolved training dataset identity requires a mutable config dictionary.")
    validated = validate_training_dataset_identity(identity)
    existing = config.get(RESOLVED_TRAINING_DATASET_IDENTITY_KEY)
    if existing is not None and existing != validated:
        raise ValueError("The training configuration already carries a different dataset identity.")
    config[RESOLVED_TRAINING_DATASET_IDENTITY_KEY] = validated
    return validated


def _training_dataset_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    from nar_vae.dataset.identity import validate_training_dataset_identity

    identity = config.get(RESOLVED_TRAINING_DATASET_IDENTITY_KEY)
    if identity is None:
        raise ValueError(
            "Training run initialization requires a resolved immutable dataset identity."
        )
    return validate_training_dataset_identity(identity)


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON object without exposing a partially written manifest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_json(destination: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete JSON file without ever replacing an existing identity."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ValueError(
                f"Training run identity already exists and is immutable: {destination}."
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}.")
    if path.is_symlink():
        raise ValueError(f"The {description} cannot be a symlink: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description}: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"The {description} must be a JSON object: {path}.")
    return payload


def _training_output_path(config: Mapping[str, Any]) -> Path:
    save_folder = config.get("save_folder")
    if not isinstance(save_folder, (str, os.PathLike)) or not str(save_folder):
        raise ValueError("save_folder must be a non-empty filesystem path.")
    return Path(save_folder).expanduser().resolve()


def _validate_training_stage(config: Mapping[str, Any], stage: str) -> None:
    if stage not in TRAINING_STAGES:
        raise ValueError(f"Unknown training stage {stage!r}.")
    if config.get("training_stage") != stage:
        raise ValueError(
            f"The training run stage is {stage!r}, but training_stage is "
            f"{config.get('training_stage')!r}."
        )


def _validate_run_manifest(
    manifest: Any,
    *,
    stage: str,
    config_hash: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("The NAR-VAE run manifest must be a JSON object.")
    expected_fields = {
        "schema_version",
        "library",
        "stage",
        "run_id",
        "config_sha256",
        "dataset",
        "dataset_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("The NAR-VAE run manifest has an incomplete or unknown schema.")
    if manifest.get("schema_version") != TRAINING_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported NAR-VAE run-manifest schema.")
    if manifest.get("library") != TRAINING_LIBRARY_NAME:
        raise ValueError("The run manifest was not produced by NAR-VAE.")
    if manifest.get("stage") != stage:
        raise ValueError(f"The run manifest stage is {manifest.get('stage')!r}, not {stage!r}.")
    run_id = manifest.get("run_id")
    try:
        parsed_run_id = uuid.UUID(run_id) if isinstance(run_id, str) else None
    except ValueError as exc:
        raise ValueError("The run manifest has an invalid run_id.") from exc
    if parsed_run_id is None or str(parsed_run_id) != run_id:
        raise ValueError("The run manifest has an invalid run_id.")
    if manifest.get("config_sha256") != config_hash:
        raise ValueError(
            "The run manifest configuration does not match this configuration. "
            "Only resume_from_checkpoint may change during same-run recovery."
        )
    expected_dataset = _training_dataset_identity(
        {RESOLVED_TRAINING_DATASET_IDENTITY_KEY: manifest.get("dataset")}
    )
    from nar_vae.dataset.identity import training_dataset_identity_sha256

    if manifest.get("dataset_sha256") != training_dataset_identity_sha256(expected_dataset):
        raise ValueError("The run manifest has an invalid dataset identity SHA-256.")
    return dict(manifest)


def load_training_run_manifest(
    config: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Load and validate the immutable identity for one training output directory."""
    _validate_training_stage(config, stage)
    output_path = _training_output_path(config)
    manifest_path = output_path / TRAINING_RUN_MANIFEST_FILENAME
    manifest = _read_json_object(manifest_path, description="NAR-VAE run manifest")
    validated = _validate_run_manifest(
        manifest,
        stage=stage,
        config_hash=training_config_sha256(config),
    )
    if validated["dataset"] != _training_dataset_identity(config):
        raise ValueError("The current training dataset does not match this immutable run.")
    return validated


def _create_training_run_manifest(
    config: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    output_path = _training_output_path(config)
    destination = output_path / TRAINING_RUN_MANIFEST_FILENAME
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"save_folder must be a directory: {output_path}.")
    if destination.exists():
        existing = load_training_run_manifest(config, stage=stage)
        unexpected = [path for path in output_path.iterdir() if path != destination]
        if unexpected:
            raise ValueError(
                f"save_folder already contains an initialized run: {output_path}. "
                "Set resume_from_checkpoint to true or select a new save_folder."
            )
        # A process interrupted after run creation but before its first artifact
        # can deterministically restart without minting a different run identity.
        return existing
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(
            f"Refusing to initialize a run in non-empty save_folder without a "
            f"{TRAINING_RUN_MANIFEST_FILENAME}: {output_path}."
        )
    manifest = {
        "schema_version": TRAINING_RUN_MANIFEST_SCHEMA_VERSION,
        "library": TRAINING_LIBRARY_NAME,
        "stage": stage,
        "run_id": str(uuid.uuid4()),
        "config_sha256": training_config_sha256(config),
        "dataset": _training_dataset_identity(config),
    }
    from nar_vae.dataset.identity import training_dataset_identity_sha256

    manifest["dataset_sha256"] = training_dataset_identity_sha256(manifest["dataset"])
    _atomic_create_json(destination, manifest)
    return _validate_run_manifest(
        manifest,
        stage=stage,
        config_hash=training_config_sha256(config),
    )


def initialize_training_run(
    config: Mapping[str, Any],
    *,
    stage: str,
    is_world_process_zero: bool,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Create a run identity on rank zero and make it visible to every rank."""
    _validate_training_stage(config, stage)
    if not is_world_process_zero and synchronize is None:
        raise ValueError("Non-zero ranks require a synchronization callback.")
    if is_world_process_zero:
        if config.get("resume_from_checkpoint") in (None, False):
            _create_training_run_manifest(config, stage=stage)
        else:
            load_training_run_manifest(config, stage=stage)
    if synchronize is not None:
        synchronize()
    return load_training_run_manifest(config, stage=stage)


def _lineage_sha256(lineage: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(lineage),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_parent_lineage_reference(parent: Any) -> dict[str, str]:
    if not isinstance(parent, dict):
        raise ValueError("The checkpoint parent lineage must be an object.")
    expected = {"lineage_sha256", "stage", "config_sha256"}
    if set(parent) != expected:
        raise ValueError("The checkpoint parent lineage is incomplete.")
    if parent["stage"] not in TRAINING_STAGES:
        raise ValueError("The checkpoint parent has an unsupported training stage.")
    for name in ("lineage_sha256", "config_sha256"):
        if not isinstance(parent[name], str) or not _SHA256.fullmatch(parent[name]):
            raise ValueError(f"The checkpoint parent has an invalid {name}.")
    return dict(parent)


def _validate_training_lineage(lineage: Any) -> dict[str, Any]:
    if not isinstance(lineage, dict):
        raise ValueError("NAR-VAE training lineage must be a JSON object.")
    if lineage.get("schema_version") != TRAINING_LINEAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported NAR-VAE training-lineage schema.")
    if lineage.get("library") != TRAINING_LIBRARY_NAME:
        raise ValueError("The checkpoint lineage was not produced by NAR-VAE.")
    if lineage.get("stage") not in TRAINING_STAGES:
        raise ValueError("The checkpoint lineage has an unsupported training stage.")
    config_hash = lineage.get("config_sha256")
    if not isinstance(config_hash, str) or not _SHA256.fullmatch(config_hash):
        raise ValueError("The checkpoint lineage has an invalid config_sha256.")
    checkpoint_file = lineage.get("checkpoint_file")
    if (
        not isinstance(checkpoint_file, str)
        or not checkpoint_file
        or Path(checkpoint_file).name != checkpoint_file
    ):
        raise ValueError("The checkpoint lineage has an invalid checkpoint_file.")
    checkpoint_hash = lineage.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not _SHA256.fullmatch(checkpoint_hash):
        raise ValueError("The checkpoint lineage has an invalid checkpoint_sha256.")
    parent = lineage.get("parent")
    if parent is not None:
        _validate_parent_lineage_reference(parent)
    return dict(lineage)


def write_training_lineage(
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    stage: str,
    checkpoint_file: str | os.PathLike[str],
    parent_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write the lineage stored beside a training weight export."""
    if stage not in TRAINING_STAGES:
        raise ValueError(f"Unknown training stage {stage!r}.")
    parent = None
    if parent_lineage is not None:
        parent_payload = dict(parent_lineage)
        if set(parent_payload) == {"lineage_sha256", "stage", "config_sha256"}:
            # Same-run SFT resume preserves the original pretraining reference
            # without rewriting it as a new SFT parent.
            parent = _validate_parent_lineage_reference(parent_payload)
        else:
            validated_parent = _validate_training_lineage(parent_payload)
            parent = {
                "lineage_sha256": _lineage_sha256(validated_parent),
                "stage": validated_parent["stage"],
                "config_sha256": validated_parent["config_sha256"],
            }
    directory = Path(output_dir)
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.name != str(checkpoint_file):
        raise ValueError("checkpoint_file must be a filename relative to output_dir.")
    checkpoint_path = directory / checkpoint_path
    if not checkpoint_path.is_file():
        raise ValueError(f"Cannot write lineage for missing checkpoint: {checkpoint_path}.")
    lineage = {
        "schema_version": TRAINING_LINEAGE_SCHEMA_VERSION,
        "library": TRAINING_LIBRARY_NAME,
        "stage": stage,
        "config_sha256": training_config_sha256(config),
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "parent": parent,
    }
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / TRAINING_LINEAGE_FILENAME
    temporary = directory / f".{TRAINING_LINEAGE_FILENAME}.tmp"
    temporary.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return lineage


def load_training_lineage(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load lineage beside a weight file or from an export directory."""
    path = Path(checkpoint_path).expanduser()
    directory = path if path.is_dir() else path.parent
    lineage_path = directory / TRAINING_LINEAGE_FILENAME
    if not lineage_path.is_file():
        raise ValueError(
            "Fresh SFT requires a NAR-VAE training lineage beside pretrained_checkpoint; "
            f"missing: {lineage_path}. Legacy/external checkpoints are inference-only."
        )
    try:
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read NAR-VAE training lineage: {lineage_path}.") from exc
    return _validate_training_lineage(lineage)


def validate_parent_checkpoint(checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Require a local, lineage-bearing NAR-VAE parent for fresh SFT."""
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise ValueError(f"SFT pretrained_checkpoint must be a local weight file: {path}.")
    lineage = load_training_lineage(path)
    if lineage["stage"] != "pretrain":
        raise ValueError(
            "Fresh SFT must start from a NAR-VAE pretraining export; use same-run resume "
            "to continue an existing SFT run."
        )
    if lineage["checkpoint_file"] != path.name:
        raise ValueError(
            "pretrained_checkpoint does not name the weight artifact bound by its lineage."
        )
    actual_hash = _file_sha256(path)
    if actual_hash != lineage["checkpoint_sha256"]:
        raise ValueError("pretrained_checkpoint does not match its lineage SHA-256.")
    return lineage


def _relative_artifact_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Training artifact is outside its export directory: {path}.") from exc
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
        raise ValueError(f"Training manifests do not accept symlinked artifacts: {path}.")
    if path.resolve().parent != root and root not in path.resolve().parents:
        raise ValueError(f"Training artifact resolves outside its export directory: {path}.")
    return relative.as_posix()


def _find_trainer_model_artifact(output_path: Path) -> Path:
    candidates = [
        output_path / filename
        for filename in ("pytorch_model.bin", "model.safetensors")
        if (output_path / filename).is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "A bound training export requires exactly one root Trainer model artifact "
            "named pytorch_model.bin or model.safetensors."
        )
    return candidates[0]


def _training_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"WORLD_SIZE must be a positive integer; received {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"WORLD_SIZE must be a positive integer; received {value}.")
    return value


def _validate_exact_trainer_resume_artifacts(
    output_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Require every state artifact needed by the bounded Transformers Trainer."""
    required = {"trainer_state.json", "optimizer.pt", "scheduler.pt"}
    if config.get("mixed_precision", "fp32") == "fp16":
        required.add("scaler.pt")
    missing = sorted(name for name in required if not (output_path / name).is_file())
    if missing:
        raise ValueError(f"A resumable Trainer checkpoint is missing required state: {missing}.")

    world_size = _training_world_size()
    expected_rng = (
        {"rng_state.pth"}
        if world_size == 1
        else {f"rng_state_{rank}.pth" for rank in range(world_size)}
    )
    actual_rng = {
        path.name
        for path in output_path.iterdir()
        if path.is_file() and _RNG_STATE_ARTIFACT.fullmatch(path.name)
    }
    if actual_rng != expected_rng:
        raise ValueError(
            "A resumable Trainer checkpoint must bind the exact RNG state set for "
            f"WORLD_SIZE={world_size}: expected={sorted(expected_rng)}, "
            f"actual={sorted(actual_rng)}."
        )


def _list_training_artifacts(output_path: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for path in sorted(output_path.rglob("*")):
        if not path.is_file() or path.name == TRAINING_CHECKPOINT_MANIFEST_FILENAME:
            continue
        relative = _relative_artifact_path(path, output_path)
        artifacts[relative] = path
    if not artifacts:
        raise ValueError(f"Cannot bind an empty training export: {output_path}.")
    return artifacts


def _collect_training_artifact_hashes(output_path: Path) -> dict[str, str]:
    return {
        relative: _file_sha256(path)
        for relative, path in _list_training_artifacts(output_path).items()
    }


def _validate_resume_lineage(
    output_path: Path,
    *,
    config: Mapping[str, Any],
    stage: str,
) -> dict[str, Any]:
    flow_path = output_path / "flow_model" / "pytorch_model.bin"
    lineage = load_training_lineage(flow_path.parent)
    if lineage["stage"] != stage:
        raise ValueError(
            f"The resume checkpoint lineage stage is {lineage['stage']!r}, not {stage!r}."
        )
    if lineage["config_sha256"] != training_config_sha256(config):
        raise ValueError("The resume checkpoint lineage configuration does not match this run.")
    if lineage["checkpoint_file"] != flow_path.name:
        raise ValueError("The resume checkpoint lineage does not bind its flow-model artifact.")
    if not flow_path.is_file() or _file_sha256(flow_path) != lineage["checkpoint_sha256"]:
        raise ValueError("The resume flow-model artifact does not match its lineage SHA-256.")
    if stage == "pretrain" and lineage["parent"] is not None:
        raise ValueError("A pretraining resume checkpoint cannot declare a parent model.")
    if stage == "sft":
        if lineage["parent"] is None:
            raise ValueError("An SFT resume checkpoint is missing its pretraining parent lineage.")
        if lineage["parent"]["stage"] != "pretrain":
            raise ValueError(
                "An SFT resume checkpoint must preserve its pretraining parent lineage."
            )
    return lineage


def write_training_checkpoint_manifest(
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    stage: str,
    kind: str,
) -> dict[str, Any]:
    """Atomically bind a completed Trainer checkpoint or final export to its run."""
    if kind not in {"trainer_checkpoint", "export"}:
        raise ValueError("kind must be 'trainer_checkpoint' or 'export'.")
    _validate_training_stage(config, stage)
    run_manifest = load_training_run_manifest(config, stage=stage)
    run_path = _training_output_path(config)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path != run_path and run_path not in output_path.parents:
        raise ValueError(
            f"Training exports must remain inside the initialized save_folder: {run_path}."
        )
    if not output_path.is_dir():
        raise ValueError(f"Cannot bind missing training export directory: {output_path}.")

    trainer_artifact = _find_trainer_model_artifact(output_path)
    flow_artifact = output_path / "flow_model" / "pytorch_model.bin"
    if not flow_artifact.is_file():
        raise ValueError(
            f"The training export is missing its flow-model artifact: {flow_artifact}."
        )
    _validate_resume_lineage(output_path, config=config, stage=stage)

    if kind == "trainer_checkpoint":
        if output_path.parent != run_path or not _CHECKPOINT_DIRECTORY.fullmatch(output_path.name):
            raise ValueError("A Trainer checkpoint must be save_folder/checkpoint-N.")
        _validate_exact_trainer_resume_artifacts(output_path, config)

    ema_required = stage == "sft" and config.get("use_ema", False) is True
    if ema_required:
        for relative in ("flow_model/ema_model.bin", "flow_model/pytorch_model_ema.bin"):
            if not (output_path / relative).is_file():
                raise ValueError(f"EMA-enabled SFT checkpoint is missing {relative}.")

    artifact_hashes = _collect_training_artifact_hashes(output_path)
    trainer_relative = _relative_artifact_path(trainer_artifact, output_path)
    flow_relative = _relative_artifact_path(flow_artifact, output_path)
    manifest = {
        "schema_version": TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "library": TRAINING_LIBRARY_NAME,
        "stage": stage,
        "kind": kind,
        "run_id": run_manifest["run_id"],
        "config_sha256": run_manifest["config_sha256"],
        "trainer_artifact": trainer_relative,
        "flow_artifact": flow_relative,
        "ema_required": ema_required,
        "artifact_sha256": artifact_hashes,
    }
    destination = output_path / TRAINING_CHECKPOINT_MANIFEST_FILENAME
    _atomic_write_json(destination, manifest)
    return validate_training_checkpoint_manifest(
        output_path,
        config,
        stage=stage,
        expected_kind=kind,
    )


def validate_training_checkpoint_manifest(
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any],
    *,
    stage: str,
    expected_kind: str = "trainer_checkpoint",
) -> dict[str, Any]:
    """Hash-check one checkpoint/export and its immutable run binding."""
    _validate_training_stage(config, stage)
    if expected_kind not in {"trainer_checkpoint", "export"}:
        raise ValueError("expected_kind must be 'trainer_checkpoint' or 'export'.")
    run_manifest = load_training_run_manifest(config, stage=stage)
    run_path = _training_output_path(config)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path != run_path and run_path not in output_path.parents:
        raise ValueError("The training checkpoint is outside the initialized save_folder.")
    manifest_path = output_path / TRAINING_CHECKPOINT_MANIFEST_FILENAME
    manifest = _read_json_object(
        manifest_path,
        description="NAR-VAE checkpoint manifest",
    )
    expected_fields = {
        "schema_version",
        "library",
        "stage",
        "kind",
        "run_id",
        "config_sha256",
        "trainer_artifact",
        "flow_artifact",
        "ema_required",
        "artifact_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("The NAR-VAE checkpoint manifest has an incomplete or unknown schema.")
    if manifest.get("schema_version") != TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported NAR-VAE checkpoint-manifest schema.")
    if manifest.get("library") != TRAINING_LIBRARY_NAME:
        raise ValueError("The checkpoint manifest was not produced by NAR-VAE.")
    for field in ("stage", "run_id", "config_sha256"):
        if manifest.get(field) != run_manifest[field]:
            raise ValueError(f"The checkpoint {field} does not match its run manifest.")
    if manifest.get("kind") != expected_kind:
        raise ValueError(f"Expected a {expected_kind!r} manifest, got {manifest.get('kind')!r}.")
    expected_ema = stage == "sft" and config.get("use_ema", False) is True
    if manifest.get("ema_required") is not expected_ema:
        raise ValueError("The checkpoint EMA requirement does not match this configuration.")

    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("The checkpoint manifest has no artifact hashes.")
    for relative, expected_hash in artifact_hashes.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("The checkpoint manifest has an unsafe artifact path.")
        if not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash):
            raise ValueError("The checkpoint manifest has an invalid artifact SHA-256.")
        artifact_path = output_path / relative
        if not artifact_path.is_file():
            raise ValueError(f"The checkpoint is missing bound artifact {relative}.")
        _relative_artifact_path(artifact_path, output_path)
        if _file_sha256(artifact_path) != expected_hash:
            raise ValueError(f"Checkpoint artifact {relative} does not match its SHA-256.")

    current_artifacts = _list_training_artifacts(output_path)
    if set(current_artifacts) != set(artifact_hashes):
        raise ValueError("The checkpoint artifact set does not match its manifest.")
    for field in ("trainer_artifact", "flow_artifact"):
        relative = manifest.get(field)
        if not isinstance(relative, str) or relative not in artifact_hashes:
            raise ValueError(f"The checkpoint manifest has an invalid {field}.")
    if manifest["trainer_artifact"] not in {"pytorch_model.bin", "model.safetensors"}:
        raise ValueError("The checkpoint manifest does not select a root Trainer artifact.")
    if manifest["flow_artifact"] != "flow_model/pytorch_model.bin":
        raise ValueError("The checkpoint manifest does not select the flow-model artifact.")
    if expected_kind == "trainer_checkpoint":
        if output_path.parent != run_path or not _CHECKPOINT_DIRECTORY.fullmatch(output_path.name):
            raise ValueError("A same-run resume must select save_folder/checkpoint-N.")
        if "trainer_state.json" not in artifact_hashes:
            raise ValueError("The checkpoint manifest does not bind trainer_state.json.")
        _validate_exact_trainer_resume_artifacts(output_path, config)
    if "flow_model/lineage.json" not in artifact_hashes:
        raise ValueError("The checkpoint manifest does not bind flow-model lineage.")
    if expected_ema:
        for relative in ("flow_model/ema_model.bin", "flow_model/pytorch_model_ema.bin"):
            if relative not in artifact_hashes:
                raise ValueError(f"The checkpoint manifest does not bind {relative}.")
    _validate_resume_lineage(output_path, config=config, stage=stage)
    return dict(manifest)


@dataclass(frozen=True)
class MixedPrecisionSettings:
    """The mutually exclusive precision flags expected by Transformers Trainer."""

    mode: str

    @property
    def bf16(self) -> bool:
        return self.mode == "bf16"

    @property
    def fp16(self) -> bool:
        return self.mode == "fp16"


@dataclass(frozen=True)
class AdamWSettings:
    """Validated AdamW implementation and hyperparameters."""

    implementation: str
    beta1: float
    beta2: float
    epsilon: float


@dataclass(frozen=True)
class MuonSettings:
    """Validated native-Muon hyperparameters for hidden linear weights."""

    learning_rate: float
    weight_decay: float
    momentum: float
    nesterov: bool
    ns_steps: int
    epsilon: float
    adjust_lr_fn: str


@dataclass(frozen=True)
class CFGDropoutSettings:
    """Resolved training-time conditioning dropout probabilities."""

    default: float
    text: float
    speaker: float


@dataclass(frozen=True)
class FrameBudgetBatchingSettings:
    """Validated variable-length batching settings shared by pretraining and SFT."""

    max_frames_per_batch: int | None
    max_examples_per_batch: int | None
    frame_bucket_size: int
    allow_legacy_frame_length_inference: bool

    @property
    def enabled(self) -> bool:
        return self.max_frames_per_batch is not None


def _config_integer(
    value: Any,
    *,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return value


def resolve_frame_budget_batching(
    config: Mapping[str, Any],
) -> FrameBudgetBatchingSettings:
    """Validate the opt-in frame-budget sampler without guessing legacy lengths."""
    raw_max_frames = config.get("max_frames_per_batch")
    if raw_max_frames is None:
        max_frames = None
    else:
        max_frames_value = _config_integer(
            raw_max_frames,
            name="max_frames_per_batch",
            minimum=0,
        )
        max_frames = max_frames_value or None
    if config.get("batching_cost", "frames") == "padded_attention" and max_frames is None:
        raise ValueError("batching_cost: padded_attention requires max_frames_per_batch > 0.")

    raw_max_examples = config.get("max_examples_per_batch")
    if raw_max_examples is None:
        raw_max_examples = config.get("batch_size")
    max_examples = (
        None
        if raw_max_examples is None
        else _config_integer(
            raw_max_examples,
            name="max_examples_per_batch",
            minimum=1,
        )
    )
    if max_frames is not None and max_examples is None:
        raise ValueError(
            "Frame-budget batching requires max_examples_per_batch or a positive batch_size."
        )

    bucket_size = _config_integer(
        config.get("frame_bucket_size", 128),
        name="frame_bucket_size",
        minimum=1,
    )
    allow_legacy = config.get("allow_legacy_frame_length_inference", False)
    if not isinstance(allow_legacy, bool):
        raise ValueError("allow_legacy_frame_length_inference must be a boolean.")

    return FrameBudgetBatchingSettings(
        max_frames_per_batch=max_frames,
        max_examples_per_batch=max_examples,
        frame_bucket_size=bucket_size,
        allow_legacy_frame_length_inference=allow_legacy,
    )


def _dropout_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return probability


def resolve_training_cfg_dropout(config: Mapping[str, Any]) -> CFGDropoutSettings:
    """Resolve one default CFG dropout with optional per-condition overrides."""
    default = _dropout_probability(config.get("cfg_dropout", 0.1), name="cfg_dropout")
    raw_text = config.get("cfg_dropout_text")
    raw_speaker = config.get("cfg_dropout_speaker")
    return CFGDropoutSettings(
        default=default,
        text=(
            default if raw_text is None else _dropout_probability(raw_text, name="cfg_dropout_text")
        ),
        speaker=(
            default
            if raw_speaker is None
            else _dropout_probability(raw_speaker, name="cfg_dropout_speaker")
        ),
    )


def resolve_mixed_precision(value: Any) -> MixedPrecisionSettings:
    """Normalize one explicit precision mode without silently changing it."""
    if value is None:
        value = "fp32"
    if not isinstance(value, str):
        raise ValueError("mixed_precision must be one of: fp32, fp16, bf16.")
    mode = value.strip().lower()
    aliases = {"no": "fp32", "none": "fp32", "float32": "fp32"}
    mode = aliases.get(mode, mode)
    if mode not in MIXED_PRECISION_MODES:
        choices = ", ".join(MIXED_PRECISION_MODES)
        raise ValueError(f"Unknown mixed_precision {value!r}; expected one of: {choices}.")
    return MixedPrecisionSettings(mode=mode)


def resolve_training_reporters(value: Any) -> tuple[str, ...]:
    """Require W&B as the single training reporter.

    Importing the W&B client remains a runtime concern of the training entry
    points; configuration validation itself stays network-free and lightweight.
    """
    if value is None:
        return ("wandb",)
    if isinstance(value, str):
        normalized = value.strip().lower()
        requested = (normalized,)
    elif isinstance(value, (list, tuple)):
        requested = tuple(str(item).strip().lower() for item in value)
    else:
        raise ValueError("report_to must be 'wandb' or a list containing only 'wandb'.")

    if not requested or any(item in {"", "none", "no", "disabled"} for item in requested):
        raise ValueError("W&B logging is mandatory; report_to cannot be empty or disabled.")

    unsupported = sorted(set(requested) - set(TRAINING_REPORTERS))
    if unsupported:
        raise ValueError(
            f"Unsupported training reporter(s): {unsupported}. "
            "W&B is the mandatory training reporter."
        )
    return tuple(dict.fromkeys(requested))


def resolve_adamw_settings(config: Mapping[str, Any]) -> AdamWSettings:
    """Validate AdamW, including the auxiliary branch used with Muon."""
    optimizer = str(config.get("optimizer", "adamw")).strip().lower()
    implementations = {
        "adamw": "adamw_torch",
        "adamw_torch": "adamw_torch",
        "adamw_torch_fused": "adamw_torch_fused",
        # Transformers 4.x does not recognize Muon. This valid sentinel is
        # replaced by MuonTrainerMixin before Trainer constructs an optimizer.
        "muon": "adamw_torch",
    }
    if optimizer not in implementations:
        choices = ", ".join(implementations)
        raise ValueError(f"Unknown optimizer {optimizer!r}; expected one of: {choices}.")

    raw_beta1 = config.get("adam_beta1", 0.9)
    raw_beta2 = config.get("adam_beta2", 0.999)
    raw_epsilon = config.get("adam_epsilon", 1e-8)
    raw_values = (raw_beta1, raw_beta2, raw_epsilon)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
        raise ValueError("AdamW beta values and epsilon must be finite numbers.")
    beta1 = float(raw_beta1)
    beta2 = float(raw_beta2)
    epsilon = float(raw_epsilon)
    if not all(math.isfinite(value) for value in (beta1, beta2, epsilon)):
        raise ValueError("AdamW beta values and epsilon must be finite numbers.")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("AdamW beta values must satisfy 0 <= beta < 1.")
    if epsilon <= 0.0:
        raise ValueError("adam_epsilon must be positive.")
    return AdamWSettings(
        implementation=implementations[optimizer],
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )


_MUON_CONFIG_FIELDS = frozenset(
    {
        "muon_adjust_lr_fn",
        "muon_epsilon",
        "muon_learning_rate",
        "muon_momentum",
        "muon_nesterov",
        "muon_ns_steps",
        "muon_weight_decay",
    }
)


def resolve_muon_settings(config: Mapping[str, Any]) -> MuonSettings | None:
    """Validate opt-in Muon controls and reject silently ignored Muon keys."""
    optimizer = str(config.get("optimizer", "adamw")).strip().lower()
    configured_fields = sorted(_MUON_CONFIG_FIELDS.intersection(config))
    if optimizer != "muon":
        if configured_fields:
            raise ValueError(f"Muon-only fields require optimizer: muon: {configured_fields}.")
        return None

    raw_learning_rate = config.get("muon_learning_rate", config.get("learning_rate"))
    raw_weight_decay = config.get("muon_weight_decay", config.get("weight_decay", 0.01))
    raw_momentum = config.get("muon_momentum", 0.95)
    raw_epsilon = config.get("muon_epsilon", 1e-7)
    numerical = {
        "muon_learning_rate": raw_learning_rate,
        "muon_weight_decay": raw_weight_decay,
        "muon_momentum": raw_momentum,
        "muon_epsilon": raw_epsilon,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
        for value in numerical.values()
    ):
        raise ValueError("Muon learning rate, weight decay, momentum, and epsilon must be finite.")
    learning_rate = float(raw_learning_rate)
    weight_decay = float(raw_weight_decay)
    momentum = float(raw_momentum)
    epsilon = float(raw_epsilon)
    if learning_rate <= 0.0:
        raise ValueError("muon_learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("muon_weight_decay must be non-negative.")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("muon_momentum must be in [0, 1).")
    if epsilon <= 0.0:
        raise ValueError("muon_epsilon must be positive.")

    nesterov = config.get("muon_nesterov", True)
    if not isinstance(nesterov, bool):
        raise ValueError("muon_nesterov must be a boolean.")
    ns_steps = config.get("muon_ns_steps", 5)
    if isinstance(ns_steps, bool) or not isinstance(ns_steps, int) or not 1 <= ns_steps < 100:
        raise ValueError("muon_ns_steps must be an integer in [1, 100).")
    adjust_lr_fn = config.get("muon_adjust_lr_fn", "match_rms_adamw")
    if adjust_lr_fn not in {"original", "match_rms_adamw"}:
        raise ValueError("muon_adjust_lr_fn must be 'original' or 'match_rms_adamw'.")
    return MuonSettings(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        epsilon=epsilon,
        adjust_lr_fn=adjust_lr_fn,
    )


def _validate_same_run_resume_selector(config: Mapping[str, Any]) -> bool:
    """Validate resume syntax without touching potentially large artifacts."""
    value = config.get("resume_from_checkpoint")
    if value is None or value is False:
        return False
    if value is not True and not isinstance(value, (str, os.PathLike)):
        raise ValueError("resume_from_checkpoint must be null, true, or a checkpoint path.")

    stage = config.get("training_stage")
    if stage not in TRAINING_STAGES:
        raise ValueError("A same-run resume requires training_stage: pretrain or sft.")
    output_path = _training_output_path(config)
    if value is not True:
        checkpoint_path = Path(value).expanduser().resolve()
        if checkpoint_path.parent != output_path or not _CHECKPOINT_DIRECTORY.fullmatch(
            checkpoint_path.name
        ):
            raise ValueError(
                "resume_from_checkpoint must be the same run's "
                f"{output_path / 'checkpoint-N'} directory. Use SFT for external weights."
            )
    return True


def resolve_same_run_resume(config: Mapping[str, Any]) -> str | None:
    """Resolve and hash-check one exact Trainer checkpoint from this immutable run."""
    if not _validate_same_run_resume_selector(config):
        return None
    value = config["resume_from_checkpoint"]
    stage = config["training_stage"]
    output_path = _training_output_path(config)
    if value is True:
        candidates: list[tuple[int, Path]] = []
        if output_path.is_dir():
            for candidate in output_path.iterdir():
                match = _CHECKPOINT_DIRECTORY.fullmatch(candidate.name)
                if match is not None and candidate.is_dir():
                    candidates.append((int(candidate.name.rsplit("-", 1)[1]), candidate))
        if not candidates:
            raise ValueError(
                f"resume_from_checkpoint is true, but {output_path} has no checkpoint-N."
            )
        incomplete: list[Path] = []
        invalid: list[tuple[Path, str]] = []
        checkpoint_path = None
        for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
            if not (candidate / TRAINING_CHECKPOINT_MANIFEST_FILENAME).is_file():
                incomplete.append(candidate)
                continue
            try:
                validate_training_checkpoint_manifest(
                    candidate,
                    config,
                    stage=stage,
                    expected_kind="trainer_checkpoint",
                )
            except Exception as exc:
                invalid.append((candidate, str(exc)))
                continue
            checkpoint_path = candidate.resolve()
            break
        if checkpoint_path is None:
            rejected = "; ".join(f"{path.name}: {reason}" for path, reason in invalid)
            details = f" Rejected sealed candidates: {rejected}." if rejected else ""
            raise ValueError(
                f"resume_from_checkpoint is true, but {output_path} has no completely "
                f"validated checkpoint-N with {TRAINING_CHECKPOINT_MANIFEST_FILENAME}."
                f"{details}"
            )
        if incomplete:
            skipped = ", ".join(path.name for path in incomplete)
            warnings.warn(
                f"Ignoring incomplete checkpoint save(s) without an atomic manifest: {skipped}. "
                f"Resuming {checkpoint_path.name}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if invalid:
            skipped = ", ".join(path.name for path, _ in invalid)
            warnings.warn(
                f"Ignoring invalid sealed checkpoint save(s): {skipped}. "
                f"Resuming {checkpoint_path.name}.",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        checkpoint_path = Path(value).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise ValueError(f"Same-run resume checkpoint does not exist: {checkpoint_path}.")
    validate_training_checkpoint_manifest(
        checkpoint_path,
        config,
        stage=stage,
        expected_kind="trainer_checkpoint",
    )
    return str(checkpoint_path)


def invalidate_training_checkpoint_manifest(checkpoint_dir: str | os.PathLike[str]) -> None:
    """Remove an old completion seal before Trainer rewrites checkpoint artifacts."""
    path = Path(checkpoint_dir)
    manifest_path = path / TRAINING_CHECKPOINT_MANIFEST_FILENAME
    manifest_path.unlink(missing_ok=True)


def build_training_argument_overrides(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build validated Trainer options shared by pretraining and SFT."""
    for name in sorted(_TRAINING_BOOLEAN_FIELDS.intersection(config)):
        if not isinstance(config[name], bool):
            raise ValueError(f"{name} must be a boolean.")
    resolve_frame_budget_batching(config)
    precision = resolve_mixed_precision(config.get("mixed_precision", "fp32"))
    reporters = resolve_training_reporters(config.get("report_to", "wandb"))
    optimizer = resolve_adamw_settings(config)

    raw_learning_rate = config["learning_rate"]
    raw_weight_decay = config.get("weight_decay", 0.01)
    raw_warmup_steps = config.get("warmup_steps", 0)
    raw_warmup_ratio = config.get("warmup_ratio", 0.0)
    raw_max_grad_norm = config.get("max_grad_norm", 1.0)
    real_fields = {
        "learning_rate": raw_learning_rate,
        "weight_decay": raw_weight_decay,
        "warmup_ratio": raw_warmup_ratio,
        "max_grad_norm": raw_max_grad_norm,
    }
    invalid_real_fields = [
        name
        for name, value in real_fields.items()
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
    ]
    if invalid_real_fields:
        raise ValueError(f"Training numerical fields must be finite: {invalid_real_fields}.")
    if isinstance(raw_warmup_steps, bool) or not isinstance(raw_warmup_steps, int):
        raise ValueError("warmup_steps must be a non-negative integer.")
    learning_rate = float(raw_learning_rate)
    weight_decay = float(raw_weight_decay)
    warmup_steps = raw_warmup_steps
    warmup_ratio = float(raw_warmup_ratio)
    max_grad_norm = float(raw_max_grad_norm)
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative.")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")
    if warmup_steps < 0 or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_steps must be non-negative and warmup_ratio must be in [0, 1).")
    if warmup_steps and warmup_ratio:
        raise ValueError("Set only one of warmup_steps and warmup_ratio.")
    resolve_muon_settings(config)
    if config.get("use_fsdp", False):
        raise ValueError("use_fsdp is not implemented by this training entry point.")
    if config.get("use_flash_attention", False):
        raise ValueError(
            "use_flash_attention is not a training option; attention kernels are selected "
            "by the model/runtime."
        )

    integer_fields = {
        "ddp_bucket_cap_mb": (config.get("ddp_bucket_cap_mb", 25), 1),
        "seed": (config.get("seed", 1337), 0),
        "data_seed": (config.get("data_seed", config.get("seed", 1337)), 0),
    }
    for name, (value, minimum) in integer_fields.items():
        _config_integer(value, name=name, minimum=minimum)

    overrides: dict[str, Any] = {
        "bf16": precision.bf16,
        "fp16": precision.fp16,
        "report_to": list(reporters),
        "optim": optimizer.implementation,
        "adam_beta1": optimizer.beta1,
        "adam_beta2": optimizer.beta2,
        "adam_epsilon": optimizer.epsilon,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "gradient_checkpointing": config.get("gradient_checkpointing", False),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "ddp_find_unused_parameters": config.get("ddp_find_unused_parameters", False),
        "ddp_bucket_cap_mb": integer_fields["ddp_bucket_cap_mb"][0],
        "seed": integer_fields["seed"][0],
        "data_seed": integer_fields["data_seed"][0],
        "save_safetensors": config.get("save_safetensors", False),
        "torch_compile": config.get("torch_compile", False),
    }
    if overrides["torch_compile"]:
        if config.get("torch_compile_backend"):
            overrides["torch_compile_backend"] = str(config["torch_compile_backend"])
        if config.get("torch_compile_mode"):
            overrides["torch_compile_mode"] = str(config["torch_compile_mode"])
    elif config.get("torch_compile_backend") or config.get("torch_compile_mode"):
        raise ValueError("torch_compile_backend/torch_compile_mode require torch_compile: true.")
    for option in (
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "logging_dir",
        "logging_first_step",
        "tf32",
    ):
        if option in config:
            overrides[option] = config[option]
    return overrides


def _validate_known_training_fields(config: Mapping[str, Any]) -> None:
    """Reject unknown stage options with a useful nearest-name hint."""
    stage = config.get("training_stage")
    known = _PRETRAINING_FIELDS if stage == "pretrain" else _SFT_FIELDS
    unknown = sorted(set(config) - known)
    if not unknown:
        return
    suggestions = []
    for name in unknown:
        matches = get_close_matches(name, sorted(known), n=1, cutoff=0.72)
        if matches:
            suggestions.append(f"{name!r} -> {matches[0]!r}")
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ValueError(f"Unknown {stage} training configuration field(s): {unknown}.{hint}")


def _validate_training_config_contract(config: Mapping[str, Any]) -> None:
    if RESOLVED_TRAINING_DATASET_IDENTITY_KEY in config:
        raise ValueError(
            f"{RESOLVED_TRAINING_DATASET_IDENTITY_KEY} is runtime-derived and cannot be set "
            "in a training YAML."
        )
    unsupported = sorted(_UNSUPPORTED_TRAINING_FIELDS.intersection(config))
    if unsupported:
        raise ValueError(
            "Unsupported or unconsumed training configuration fields: "
            f"{unsupported}. Inference settings belong in configs/inference.toml; "
            "dataset mixing and generated-audio validation require an implemented pipeline."
        )
    _validate_known_training_fields(config)

    for name in sorted(_TRAINING_BOOLEAN_FIELDS.intersection(config)):
        if not isinstance(config[name], bool):
            raise ValueError(f"{name} must be a boolean.")

    integer_options = {
        "batch_size": 1,
        "dataloader_prefetch_factor": 1,
        "dataloader_num_workers": 0,
        "ddp_bucket_cap_mb": 1,
        "epochs": 1,
        "freeze_first_n_layers": 0,
        "gradient_accumulation_steps": 1,
        "logging_steps": 1,
        "num_strata": 1,
        "reference_seed": 0,
        "save_steps": 1,
        "save_total_limit": 1,
        "speaker_num_summary_tokens": 0,
        "target_patch_size": 1,
    }
    for name, minimum in integer_options.items():
        if name in config:
            _config_integer(config[name], name=name, minimum=minimum)
    if config.get("speaker_num_summary_tokens", 0) > 0 and not config.get(
        "use_speaker_conditioning", False
    ):
        raise ValueError("speaker_num_summary_tokens requires use_speaker_conditioning: true.")

    batching_cost = config.get("batching_cost", "frames")
    if batching_cost not in {"frames", "padded_attention"}:
        raise ValueError("batching_cost must be 'frames' or 'padded_attention'.")
    if "max_attention_cost_per_batch" in config:
        _config_integer(
            config["max_attention_cost_per_batch"],
            name="max_attention_cost_per_batch",
            minimum=1,
        )
    if batching_cost == "padded_attention" and "max_attention_cost_per_batch" not in config:
        raise ValueError("batching_cost: padded_attention requires max_attention_cost_per_batch.")

    workers = int(config.get("dataloader_num_workers", 0))
    if workers == 0 and config.get("dataloader_persistent_workers", False):
        raise ValueError("dataloader_persistent_workers requires dataloader_num_workers > 0.")
    if workers == 0 and "dataloader_prefetch_factor" in config:
        raise ValueError("dataloader_prefetch_factor requires dataloader_num_workers > 0.")

    reference_seconds = {
        "min_reference_seconds": config.get("min_reference_seconds", 3.0),
        "short_reference_max_seconds": config.get("short_reference_max_seconds", 8.0),
        "max_reference_seconds": config.get("max_reference_seconds", 12.0),
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in reference_seconds.values()
    ):
        raise ValueError("Dynamic reference durations must be finite positive numbers.")
    minimum_reference = float(reference_seconds["min_reference_seconds"])
    short_reference_maximum = float(reference_seconds["short_reference_max_seconds"])
    maximum_reference = float(reference_seconds["max_reference_seconds"])
    if not minimum_reference <= short_reference_maximum <= maximum_reference:
        raise ValueError(
            "Reference durations must satisfy min_reference_seconds <= "
            "short_reference_max_seconds <= max_reference_seconds."
        )
    short_probability = config.get("short_reference_probability", 0.8)
    if (
        isinstance(short_probability, bool)
        or not isinstance(short_probability, Real)
        or not math.isfinite(float(short_probability))
        or not 0.0 <= float(short_probability) <= 1.0
    ):
        raise ValueError("short_reference_probability must be a finite number in [0, 1].")
    reference_options = {
        "dynamic_reference_strict",
        "max_reference_seconds",
        "min_reference_seconds",
        "reference_seed",
        "short_reference_max_seconds",
        "short_reference_probability",
    }
    configured_reference_options = sorted(reference_options.intersection(config))
    if configured_reference_options and not config.get("use_speaker_conditioning", False):
        raise ValueError(
            "Dynamic reference options require use_speaker_conditioning: true: "
            f"{configured_reference_options}."
        )

    distribution = config.get("timestep_distribution", "stratified_logit_normal")
    if distribution not in {"uniform", "logit_normal", "stratified_logit_normal"}:
        raise ValueError(
            "timestep_distribution must be 'uniform', 'logit_normal', or 'stratified_logit_normal'."
        )
    flow_velocity_weighted = config.get("flow_velocity_weighted", False)
    if not isinstance(flow_velocity_weighted, bool):
        raise ValueError("flow_velocity_weighted must be a boolean.")
    objective = normalize_generative_objective(
        config.get("generative_objective", RECTIFIED_FLOW_OBJECTIVE)
    )
    schedule_shift = validate_diffusion_schedule_shift(
        config.get("diffusion_schedule_shift", DEFAULT_DIFFUSION_SCHEDULE_SHIFT)
    )
    if objective == RECTIFIED_FLOW_OBJECTIVE and schedule_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT:
        raise ValueError(
            "diffusion_schedule_shift is only meaningful with generative_objective: vp_diffusion_v."
        )
    logit_fields = {
        "logit_normal_loc": config.get("logit_normal_loc", 0.0),
        "logit_normal_scale": config.get("logit_normal_scale", 1.0),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value))
        for value in logit_fields.values()
    ):
        raise ValueError("Logit-normal location and scale must be finite numbers.")
    if float(logit_fields["logit_normal_scale"]) <= 0:
        raise ValueError("logit_normal_scale must be positive.")
    flow_sigma_min = config.get("flow_sigma_min", 1e-4)
    if (
        isinstance(flow_sigma_min, bool)
        or not isinstance(flow_sigma_min, Real)
        or not math.isfinite(float(flow_sigma_min))
        or float(flow_sigma_min) != 1e-4
    ):
        raise ValueError(
            "flow_sigma_min is a legacy no-op for the straight flow path and must remain 1e-4; "
            "non-default terminal-noise semantics require a versioned flow objective."
        )
    norm_eps = config.get("norm_eps", 1e-6)
    if (
        isinstance(norm_eps, bool)
        or not isinstance(norm_eps, Real)
        or not math.isfinite(float(norm_eps))
        or float(norm_eps) <= 0
    ):
        raise ValueError("norm_eps must be a finite positive number.")

    do_validation = config.get("do_validation", False)
    if not isinstance(do_validation, bool):
        raise ValueError("do_validation must be a boolean.")
    if do_validation:
        raise ValueError(
            "do_validation: true is not implemented: the training entry points do not "
            "construct a held-out evaluation dataset or a generated-audio quality loop. "
            "Keep it false and evaluate exported checkpoints with explicit evaluators."
        )
    allow_legacy_representation = config.get("allow_legacy_representation", False)
    if not isinstance(allow_legacy_representation, bool):
        raise ValueError("allow_legacy_representation must be a boolean.")
    text_conditioning_mode = config.get("text_conditioning_mode", "scratch_tokens")
    if text_conditioning_mode not in {"scratch_tokens", "frozen_features"}:
        raise ValueError("text_conditioning_mode must be 'scratch_tokens' or 'frozen_features'.")
    if text_conditioning_mode == "frozen_features":
        _config_integer(
            config.get("conditioning_feature_size"),
            name="conditioning_feature_size",
            minimum=1,
        )
        if config.get("text_num_layers", 0) != 0:
            raise ValueError("frozen_features text conditioning requires text_num_layers: 0.")
        if config.get("freeze_text_encoder", False):
            raise ValueError(
                "freeze_text_encoder would freeze the small trainable feature adapter; the "
                "pretrained backbone is already external and frozen."
            )
        required_frozen_fields = {
            "conditioning_feature_dtype",
            "frozen_text_alignment",
            "frozen_text_cache_version",
            "frozen_text_config_sha256",
            "frozen_text_encoder_id",
            "frozen_text_encoder_revision",
            "frozen_text_frontend",
            "frozen_text_hidden_layer",
            "frozen_text_model_filename",
            "frozen_text_model_sha256",
            "frozen_text_tokenizer_filename",
            "frozen_text_tokenizer_id",
            "frozen_text_tokenizer_revision",
            "frozen_text_tokenizer_sha256",
        }
        missing_frozen = sorted(name for name in required_frozen_fields if config.get(name) is None)
        if missing_frozen:
            raise ValueError(
                "frozen_features requires an immutable conditioner contract; missing: "
                f"{missing_frozen}."
            )
        for name in ("frozen_text_encoder_id", "frozen_text_tokenizer_id"):
            value = config[name]
            if not isinstance(value, str) or value.count("/") != 1 or not all(value.split("/")):
                raise ValueError(f"{name} must use the non-empty 'owner/name' Hub format.")
        for name in ("frozen_text_encoder_revision", "frozen_text_tokenizer_revision"):
            value = config[name]
            if not isinstance(value, str) or not _HUB_COMMIT.fullmatch(value):
                raise ValueError(f"{name} must be a full 40-character Hub commit.")
        for name in (
            "frozen_text_config_sha256",
            "frozen_text_model_sha256",
            "frozen_text_tokenizer_sha256",
        ):
            value = config[name]
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase 64-character SHA-256.")
        for name in ("frozen_text_model_filename", "frozen_text_tokenizer_filename"):
            value = config[name]
            if (
                not isinstance(value, str)
                or not value.strip()
                or Path(value).is_absolute()
                or ".." in Path(value).parts
            ):
                raise ValueError(f"{name} must be a non-empty repository-relative filename.")
        if config["conditioning_feature_dtype"] not in {"float16", "float32"}:
            raise ValueError(
                "conditioning_feature_dtype must be float16 or float32; prepared Arrow "
                "datasets do not losslessly preserve bfloat16."
            )
        if config["frozen_text_frontend"] not in {"phonemes", "raw_text"}:
            raise ValueError("frozen_text_frontend must be 'phonemes' or 'raw_text'.")
        if config["frozen_text_alignment"] != "hf_non_special_tokens_v1":
            raise ValueError(
                "frozen_text_alignment must be the versioned hf_non_special_tokens_v1 policy."
            )
        _config_integer(
            config["frozen_text_cache_version"],
            name="frozen_text_cache_version",
            minimum=1,
        )
        hidden_layer = config["frozen_text_hidden_layer"]
        if isinstance(hidden_layer, bool) or not isinstance(hidden_layer, int):
            raise ValueError("frozen_text_hidden_layer must be an integer layer index.")
    else:
        if config.get("conditioning_feature_size") not in (None, 0):
            raise ValueError(
                "conditioning_feature_size is only valid for frozen_features text conditioning."
            )
        unexpected_frozen = sorted(name for name in config if name.startswith("frozen_text_"))
        if unexpected_frozen:
            raise ValueError(
                "Frozen text conditioner fields require text_conditioning_mode: "
                f"frozen_features: {unexpected_frozen}."
            )

    if "text_vocab_size" in config or "pad_token" in config:
        text_vocab_size = _config_integer(
            config.get("text_vocab_size"),
            name="text_vocab_size",
            minimum=1,
        )
        pad_token = _config_integer(config.get("pad_token"), name="pad_token", minimum=0)
        if (
            text_conditioning_mode == "scratch_tokens"
            and not allow_legacy_representation
            and (text_vocab_size != TOTAL_VOCAB_SIZE or pad_token != PAD_TOKEN)
        ):
            raise ValueError(
                "Current prepared data requires the versioned compact frontend contract: "
                f"text_vocab_size={TOTAL_VOCAB_SIZE} and pad_token={PAD_TOKEN}."
            )
    resolve_training_cfg_dropout(config)


def validate_pretraining_config(config: Mapping[str, Any]) -> None:
    """Fail closed unless this is reproducible, random NAR-VAE pretraining."""
    if not isinstance(config, Mapping):
        raise ValueError("Pretraining configuration must be a mapping.")
    if config.get("training_stage") != "pretrain":
        raise ValueError("pretrain() requires training_stage: pretrain.")
    if config.get("model_initialization") != "random":
        raise ValueError("Pretraining requires model_initialization: random.")
    _validate_training_config_contract(config)

    external = [key for key in _PRETRAINED_INITIALIZATION_KEYS if config.get(key)]
    if external:
        raise ValueError(
            "Pretraining cannot initialize from external TTS weights. "
            f"Remove: {sorted(external)}. Use the SFT stage instead."
        )
    for option in (
        "initialize_speaker_conditioning",
        "initialize_language_conditioning",
        "initialize_cross_lingual_capability",
        "initialize_duration_predictor",
    ):
        if config.get(option, False):
            raise ValueError(
                f"{option} is a checkpoint-migration option and is invalid in pretraining."
            )
    if config.get("freeze_text_encoder", False) or int(config.get("freeze_first_n_layers", 0)):
        raise ValueError(
            "From-scratch pretraining cannot freeze randomly initialized text/DiT layers."
        )
    if config.get("use_language_conditioning", False) and config.get(
        "freeze_language_embedding", False
    ):
        raise ValueError("From-scratch language conditioning cannot freeze its random embedding.")
    if config.get("use_speaker_conditioning", False) and config.get(
        "freeze_speaker_encoder", False
    ):
        raise ValueError("From-scratch speaker conditioning cannot freeze its random encoder.")

    seed = config.get("seed", 1337)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    _validate_same_run_resume_selector(config)
    build_training_argument_overrides(config)


def validate_sft_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate supervised continuation separately from random pretraining."""
    if not isinstance(config, Mapping):
        raise ValueError("SFT configuration must be a mapping.")
    if config.get("training_stage") != "sft":
        raise ValueError("finetune() requires training_stage: sft.")
    _validate_training_config_contract(config)
    if not isinstance(config.get("use_ema", False), bool):
        raise ValueError("use_ema must be a boolean.")
    resume = _validate_same_run_resume_selector(config)
    if not config.get("pretrained_checkpoint") and not resume:
        raise ValueError("SFT requires pretrained_checkpoint or a same-run resume checkpoint.")
    parent_lineage = (
        validate_parent_checkpoint(config["pretrained_checkpoint"]) if not resume else None
    )
    if config.get("finetune_mode", "full") != "full":
        raise ValueError("Only finetune_mode: full is currently implemented.")
    build_training_argument_overrides(config)
    return parent_lineage


def cfg_guidance_active(
    *,
    cfg_scale: float,
    cfg_mode: str,
    cfg_scale_text: float | None = None,
    cfg_scale_speaker: float | None = None,
) -> bool:
    """Return whether the selected CFG formula differs from its neutral form.

    Joint and alternating guidance interpolate from an unconditional branch, so
    their neutral scale is one. Independent guidance adds text and speaker
    corrections to the conditional prediction, so each branch is neutral at
    zero. Keeping this decision in one place prevents the solver and Cache-DiT
    validator from silently disagreeing about explicit per-condition scales.
    """
    if cfg_mode not in CFG_MODES:
        raise ValueError(f"cfg_mode must be one of {CFG_MODES}.")
    text_scale = cfg_scale if cfg_scale_text is None else cfg_scale_text
    speaker_scale = cfg_scale if cfg_scale_speaker is None else cfg_scale_speaker
    if cfg_mode == "joint":
        return cfg_scale != 1.0
    if cfg_mode == "independent":
        return text_scale != 0.0 or speaker_scale != 0.0
    return text_scale != 1.0 or speaker_scale != 1.0


def validate_cache_dit_options(
    *,
    num_steps: int,
    solver: str,
    cfg_scale: float,
    cfg_mode: str,
    cfg_scale_text: float | None = None,
    cfg_scale_speaker: float | None = None,
    cfg_min_t: float,
    cfg_max_t: float,
) -> None:
    """Validate the fixed-shape solver path required by Cache-DiT."""
    if num_steps < CACHE_DIT_MIN_STEPS:
        raise ValueError(f"Cache-DiT turbo mode requires at least {CACHE_DIT_MIN_STEPS} steps.")
    if solver != "euler":
        raise ValueError("Cache-DiT turbo mode currently requires the Euler solver.")
    if cfg_guidance_active(
        cfg_scale=cfg_scale,
        cfg_mode=cfg_mode,
        cfg_scale_text=cfg_scale_text,
        cfg_scale_speaker=cfg_scale_speaker,
    ):
        if cfg_mode == "alternating":
            raise ValueError("Cache-DiT turbo mode does not support alternating CFG.")
        if cfg_min_t != 0.0 or cfg_max_t != 1.0:
            raise ValueError(
                "Cache-DiT requires a fixed CFG batch; use cfg_min_t=0 and cfg_max_t=1."
            )


@dataclass(frozen=True)
class DurationConfig:
    """Rules used when an output duration is not supplied explicitly."""

    seconds_per_character: float
    minimum_seconds: float
    maximum_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.seconds_per_character,
            self.minimum_seconds,
            self.maximum_seconds,
        )
        try:
            valid = all(
                not isinstance(value, bool) and math.isfinite(value) and value > 0
                for value in values
            )
        except TypeError:
            valid = False
        if not valid:
            raise ValueError("Duration settings must be finite positive numbers.")
        if self.minimum_seconds > self.maximum_seconds:
            raise ValueError("minimum_seconds cannot exceed maximum_seconds.")

    def estimate(self, text: str, explicit_duration: float | None = None) -> float:
        """Return a duration, rejecting requests that exceed the configured ceiling."""
        if explicit_duration is not None:
            try:
                valid_duration = not isinstance(explicit_duration, bool) and math.isfinite(
                    explicit_duration
                )
            except TypeError:
                valid_duration = False
            if not valid_duration:
                raise ValueError("duration must be a finite number.")
            if explicit_duration < self.minimum_seconds:
                raise ValueError(
                    f"duration must be at least the configured minimum of "
                    f"{self.minimum_seconds:g} seconds."
                )
            if explicit_duration > self.maximum_seconds:
                raise ValueError(
                    f"duration cannot exceed the configured maximum of "
                    f"{self.maximum_seconds:g} seconds."
                )
            return explicit_duration

        estimate = max(len(text.strip()) * self.seconds_per_character, self.minimum_seconds)
        if estimate > self.maximum_seconds:
            raise ValueError(
                f"Heuristic duration estimate {estimate:g}s exceeds the configured maximum of "
                f"{self.maximum_seconds:g}s. Split the text or use a checkpoint-specific "
                "duration policy with a larger validated limit."
            )
        return estimate


@dataclass(frozen=True)
class GenerationConfig:
    """All numerical choices that define one ODE generation profile."""

    name: str
    num_steps: int
    solver: str
    cfg_scale: float
    cfg_mode: str
    cfg_scale_text: float
    cfg_scale_speaker: float
    cfg_min_t: float
    cfg_max_t: float
    initial_noise_scale: float
    temporal_rescale_k: float
    temporal_rescale_sigma: float
    target_latent_std: float | None = None
    cache_mode: str = "none"

    def __post_init__(self) -> None:
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int):
            raise ValueError("num_steps must be a non-boolean integer.")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.solver not in SOLVERS:
            raise ValueError(f"Unknown solver {self.solver!r}; expected one of {SOLVERS}.")
        if self.cfg_mode not in CFG_MODES:
            raise ValueError(f"Unknown CFG mode {self.cfg_mode!r}; expected one of {CFG_MODES}.")
        if self.cache_mode not in CACHE_MODES:
            raise ValueError(
                f"Unknown cache mode {self.cache_mode!r}; expected one of {CACHE_MODES}."
            )
        finite_fields = {
            "cfg_scale": self.cfg_scale,
            "cfg_scale_text": self.cfg_scale_text,
            "cfg_scale_speaker": self.cfg_scale_speaker,
            "cfg_min_t": self.cfg_min_t,
            "cfg_max_t": self.cfg_max_t,
            "initial_noise_scale": self.initial_noise_scale,
            "temporal_rescale_k": self.temporal_rescale_k,
            "temporal_rescale_sigma": self.temporal_rescale_sigma,
        }
        try:
            invalid = [
                name
                for name, value in finite_fields.items()
                if isinstance(value, bool) or not math.isfinite(value)
            ]
        except TypeError as exc:
            raise ValueError("Generation numerical fields must be finite numbers.") from exc
        if invalid:
            raise ValueError(f"Generation numerical fields must be finite numbers: {invalid}.")
        if min(self.cfg_scale, self.cfg_scale_text, self.cfg_scale_speaker) < 0:
            raise ValueError("CFG scales must be nonnegative.")
        if not 0 <= self.cfg_min_t <= self.cfg_max_t <= 1:
            raise ValueError("CFG bounds must satisfy 0 <= cfg_min_t <= cfg_max_t <= 1.")
        if self.initial_noise_scale <= 0:
            raise ValueError("initial_noise_scale must be positive.")
        if self.temporal_rescale_k <= 0 or self.temporal_rescale_sigma <= 0:
            raise ValueError("Temporal rescale k and sigma must be positive.")
        if self.target_latent_std is not None:
            try:
                valid_target_std = not isinstance(self.target_latent_std, bool) and math.isfinite(
                    self.target_latent_std
                )
            except TypeError:
                valid_target_std = False
            if not valid_target_std or self.target_latent_std <= 0:
                raise ValueError("target_latent_std must be a finite positive number.")
        if self.cache_mode == "cache_dit":
            validate_cache_dit_options(
                num_steps=self.num_steps,
                solver=self.solver,
                cfg_scale=self.cfg_scale,
                cfg_mode=self.cfg_mode,
                cfg_scale_text=self.cfg_scale_text,
                cfg_scale_speaker=self.cfg_scale_speaker,
                cfg_min_t=self.cfg_min_t,
                cfg_max_t=self.cfg_max_t,
            )

    def with_overrides(self, **overrides: Any) -> "GenerationConfig":
        """Return a validated copy with non-None overrides applied."""
        selected = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **selected)


@dataclass(frozen=True)
class InferenceSettings:
    """Packaged model, codec, duration, and generation settings."""

    model: dict[str, Any]
    codec: dict[str, Any]
    duration: DurationConfig
    profiles: dict[str, GenerationConfig]

    def profile(self, name: str) -> GenerationConfig:
        """Return a named generation profile with a useful error for typos."""
        try:
            return self.profiles[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self.profiles))
            raise ValueError(
                f"Unknown inference profile {name!r}. Expected one of: {choices}."
            ) from exc


@lru_cache(maxsize=1)
def load_inference_settings() -> InferenceSettings:
    """Load the packaged TOML configuration once per process."""
    resource = resources.files("nar_vae.configs").joinpath("inference.toml")
    with resource.open("rb") as handle:
        raw = tomllib.load(handle)

    profiles = {
        name: GenerationConfig(name=name, **values) for name, values in raw["profiles"].items()
    }
    return InferenceSettings(
        model=dict(raw["model"]),
        codec=dict(raw["codec"]),
        duration=DurationConfig(**raw["duration"]),
        profiles=profiles,
    )


__all__ = [
    "AdamWSettings",
    "CACHE_DIT_MIN_STEPS",
    "CACHE_MODES",
    "CFG_MODES",
    "CFGDropoutSettings",
    "MIXED_PRECISION_MODES",
    "RESOLVED_TRAINING_DATASET_IDENTITY_KEY",
    "SOLVERS",
    "SOLVER_NFE_PER_STEP",
    "TRAINING_REPORTERS",
    "TRAINING_STAGES",
    "TRAINING_LIBRARY_NAME",
    "TRAINING_CHECKPOINT_MANIFEST_FILENAME",
    "TRAINING_CHECKPOINT_MANIFEST_SCHEMA_VERSION",
    "TRAINING_LINEAGE_FILENAME",
    "TRAINING_LINEAGE_SCHEMA_VERSION",
    "TRAINING_RUN_MANIFEST_FILENAME",
    "TRAINING_RUN_MANIFEST_SCHEMA_VERSION",
    "DurationConfig",
    "FrameBudgetBatchingSettings",
    "GenerationConfig",
    "InferenceSettings",
    "MixedPrecisionSettings",
    "MuonSettings",
    "build_training_argument_overrides",
    "bind_training_dataset_identity",
    "cfg_guidance_active",
    "initialize_training_run",
    "load_inference_settings",
    "load_training_run_manifest",
    "load_training_lineage",
    "resolve_adamw_settings",
    "resolve_frame_budget_batching",
    "resolve_mixed_precision",
    "resolve_muon_settings",
    "resolve_same_run_resume",
    "resolve_training_cfg_dropout",
    "resolve_training_reporters",
    "training_config_sha256",
    "validate_cache_dit_options",
    "validate_pretraining_config",
    "validate_parent_checkpoint",
    "validate_sft_config",
    "validate_training_checkpoint_manifest",
    "write_training_checkpoint_manifest",
    "write_training_lineage",
]
