"""CPU-only tests for explicit pretraining and SFT configuration contracts."""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nar_vae.configuration import (
    build_training_argument_overrides,
    load_training_lineage,
    resolve_frame_budget_batching,
    resolve_same_run_resume,
    resolve_training_cfg_dropout,
    validate_parent_checkpoint,
    validate_pretraining_config,
    validate_sft_config,
    write_training_lineage,
)
from nar_vae.finetune import _require_wandb as _require_sft_wandb
from nar_vae.train import (
    _load_pretraining_dataset,
    _load_pretraining_yaml,
    _resolve_pretraining_dataset_source,
)
from nar_vae.train import _require_wandb as _require_pretraining_wandb

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "d" * 40


class TrainingConfigurationTest(unittest.TestCase):
    def _pretraining_config(self, output: Path) -> dict:
        return {
            "training_stage": "pretrain",
            "model_initialization": "random",
            "save_folder": str(output),
            "learning_rate": 3e-4,
            "weight_decay": 0.05,
            "warmup_steps": 0,
            "warmup_ratio": 0.05,
            "optimizer": "adamw",
            "adam_beta1": 0.8,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-7,
            "mixed_precision": "fp16",
            "report_to": "wandb",
            "gradient_checkpointing": True,
            "seed": 17,
        }

    def test_pretraining_is_random_and_rejects_external_tts_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._pretraining_config(Path(directory) / "run")
            validate_pretraining_config(config)

            for key in (
                "pretrained_checkpoint",
                "pretrained_model_name_or_path",
                "model_name_or_path",
                "init_checkpoint",
            ):
                invalid = dict(config)
                invalid[key] = "someone-elses-model.bin"
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(
                        ValueError,
                        "cannot initialize from external TTS weights",
                    ),
                ):
                    validate_pretraining_config(invalid)

    def test_unknown_training_fields_and_extended_yaml_typos_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._pretraining_config(root / "pretrain")
            config["gradient_accumlation_steps"] = 8
            with self.assertRaisesRegex(
                ValueError,
                "gradient_accumlation_steps.*gradient_accumulation_steps",
            ):
                validate_pretraining_config(config)

            sft = {
                "training_stage": "sft",
                "save_folder": str(root / "sft"),
                "learning_rate": 1e-5,
                "resume_from_checkpoint": True,
                "learning_rate_scheduler": "linear",
            }
            with self.assertRaisesRegex(ValueError, "learning_rate_scheduler"):
                validate_sft_config(sft)

            if yaml is None:
                return
            base_path = root / "base.yaml"
            child_path = root / "child.yaml"
            base_path.write_text(
                yaml.safe_dump(self._pretraining_config(root / "run")),
                encoding="utf-8",
            )
            child_path.write_text(
                yaml.safe_dump(
                    {
                        "extends": base_path.name,
                        "gradient_accumlation_steps": 8,
                    }
                ),
                encoding="utf-8",
            )
            resolved = _load_pretraining_yaml(child_path)
            self.assertNotIn("extends", resolved)
            with self.assertRaisesRegex(ValueError, "gradient_accumlation_steps"):
                validate_pretraining_config(resolved)

    def test_flow_sigma_min_requires_the_versioned_straight_flow_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretraining = self._pretraining_config(root / "pretrain")
            sft = {
                "training_stage": "sft",
                "save_folder": str(root / "sft"),
                "learning_rate": 1e-5,
                "resume_from_checkpoint": True,
            }

            for config in (pretraining, sft):
                with self.subTest(stage=config["training_stage"]):
                    accepted = dict(config, flow_sigma_min=1e-4)
                    if config["training_stage"] == "pretrain":
                        self.assertIsNone(validate_pretraining_config(accepted))
                    else:
                        self.assertIsNone(validate_sft_config(accepted))

                    rejected = dict(config, flow_sigma_min=0.2)
                    validator = (
                        validate_pretraining_config
                        if config["training_stage"] == "pretrain"
                        else validate_sft_config
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "legacy no-op.*must remain 1e-4.*versioned flow objective",
                    ):
                        validator(rejected)

    def test_pretrain_is_the_canonical_api_and_train_remains_compatible(self):
        source = (ROOT / "nar_vae" / "train.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        functions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}

        self.assertIn("pretrain", functions)
        self.assertIn("train", functions)
        train_calls = [
            node
            for node in ast.walk(functions["train"])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        self.assertEqual([call.func.id for call in train_calls], ["pretrain"])
        self.assertIn('"configs/pretrain_config.yaml"', source)

        canonical_config = ROOT / "nar_vae" / "configs" / "pretrain_config.yaml"
        self.assertTrue(canonical_config.is_file())
        self.assertIn('extends: "echodit_config.yaml"', canonical_config.read_text())

    def test_pretraining_rejects_checkpoint_migration_and_random_layer_freezing(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._pretraining_config(Path(directory) / "run")
            for key, value in (
                ("initialize_duration_predictor", True),
                ("freeze_text_encoder", True),
                ("freeze_first_n_layers", 1),
            ):
                invalid = dict(config)
                invalid[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_pretraining_config(invalid)

    def test_resume_path_must_belong_to_the_same_output_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            config = self._pretraining_config(output)
            config["resume_from_checkpoint"] = str(output / "checkpoint-42")

            with self.assertRaisesRegex(ValueError, "does not exist"):
                resolve_same_run_resume(config)

            config["resume_from_checkpoint"] = str(root / "external" / "checkpoint-42")
            with self.assertRaisesRegex(ValueError, "same run"):
                validate_pretraining_config(config)

            config["resume_from_checkpoint"] = str(output / "model.bin")
            with self.assertRaisesRegex(ValueError, "checkpoint-N"):
                validate_pretraining_config(config)

    def test_training_arguments_wire_precision_optimizer_and_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._pretraining_config(Path(directory) / "run")
            options = build_training_argument_overrides(config)

        self.assertTrue(options["fp16"])
        self.assertFalse(options["bf16"])
        self.assertEqual(options["report_to"], ["wandb"])
        self.assertEqual(options["optim"], "adamw_torch")
        self.assertEqual(options["adam_beta1"], 0.8)
        self.assertEqual(options["adam_beta2"], 0.95)
        self.assertEqual(options["adam_epsilon"], 1e-7)
        self.assertTrue(options["gradient_checkpointing"])
        self.assertEqual(options["gradient_checkpointing_kwargs"], {"use_reentrant": False})

    def test_frame_budget_settings_validate_and_default_to_batch_size(self):
        settings = resolve_frame_budget_batching(
            {
                "batch_size": 7,
                "max_frames_per_batch": 4096,
                "frame_bucket_size": 64,
                "allow_legacy_frame_length_inference": False,
            }
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.max_frames_per_batch, 4096)
        self.assertEqual(settings.max_examples_per_batch, 7)
        self.assertEqual(settings.frame_bucket_size, 64)
        self.assertFalse(settings.allow_legacy_frame_length_inference)
        self.assertFalse(resolve_frame_budget_batching({"max_frames_per_batch": 0}).enabled)
        self.assertFalse(resolve_frame_budget_batching({"max_frames_per_batch": None}).enabled)

    def test_frame_budget_settings_fail_closed_on_invalid_values(self):
        invalid = (
            ({"max_frames_per_batch": -1}, "max_frames_per_batch"),
            ({"max_frames_per_batch": True}, "max_frames_per_batch"),
            (
                {"max_frames_per_batch": 10, "max_examples_per_batch": 0},
                "max_examples_per_batch",
            ),
            ({"frame_bucket_size": 0}, "frame_bucket_size"),
            (
                {"allow_legacy_frame_length_inference": "yes"},
                "allow_legacy_frame_length_inference",
            ),
        )
        for config, message in invalid:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                resolve_frame_budget_batching(config)

        with self.assertRaisesRegex(ValueError, "requires max_examples_per_batch"):
            resolve_frame_budget_batching({"max_frames_per_batch": 10})

    def test_packaged_training_configs_enable_persisted_frame_budgets(self):
        if yaml is None:
            self.skipTest("PyYAML dependency is not installed")

        for filename in ("echodit_config.yaml", "finetune_config.yaml"):
            with self.subTest(filename=filename):
                with (ROOT / "nar_vae" / "configs" / filename).open(encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                settings = resolve_frame_budget_batching(config)
                self.assertTrue(settings.enabled)
                self.assertEqual(settings.max_examples_per_batch, config["batch_size"])
                self.assertFalse(settings.allow_legacy_frame_length_inference)
                self.assertFalse(config["allow_legacy_representation"])
                self.assertEqual(config["report_to"], "wandb")

        pretraining = _load_pretraining_yaml(ROOT / "nar_vae" / "configs" / "pretrain_config.yaml")
        validate_pretraining_config(pretraining)
        with (ROOT / "nar_vae" / "configs" / "finetune_config.yaml").open(encoding="utf-8") as file:
            sft = yaml.safe_load(file)
        for name in (
            "use_mas_duration",
            "duration_alignment_hidden_size",
            "mas_duration_loss_weight",
            "mas_alignment_loss_weight",
        ):
            self.assertEqual(sft[name], pretraining[name], name)

    def test_configured_local_dataset_fails_closed_instead_of_using_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-prepared-data"
            config = {
                "TTS_dataset_local": str(missing),
                "TTS_dataset": "valid/dataset",
                "TTS_dataset_revision": DATASET_REVISION,
                "dataset_download_workers": 4,
            }

            with self.assertRaisesRegex(FileNotFoundError, "configured but does not exist"):
                _resolve_pretraining_dataset_source(config)

            missing.mkdir()
            source = _resolve_pretraining_dataset_source(config)
            self.assertEqual(source.kind, "local")
            self.assertEqual(source.location, str(missing))
            self.assertIsNone(source.revision)

    def test_remote_dataset_requires_repo_revision_and_bounded_workers(self):
        config = {
            "TTS_dataset_local": None,
            "TTS_dataset": "speech-corpus/paired-audio",
            "TTS_dataset_revision": DATASET_REVISION,
            "dataset_download_workers": 6,
        }
        source = _resolve_pretraining_dataset_source(config)
        self.assertEqual(source.kind, "remote")
        self.assertEqual(source.location, "speech-corpus/paired-audio")
        self.assertEqual(source.revision, DATASET_REVISION)
        self.assertEqual(source.download_workers, 6)

        for updates, message in (
            ({"TTS_dataset": "owner/dataset"}, "non-placeholder"),
            ({"TTS_dataset_local": 0}, "filesystem path"),
            ({"TTS_dataset_revision": None}, "40-character TTS_dataset_revision"),
            ({"TTS_dataset_revision": "main"}, "40-character Hub commit SHA"),
            ({"TTS_dataset_revision": "release-v1"}, "40-character Hub commit SHA"),
            ({"dataset_download_workers": 0}, "between 1 and 32"),
            ({"dataset_download_workers": 33}, "between 1 and 32"),
        ):
            invalid = dict(config, **updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, message):
                _resolve_pretraining_dataset_source(invalid)

    def test_remote_revision_loads_only_a_manifested_prepared_snapshot(self):
        source = _resolve_pretraining_dataset_source(
            {
                "TTS_dataset_local": None,
                "TTS_dataset": "speech-corpus/paired-audio",
                "TTS_dataset_revision": DATASET_REVISION,
                "dataset_download_workers": 5,
            }
        )
        expected_dataset = object()
        process = SimpleNamespace(is_main_process=True, is_distributed=False)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "nar_vae_dataset_manifest.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "nar_vae.train.snapshot_download", return_value=str(snapshot)
                ) as snapshot_download,
                patch("nar_vae.train.load_from_disk", return_value=expected_dataset) as load_disk,
            ):
                loaded, prepared_path = _load_pretraining_dataset(source, process)

        self.assertIs(loaded, expected_dataset)
        self.assertEqual(prepared_path, snapshot.resolve())
        snapshot_download.assert_called_once_with(
            repo_id="speech-corpus/paired-audio",
            repo_type="dataset",
            revision=DATASET_REVISION,
            max_workers=5,
        )
        load_disk.assert_called_once_with(str(snapshot.resolve()))

    def test_remote_dataset_script_without_prepared_manifest_is_rejected(self):
        source = _resolve_pretraining_dataset_source(
            {
                "TTS_dataset_local": None,
                "TTS_dataset": "speech-corpus/paired-audio",
                "TTS_dataset_revision": DATASET_REVISION,
            }
        )
        process = SimpleNamespace(is_main_process=True, is_distributed=False)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("nar_vae.train.snapshot_download", return_value=directory),
            self.assertRaisesRegex(ValueError, "mutable external URLs"),
        ):
            _load_pretraining_dataset(source, process)

    def test_each_node_leader_materializes_its_own_remote_snapshot(self):
        source = _resolve_pretraining_dataset_source(
            {
                "TTS_dataset_local": None,
                "TTS_dataset": "speech-corpus/paired-audio",
                "TTS_dataset_revision": DATASET_REVISION,
            }
        )
        process = SimpleNamespace(
            is_main_process=False,
            is_distributed=True,
            local_rank=0,
            rank=2,
            world_size=4,
        )
        expected_dataset = object()
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "nar_vae_dataset_manifest.json").write_text("{}", encoding="utf-8")

            phases = iter(("download_errors", "paths", "load_errors"))

            def gather_paths(outputs, local_value):
                phase = next(phases)
                if phase == "paths":
                    outputs[:] = ["/node-zero", None, local_value, None]
                else:
                    self.assertIsNone(local_value)
                    outputs[:] = [None] * 4

            with (
                patch("nar_vae.train.snapshot_download", return_value=str(snapshot)) as download,
                patch("nar_vae.train.dist.all_gather_object", side_effect=gather_paths),
                patch("nar_vae.train.load_from_disk", return_value=expected_dataset),
            ):
                loaded, prepared_path = _load_pretraining_dataset(source, process)

        self.assertIs(loaded, expected_dataset)
        self.assertEqual(prepared_path, snapshot.resolve())
        download.assert_called_once()

    def test_wandb_is_the_mandatory_reporter(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._pretraining_config(Path(directory) / "run")
            self.assertEqual(build_training_argument_overrides(config)["report_to"], ["wandb"])

            omitted = dict(config)
            omitted.pop("report_to")
            self.assertEqual(
                build_training_argument_overrides(omitted)["report_to"],
                ["wandb"],
            )

            for disabled in ("none", "disabled", []):
                rejected = dict(config, report_to=disabled)
                with (
                    self.subTest(disabled=disabled),
                    self.assertRaisesRegex(
                        ValueError,
                        "W&B logging is mandatory",
                    ),
                ):
                    build_training_argument_overrides(rejected)

            config["report_to"] = "tensorboard"
            with self.assertRaisesRegex(ValueError, "Unsupported training reporter"):
                build_training_argument_overrides(config)

    def test_training_entrypoints_require_wandb_without_eager_imports(self):
        for target, require in (
            ("nar_vae.train.import_module", _require_pretraining_wandb),
            ("nar_vae.finetune.import_module", _require_sft_wandb),
        ):
            with self.subTest(target=target), patch(target, side_effect=ImportError("missing")):
                with self.assertRaisesRegex(RuntimeError, "W&B is required"):
                    require()

    def test_cfg_dropout_default_and_branch_overrides_are_bounded(self):
        inherited = resolve_training_cfg_dropout({"cfg_dropout": 0.25})
        self.assertEqual(inherited.default, 0.25)
        self.assertEqual(inherited.text, 0.25)
        self.assertEqual(inherited.speaker, 0.25)

        overridden = resolve_training_cfg_dropout(
            {
                "cfg_dropout": 0.25,
                "cfg_dropout_text": 0.0,
                "cfg_dropout_speaker": 1.0,
            }
        )
        self.assertEqual(overridden.text, 0.0)
        self.assertEqual(overridden.speaker, 1.0)

        for key, value in (
            ("cfg_dropout", -0.01),
            ("cfg_dropout", float("nan")),
            ("cfg_dropout_text", 1.01),
            ("cfg_dropout_speaker", True),
            ("cfg_dropout_speaker", "0.1"),
        ):
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                resolve_training_cfg_dropout({key: value})

    def test_generated_audio_validation_fails_closed_for_both_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            pretraining = self._pretraining_config(Path(directory) / "pretrain")
            pretraining["do_validation"] = True
            with self.assertRaisesRegex(ValueError, "do_validation: true is not implemented"):
                validate_pretraining_config(pretraining)

            sft = {
                "training_stage": "sft",
                "save_folder": str(Path(directory) / "sft"),
                "learning_rate": 1e-5,
                "resume_from_checkpoint": True,
                "do_validation": True,
            }
            with self.assertRaisesRegex(ValueError, "do_validation: true is not implemented"):
                validate_sft_config(sft)

    def test_training_configs_reject_inference_and_unimplemented_mixing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self._pretraining_config(Path(directory) / "pretrain")
            for key, value in (
                ("ode_solver", "euler"),
                ("cfg_scale", 1.0),
                ("initial_noise_scale", 1.0),
                ("ratio", "1:1"),
                ("text_QA_dataset", "owner/prompts"),
                ("eval_steps", 500),
                ("validation_solver", "heun"),
            ):
                invalid = dict(base, **{key: value})
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ValueError, "Unsupported or unconsumed"),
                ):
                    validate_pretraining_config(invalid)

    def test_packaged_training_yaml_contains_only_wired_training_controls(self):
        if yaml is None:
            self.skipTest("PyYAML dependency is not installed")

        removed = {
            "cfg_max_t",
            "cfg_min_t",
            "cfg_mode",
            "cfg_scale",
            "cfg_scale_speaker",
            "cfg_scale_text",
            "eval_steps",
            "final_ratio",
            "flow_sigma_min",
            "initial_noise_scale",
            "initial_ratio",
            "lr_min_ratio",
            "ode_solver",
            "ode_steps_inference",
            "ratio",
            "temporal_rescale_k",
            "temporal_rescale_sigma",
            "text_QA_dataset",
            "use_echodit",
            "validation_cfg_scale",
            "validation_ode_steps",
            "validation_samples",
            "validation_solver",
            "validation_steps",
        }
        for filename in ("echodit_config.yaml", "finetune_config.yaml"):
            with self.subTest(filename=filename):
                with (ROOT / "nar_vae" / "configs" / filename).open(encoding="utf-8") as file:
                    config = yaml.safe_load(file)
                self.assertFalse(removed.intersection(config))
                self.assertIs(config["do_validation"], False)

    def test_torch_compile_is_explicit_and_wires_backend_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._pretraining_config(Path(directory) / "run")
            config.update(
                torch_compile=True,
                torch_compile_backend="inductor",
                torch_compile_mode="reduce-overhead",
            )
            options = build_training_argument_overrides(config)

            self.assertTrue(options["torch_compile"])
            self.assertEqual(options["torch_compile_backend"], "inductor")
            self.assertEqual(options["torch_compile_mode"], "reduce-overhead")

            config["torch_compile"] = False
            with self.assertRaisesRegex(ValueError, "require torch_compile: true"):
                build_training_argument_overrides(config)

    def test_mixed_precision_and_optimizer_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._pretraining_config(Path(directory) / "run")
            config["mixed_precision"] = "automatic"
            with self.assertRaisesRegex(ValueError, "Unknown mixed_precision"):
                build_training_argument_overrides(config)

            config["mixed_precision"] = "bf16"
            config["optimizer"] = "mystery"
            with self.assertRaisesRegex(ValueError, "Unknown optimizer"):
                build_training_argument_overrides(config)

    def test_nonfinite_optimizer_and_training_numerics_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self._pretraining_config(Path(directory) / "run")
            for field, value in (
                ("learning_rate", float("nan")),
                ("learning_rate", float("inf")),
                ("weight_decay", float("nan")),
                ("warmup_ratio", float("nan")),
                ("warmup_steps", 0.5),
                ("max_grad_norm", float("nan")),
                ("max_grad_norm", 0.0),
                ("adam_beta1", float("nan")),
                ("adam_beta2", float("inf")),
                ("adam_epsilon", float("nan")),
                ("adam_epsilon", True),
            ):
                invalid = dict(base, **{field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    build_training_argument_overrides(invalid)

    def test_boolean_and_flow_sampling_controls_are_strictly_typed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = self._pretraining_config(Path(directory) / "run")
            for field, value in (
                ("gradient_checkpointing", "false"),
                ("dataloader_pin_memory", 1),
                ("flow_velocity_weighted", "false"),
                ("norm_eps", 0.0),
                ("norm_eps", -1e-6),
                ("logit_normal_loc", float("nan")),
                ("logit_normal_scale", 0.0),
                ("num_strata", True),
                ("timestep_distribution", "mystery"),
            ):
                invalid = dict(base, **{field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    validate_pretraining_config(invalid)

    def test_packaged_yaml_does_not_advertise_unconsumed_token_controls(self):
        if yaml is None:
            self.skipTest("PyYAML dependency is not installed")
        removed = {
            "tokeniser_length",
            "start_of_text",
            "end_of_text",
            "start_of_speech",
            "end_of_speech",
            "start_of_human",
            "end_of_human",
            "start_of_ai",
            "end_of_ai",
        }
        for filename in ("echodit_config.yaml", "finetune_config.yaml"):
            with (ROOT / "nar_vae" / "configs" / filename).open(encoding="utf-8") as file:
                config = yaml.safe_load(file)
            self.assertFalse(removed.intersection(config))
            self.assertEqual(config["pad_token"], 100286)

    def test_sft_requires_a_parent_or_same_run_resume_and_supported_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sft"
            parent = Path(directory) / "pretrain" / "pytorch_model.bin"
            config = {
                "training_stage": "sft",
                "save_folder": str(output),
                "learning_rate": 1e-5,
                "pretrained_checkpoint": None,
                "report_to": "wandb",
            }
            with self.assertRaisesRegex(ValueError, "requires pretrained_checkpoint"):
                validate_sft_config(config)

            resumed = dict(config, resume_from_checkpoint=True)
            self.assertIsNone(validate_sft_config(resumed))

            parent.parent.mkdir()
            parent.write_bytes(b"test weights")
            write_training_lineage(
                parent.parent,
                self._pretraining_config(Path(directory) / "pretrain-run"),
                stage="pretrain",
                checkpoint_file=parent.name,
            )
            config["pretrained_checkpoint"] = str(parent)
            parent_lineage = validate_sft_config(config)
            self.assertEqual(parent_lineage["stage"], "pretrain")
            config["finetune_mode"] = "lora"
            with self.assertRaisesRegex(ValueError, "Only finetune_mode: full"):
                validate_sft_config(config)

            config["finetune_mode"] = "full"
            parent.write_bytes(b"tampered weights")
            with self.assertRaisesRegex(ValueError, "lineage SHA-256"):
                validate_sft_config(config)

    def test_fresh_sft_rejects_external_or_legacy_weights_without_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "external" / "model.bin"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"external weights")
            config = {
                "training_stage": "sft",
                "save_folder": str(root / "sft"),
                "learning_rate": 1e-5,
                "pretrained_checkpoint": str(checkpoint),
                "report_to": "wandb",
            }

            with self.assertRaisesRegex(ValueError, "Legacy/external checkpoints"):
                validate_sft_config(config)

    def test_sft_lineage_records_its_pretraining_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrain_dir = root / "pretrain"
            pretrain_dir.mkdir()
            (pretrain_dir / "pytorch_model.bin").write_bytes(b"pretraining weights")
            parent = write_training_lineage(
                pretrain_dir,
                self._pretraining_config(root / "pretrain-run"),
                stage="pretrain",
                checkpoint_file="pytorch_model.bin",
            )
            sft_dir = root / "sft"
            sft_dir.mkdir()
            (sft_dir / "pytorch_model.bin").write_bytes(b"sft weights")
            child = write_training_lineage(
                sft_dir,
                {
                    "training_stage": "sft",
                    "learning_rate": 1e-5,
                },
                stage="sft",
                checkpoint_file="pytorch_model.bin",
                parent_lineage=parent,
            )

            loaded = load_training_lineage(root / "sft")
            self.assertEqual(loaded, child)
            self.assertEqual(child["stage"], "sft")
            self.assertEqual(child["parent"]["stage"], "pretrain")
            self.assertEqual(child["parent"]["config_sha256"], parent["config_sha256"])

            sft_weights = sft_dir / "pytorch_model.bin"
            with self.assertRaisesRegex(ValueError, "pretraining export"):
                validate_parent_checkpoint(sft_weights)


if __name__ == "__main__":
    unittest.main()
