"""CPU-only tests for hash-bound, same-run Trainer recovery."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vyvotts.configuration import (
    RESOLVED_TRAINING_DATASET_IDENTITY_KEY,
    TRAINING_CHECKPOINT_MANIFEST_FILENAME,
    bind_training_dataset_identity,
    initialize_training_run,
    invalidate_training_checkpoint_manifest,
    resolve_same_run_resume,
    training_config_sha256,
    validate_training_checkpoint_manifest,
    write_training_checkpoint_manifest,
    write_training_lineage,
)


class ResumeIntegrityTest(unittest.TestCase):
    def _config(self, output: Path, *, stage: str = "pretrain", use_ema: bool = False) -> dict:
        config = {
            "training_stage": stage,
            "save_folder": str(output),
            "learning_rate": 1e-4,
            "seed": 17,
            "resume_from_checkpoint": None,
        }
        if stage == "pretrain":
            config["model_initialization"] = "random"
        else:
            config["use_ema"] = use_ema
        bind_training_dataset_identity(
            config,
            {
                "schema_version": 1,
                "library": "nar-vae",
                "kind": "local-prepared",
                "source": str(output.parent / "prepared-data"),
                "revision": None,
                "split": None,
                "fingerprint": "test-fingerprint",
                "num_rows": 2,
                "columns": ["latents", "conditioning_ids"],
                "content_sha256": "d" * 64,
            },
        )
        return config

    def _pretraining_parent(self, root: Path) -> dict:
        parent_dir = root / "pretraining-export"
        parent_dir.mkdir()
        (parent_dir / "pytorch_model.bin").write_bytes(b"pretraining-parent")
        return write_training_lineage(
            parent_dir,
            self._config(root / "pretraining-run"),
            stage="pretrain",
            checkpoint_file="pytorch_model.bin",
        )

    def _checkpoint(
        self,
        config: dict,
        step: int,
        *,
        parent_lineage: dict | None = None,
    ) -> Path:
        checkpoint = Path(config["save_folder"]) / f"checkpoint-{step}"
        flow_dir = checkpoint / "flow_model"
        flow_dir.mkdir(parents=True)
        (checkpoint / "pytorch_model.bin").write_bytes(f"trainer-{step}".encode())
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}),
            encoding="utf-8",
        )
        (checkpoint / "optimizer.pt").write_bytes(f"optimizer-{step}".encode())
        (checkpoint / "scheduler.pt").write_bytes(f"scheduler-{step}".encode())
        (checkpoint / "rng_state.pth").write_bytes(f"rng-{step}".encode())
        if config.get("mixed_precision") == "fp16":
            (checkpoint / "scaler.pt").write_bytes(f"scaler-{step}".encode())
        (flow_dir / "pytorch_model.bin").write_bytes(f"flow-{step}".encode())
        stage = config["training_stage"]
        if stage == "sft" and config.get("use_ema"):
            (flow_dir / "ema_model.bin").write_bytes(f"ema-state-{step}".encode())
            (flow_dir / "pytorch_model_ema.bin").write_bytes(f"ema-weights-{step}".encode())
        write_training_lineage(
            flow_dir,
            config,
            stage=stage,
            checkpoint_file="pytorch_model.bin",
            parent_lineage=parent_lineage,
        )
        write_training_checkpoint_manifest(
            checkpoint,
            config,
            stage=stage,
            kind="trainer_checkpoint",
        )
        return checkpoint

    def test_resume_selector_is_the_only_config_field_excluded_from_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            baseline = training_config_sha256(config)
            self.assertEqual(
                baseline,
                training_config_sha256(dict(config, resume_from_checkpoint=True)),
            )
            self.assertEqual(
                baseline,
                training_config_sha256(dict(config, resume_from_checkpoint="checkpoint-123")),
            )
            self.assertNotEqual(
                baseline,
                training_config_sha256(dict(config, learning_rate=2e-4)),
            )

    def test_run_manifest_is_atomic_synchronized_and_stable_before_first_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            synchronize = Mock()
            first = initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
                synchronize=synchronize,
            )
            second = initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )

            synchronize.assert_called_once_with()
            self.assertEqual(first, second)
            self.assertEqual(str(uuid.UUID(first["run_id"])), first["run_id"])

    def test_explicit_and_latest_resume_resolve_one_validated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )
            first = self._checkpoint(config, 3)
            latest = self._checkpoint(config, 12)

            explicit = dict(config, resume_from_checkpoint=str(first))
            automatic = dict(config, resume_from_checkpoint=True)
            self.assertEqual(resolve_same_run_resume(explicit), str(first.resolve()))
            self.assertEqual(resolve_same_run_resume(automatic), str(latest.resolve()))

    def test_latest_incomplete_save_falls_back_only_to_a_validated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )
            complete = self._checkpoint(config, 1)
            (Path(config["save_folder"]) / "checkpoint-2").mkdir()

            with self.assertWarnsRegex(RuntimeWarning, "Ignoring incomplete checkpoint"):
                selected = resolve_same_run_resume(dict(config, resume_from_checkpoint=True))
            self.assertEqual(selected, str(complete.resolve()))

    def test_latest_stale_seal_falls_back_to_prior_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )
            complete = self._checkpoint(config, 1)
            stale = self._checkpoint(config, 2)
            (stale / "optimizer.pt").write_bytes(b"partially-rewritten-optimizer")

            with self.assertWarnsRegex(RuntimeWarning, "Ignoring invalid sealed checkpoint"):
                selected = resolve_same_run_resume(dict(config, resume_from_checkpoint=True))
            self.assertEqual(selected, str(complete.resolve()))
            with self.assertRaisesRegex(ValueError, "does not match its SHA-256"):
                resolve_same_run_resume(dict(config, resume_from_checkpoint=str(stale)))

    def test_checkpoint_seal_invalidation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(config, stage="pretrain", is_world_process_zero=True)
            checkpoint = self._checkpoint(config, 3)
            seal = checkpoint / TRAINING_CHECKPOINT_MANIFEST_FILENAME

            invalidate_training_checkpoint_manifest(checkpoint)
            invalidate_training_checkpoint_manifest(checkpoint)

            self.assertFalse(seal.exists())

    def test_checkpoint_seal_requires_optimizer_scheduler_rng_and_fp16_scaler(self):
        for missing in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                config = self._config(Path(directory) / "run")
                initialize_training_run(config, stage="pretrain", is_world_process_zero=True)
                checkpoint = self._checkpoint(config, 3)
                invalidate_training_checkpoint_manifest(checkpoint)
                (checkpoint / missing).unlink()
                with self.assertRaisesRegex(ValueError, "missing required state|exact RNG state"):
                    write_training_checkpoint_manifest(
                        checkpoint,
                        config,
                        stage="pretrain",
                        kind="trainer_checkpoint",
                    )

        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            config["mixed_precision"] = "fp16"
            initialize_training_run(config, stage="pretrain", is_world_process_zero=True)
            checkpoint = self._checkpoint(config, 3)
            invalidate_training_checkpoint_manifest(checkpoint)
            (checkpoint / "scaler.pt").unlink()
            with self.assertRaisesRegex(ValueError, "scaler.pt"):
                write_training_checkpoint_manifest(
                    checkpoint,
                    config,
                    stage="pretrain",
                    kind="trainer_checkpoint",
                )

    def test_copied_manifest_and_modified_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )
            checkpoint = self._checkpoint(config, 4)
            manifest_path = checkpoint / TRAINING_CHECKPOINT_MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["run_id"] = str(uuid.uuid4())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            resumed = dict(config, resume_from_checkpoint=str(checkpoint))
            with self.assertRaisesRegex(ValueError, "run_id"):
                resolve_same_run_resume(resumed)

            # Restore a correctly bound manifest, then alter an actual Trainer artifact.
            write_training_checkpoint_manifest(
                checkpoint,
                config,
                stage="pretrain",
                kind="trainer_checkpoint",
            )
            (checkpoint / "pytorch_model.bin").write_bytes(b"stale-copied-weights")
            with self.assertRaisesRegex(ValueError, "does not match its SHA-256"):
                resolve_same_run_resume(resumed)

    def test_resume_rejects_changed_dataset_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )
            self._checkpoint(config, 5)
            changed = copy.deepcopy(config)
            changed.pop(RESOLVED_TRAINING_DATASET_IDENTITY_KEY)
            changed_identity = copy.deepcopy(config[RESOLVED_TRAINING_DATASET_IDENTITY_KEY])
            changed_identity["content_sha256"] = "e" * 64
            bind_training_dataset_identity(changed, changed_identity)
            changed["resume_from_checkpoint"] = True

            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                resolve_same_run_resume(changed)

    def test_sft_resume_requires_bound_lineage_and_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self._pretraining_parent(root)
            config = self._config(root / "sft-run", stage="sft", use_ema=True)
            initialize_training_run(
                config,
                stage="sft",
                is_world_process_zero=True,
            )
            checkpoint = self._checkpoint(config, 7, parent_lineage=parent)
            resumed = dict(config, resume_from_checkpoint=True)
            self.assertEqual(resolve_same_run_resume(resumed), str(checkpoint.resolve()))

            (checkpoint / "flow_model" / "ema_model.bin").unlink()
            with self.assertRaisesRegex(ValueError, "missing bound artifact.*ema_model.bin"):
                resolve_same_run_resume(resumed)

            (checkpoint / "flow_model" / "ema_model.bin").write_bytes(b"ema-state-7")
            (checkpoint / "flow_model" / "lineage.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing bound artifact.*lineage.json"):
                resolve_same_run_resume(resumed)

    def test_sft_resume_preserves_the_original_pretraining_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self._pretraining_parent(root)
            config = self._config(root / "sft-run", stage="sft", use_ema=False)
            initialize_training_run(
                config,
                stage="sft",
                is_world_process_zero=True,
            )
            checkpoint = self._checkpoint(config, 9, parent_lineage=parent)
            resumed = dict(config, resume_from_checkpoint=True)

            import vyvotts.finetune as finetune_module

            if not hasattr(finetune_module.Trainer, "_load_from_checkpoint"):
                self.skipTest("Transformers training extra is unavailable")
            fine_tuner = object.__new__(finetune_module.EchoDiTFineTuner)
            fine_tuner.config = resumed
            fine_tuner.ema_model = None
            fine_tuner.parent_lineage = None
            fine_tuner.parent_model_manifest = None
            parent_model_manifest = {
                "manifest_sha256": "a" * 64,
                "stage": "pretrain",
                "weights_sha256": "b" * 64,
                "representation_sha256": "c" * 64,
            }
            resumed_model_manifest = SimpleNamespace(
                parent=parent_model_manifest,
                sha256="d" * 64,
            )
            with (
                patch.object(
                    finetune_module.Trainer,
                    "_load_from_checkpoint",
                    return_value="loaded",
                ),
                patch.object(
                    finetune_module,
                    "load_model_manifest",
                    return_value=resumed_model_manifest,
                ),
                patch.object(finetune_module, "validate_manifest_weight"),
                patch.object(finetune_module, "validate_sft_resume_manifest"),
            ):
                result = fine_tuner._load_from_checkpoint(str(checkpoint))

            self.assertEqual(result, "loaded")
            self.assertEqual(fine_tuner.parent_lineage["stage"], "pretrain")
            self.assertEqual(fine_tuner.parent_model_manifest, parent_model_manifest)
            next_export = root / "next-sft-export"
            next_export.mkdir()
            (next_export / "pytorch_model.bin").write_bytes(b"next-sft")
            next_lineage = write_training_lineage(
                next_export,
                resumed,
                stage="sft",
                checkpoint_file="pytorch_model.bin",
                parent_lineage=fine_tuner.parent_lineage,
            )
            self.assertEqual(next_lineage["parent"], fine_tuner.parent_lineage)

    def test_checkpoint_manifest_validation_rejects_unbound_extra_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory) / "run")
            initialize_training_run(
                config,
                stage="pretrain",
                is_world_process_zero=True,
            )
            checkpoint = self._checkpoint(config, 2)
            (checkpoint / "unbound-model.bin").write_bytes(b"unexpected")

            with self.assertRaisesRegex(ValueError, "artifact set"):
                validate_training_checkpoint_manifest(
                    checkpoint,
                    config,
                    stage="pretrain",
                )

    def test_trainer_checkpoint_write_failure_is_synchronized_before_seal_barrier(self):
        import vyvotts.finetune as finetune_module
        import vyvotts.train as train_module

        cases = (
            (train_module, train_module.EchoDiTTrainer, "training_config"),
            (finetune_module, finetune_module.EchoDiTFineTuner, "config"),
        )
        for module, trainer_type, config_attribute in cases:
            with self.subTest(trainer=trainer_type.__name__), tempfile.TemporaryDirectory() as root:
                if not hasattr(module.Trainer, "_save_checkpoint"):
                    self.skipTest("Transformers training extra is unavailable")
                trainer = object.__new__(trainer_type)
                trainer.state = SimpleNamespace(global_step=4)
                setattr(trainer, config_attribute, {})
                trainer._get_output_dir = lambda trial: root
                trainer.is_world_process_zero = lambda: True
                failure = OSError("disk full")

                with (
                    patch.object(module.Trainer, "_save_checkpoint", side_effect=failure),
                    patch.object(
                        module,
                        "propagate_process_group_error",
                        side_effect=failure,
                    ) as propagate,
                    patch.object(module.dist, "is_available", return_value=True),
                    patch.object(module.dist, "is_initialized", return_value=True),
                    patch.object(module.dist, "broadcast_object_list"),
                    patch.object(module.dist, "barrier") as barrier,
                    self.assertRaises(OSError) as raised,
                ):
                    trainer._save_checkpoint(None, None)

                self.assertIs(raised.exception, failure)
                self.assertIs(propagate.call_args.args[0], failure)
                # The first barrier follows seal invalidation; no rank may enter
                # the post-save sealing barrier after one rank's write failed.
                self.assertEqual(barrier.call_count, 1)


if __name__ == "__main__":
    unittest.main()
