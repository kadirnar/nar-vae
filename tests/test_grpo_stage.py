"""CPU-only contracts for the executable, resumable GRPO post-training stage."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from nar_vae.checkpoint import (
    DurationCheckpointInfo,
    LanguageCheckpointInfo,
    MonotonicAlignmentCheckpointInfo,
    ReferenceLanguageCheckpointInfo,
)
from nar_vae.configuration import write_training_lineage
from nar_vae.dataset.finetune_prepare import (
    _content_bound_utterance_id,
    _validate_unique_prompt_ids,
)
from nar_vae.distributed import DistributedContext
from nar_vae.languages import LanguagePair, language_id
from nar_vae.losses.flow_matching_loss import FlowMatchingLoss
from nar_vae.model_manifest import (
    MODEL_MANIFEST_FILENAME,
    ModelManifestError,
    load_model_manifest,
    validate_grpo_parent_manifest,
    write_model_manifest,
)
from nar_vae.models.duration import MONOTONIC_ALIGNMENT_VERSION
from nar_vae.objectives import RECTIFIED_FLOW_OBJECTIVE
from nar_vae.post_training import (
    DEFAULT_GRPO_CONFIG_PATH,
    FlowGRPOConfig,
    FlowGRPOTrainer,
    GRPOStageConfig,
    GRPOStageError,
    GRPOStageRuntime,
    RankPromptSampler,
    bind_reward_evaluator_manifest,
    grpo_reference_identity,
    run_grpo_stage,
)
from nar_vae.post_training import grpo as grpo_module
from nar_vae.post_training import nar_vae_stage as nar_stage_module
from nar_vae.post_training import stage as stage_module
from nar_vae.post_training.nar_vae_stage import (
    NARVAEGRPOCollator,
    _decode_exact_latent_lengths,
    _evaluate_exact_audio_lengths,
    _pad_token_durations_to_batch_frames,
    _preserve_flow_only_trainability,
    _velocity_adapter,
)
from nar_vae.post_training.stage import GRPOPreparedBatch


def model_config(codec_source: str = "./codec/weights.pth") -> dict:
    return {
        "model_preset": "tiny",
        "dacvae_model": codec_source,
        "dacvae_backend": "bundled",
        "dacvae_sample_rate": 44100,
        "dacvae_hop_length": 512,
        "dacvae_latent_dim": 128,
        "dacvae_sha256": "c" * 64,
        "text_vocab_size": 530,
        "target_patch_size": 1,
        "speaker_patch_size": 4,
        "norm_eps": 1e-6,
        "use_speaker_conditioning": False,
        "use_language_conditioning": False,
        "supported_languages": ["en"],
        "supported_reference_languages": None,
        "use_duration_predictor": True,
        "duration_predictor_hidden_size": 256,
        "duration_predictor_num_layers": 2,
        "duration_predictor_use_speaker": False,
        "use_mas_duration": True,
        "duration_alignment_hidden_size": 64,
    }


def frozen_model_config(codec_source: str = "./codec/weights.pth") -> dict:
    config = model_config(codec_source)
    config.update(
        text_conditioning_mode="frozen_features",
        text_num_layers=0,
        text_vocab_size=23,
        pad_token=1,
        conditioning_feature_size=4,
        conditioning_feature_dtype="float16",
        frozen_text_alignment="hf_non_special_tokens_v1",
        frozen_text_cache_version=1,
        frozen_text_config_sha256="a" * 64,
        frozen_text_encoder_id="example/encoder",
        frozen_text_encoder_revision="a" * 40,
        frozen_text_frontend="phonemes",
        frozen_text_hidden_layer=-1,
        frozen_text_model_filename="model.safetensors",
        frozen_text_model_sha256="b" * 64,
        frozen_text_tokenizer_filename="tokenizer.json",
        frozen_text_tokenizer_id="example/tokenizer",
        frozen_text_tokenizer_revision="a" * 40,
        frozen_text_tokenizer_sha256="c" * 64,
    )
    return config


class ToyPromptDataset:
    column_names = ["utterance_id", "conditioning_ids", "latents"]

    def __init__(self, size: int = 2) -> None:
        self.rows = [
            {
                "utterance_id": f"prompt-{index}",
                "conditioning_ids": [index + 1],
                "latents": [[float(index)]],
            }
            for index in range(size)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ToyVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, state, time, conditioning):
        del time, conditioning
        return torch.ones_like(state) * self.scale


def toy_runtime(parent_manifest):
    policy = ToyVelocity()
    reference = ToyVelocity()
    reference.load_state_dict(policy.state_dict())

    def collate(rows):
        return {"prompt_ids": tuple(row["utterance_id"] for row in rows)}

    def prepare(batch, device, group_size, generator):
        batch_size = len(batch["prompt_ids"])
        return GRPOPreparedBatch(
            initial_state=torch.randn(
                batch_size,
                group_size,
                4,
                device=device,
                generator=generator,
            ),
            conditioning=None,
            trainer_batch=batch,
        )

    def velocity(model, state, time, conditioning):
        return model(state, time, conditioning)

    def reward(audio, batch):
        del batch
        return {"quality": audio[..., 0]}

    return GRPOStageRuntime(
        policy=policy,
        reference_policy=reference,
        collate_fn=collate,
        prepare_batch=prepare,
        velocity_adapter=velocity,
        decode=lambda state, batch: state,
        reward=reward,
        model_export_config=model_config(),
        parent_model_manifest=parent_manifest,
    )


def stochastic_replay_runtime(parent_manifest, *, fail_on_reward_call: int | None = None):
    runtime = toy_runtime(parent_manifest)
    reward_calls = 0

    def reward(audio, batch):
        nonlocal reward_calls
        del batch
        reward_calls += 1
        if reward_calls == fail_on_reward_call:
            raise RuntimeError("simulated interruption")
        return {"quality": audio[..., 0]}

    def supervised_loss(model, batch):
        del batch
        target = torch.rand((), device=next(model.parameters()).device)
        return (model.scale - target).square()

    runtime.reward = reward
    runtime.supervised_loss = supervised_loss
    return runtime


def dataset_identity(root: Path, size: int = 2) -> dict:
    return {
        "schema_version": 1,
        "library": "nar-vae",
        "kind": "local-prepared",
        "source": str(root),
        "revision": None,
        "split": None,
        "fingerprint": "toy-prompts-v1",
        "num_rows": size,
        "columns": ["utterance_id", "conditioning_ids", "latents"],
        "content_sha256": "d" * 64,
    }


def stage_config(
    root: Path,
    parent: Path,
    *,
    resume=False,
    max_steps: int = 1,
    save_folder: str = "run",
    supervised_replay_weight: float = 0.0,
) -> GRPOStageConfig:
    return GRPOStageConfig(
        parent_checkpoint=parent.resolve(),
        prompt_dataset_local=(root / "dataset").resolve(),
        save_folder=(root / save_folder).resolve(),
        resume_from_checkpoint=resume,
        reward_weights={"quality": 1.0},
        reward_evaluators={
            "quality": {
                "implementation": "tests.toy_quality",
                "revision": "v1",
                "sha256": "e" * 64,
            }
        },
        epochs=1,
        max_steps=max_steps,
        prompt_batch_size=1,
        learning_rate=0.05,
        save_steps=1,
        logging_steps=1,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        mixed_precision="fp32",
        report_to="wandb",
        num_steps=4,
        group_size=2,
        sde_window_start=1,
        sde_window_size=2,
        supervised_replay_weight=supervised_replay_weight,
        policy_update_epochs=2,
    )


def write_sft_parent(root: Path, *, export_config: dict | None = None):
    export_config = model_config() if export_config is None else export_config
    pretrain = root / "pretrain"
    pretrain.mkdir()
    pretrain_weights = pretrain / "pytorch_model.bin"
    pretrain_weights.write_bytes(b"pretraining weights")
    pretrain_config = {"training_stage": "pretrain", "name": "toy"}
    pretrain_lineage = write_training_lineage(
        pretrain,
        pretrain_config,
        stage="pretrain",
        checkpoint_file=pretrain_weights.name,
    )
    pretrain_manifest = write_model_manifest(
        pretrain,
        export_config,
        stage="pretrain",
        checkpoint_files=(pretrain_weights.name,),
    )

    sft = root / "sft"
    sft.mkdir()
    sft_weights = sft / "pytorch_model.bin"
    torch.save(ToyVelocity().state_dict(), sft_weights)
    write_training_lineage(
        sft,
        {"training_stage": "sft", "name": "toy"},
        stage="sft",
        checkpoint_file=sft_weights.name,
        parent_lineage=pretrain_lineage,
    )
    sft_manifest = write_model_manifest(
        sft,
        export_config,
        stage="sft",
        checkpoint_files=(sft_weights.name,),
        parent_manifest=pretrain_manifest,
    )
    return sft_weights, sft_manifest


def matching_grpo_capability_contract():
    checkpoint = Mock()
    checkpoint.validate_architecture.return_value = 4
    checkpoint.infer_target_patch_size.return_value = 1
    checkpoint.infer_speaker_num_summary_tokens.return_value = 0
    checkpoint.infer_speaker_conditioning.return_value = False
    checkpoint.language_capability.return_value = LanguageCheckpointInfo(enabled=False)
    checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(
        enabled=False
    )
    checkpoint.duration_capability.return_value = DurationCheckpointInfo(
        enabled=True,
        hidden_size=256,
        num_layers=2,
        uses_speaker=False,
    )
    checkpoint.monotonic_alignment_capability.return_value = MonotonicAlignmentCheckpointInfo(
        enabled=True,
        hidden_size=64,
        version=MONOTONIC_ALIGNMENT_VERSION,
    )
    manifest = SimpleNamespace(
        architecture={
            "speaker_patch_size": 4,
            "speaker_num_summary_tokens": 0,
            "target_patch_size": 1,
        },
        capabilities={
            "speaker_conditioning": False,
            "language_conditioning": False,
            "supported_languages": ["en"],
            "supported_language_pairs": [],
            "duration_predictor": True,
            "duration_predictor_hidden_size": 256,
            "duration_predictor_num_layers": 2,
            "duration_predictor_use_speaker": False,
            "monotonic_alignment": True,
            "duration_alignment_hidden_size": 64,
        },
    )
    return checkpoint, manifest


class GRPOStageTest(unittest.TestCase):
    def setUp(self):
        class NoOpWandbLogger:
            def log(self, metrics, *, step):
                del metrics, step

            def finish(self):
                return None

        # Exercise the training loop without importing W&B or making network calls.
        self._wandb_logger_patch = patch.object(
            stage_module,
            "_WandbLogger",
            return_value=NoOpWandbLogger(),
        )
        self._wandb_logger_patch.start()
        self.addCleanup(self._wandb_logger_patch.stop)

    def test_text_only_manifest_does_not_pass_empty_reference_capabilities_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest = write_sft_parent(Path(directory))
            expected = nn.Identity()

            with patch.object(
                nar_stage_module,
                "create_flow_matching_echodit",
                return_value=expected,
            ) as create_model:
                model = nar_stage_module._new_model_from_manifest(manifest)

            self.assertIs(model, expected)
            self.assertEqual(create_model.call_args.kwargs["speaker_num_summary_tokens"], 0)
            self.assertEqual(
                nar_stage_module.model_export_config_from_manifest(manifest)[
                    "speaker_num_summary_tokens"
                ],
                0,
            )
            self.assertIsNone(create_model.call_args.kwargs["supported_reference_languages"])
            self.assertIsNone(create_model.call_args.kwargs["supported_language_pairs"])

    def test_grpo_rejects_checkpoint_pair_metadata_that_disagrees_with_manifest(self):
        checkpoint = Mock()
        checkpoint.validate_architecture.return_value = 4
        checkpoint.infer_target_patch_size.return_value = 1
        checkpoint.infer_speaker_num_summary_tokens.return_value = 0
        checkpoint.infer_speaker_conditioning.return_value = True
        checkpoint.infer_speaker_patch_size.return_value = 4
        checkpoint.language_capability.return_value = LanguageCheckpointInfo(
            enabled=True,
            supported_languages=("en", "es"),
        )
        checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(
            enabled=True,
            supported_languages=("en",),
            supported_pairs=(LanguagePair("es", "en"),),
        )
        manifest = SimpleNamespace(
            architecture={
                "speaker_patch_size": 4,
                "speaker_num_summary_tokens": 0,
                "target_patch_size": 1,
            },
            capabilities={
                "speaker_conditioning": True,
                "language_conditioning": True,
                "supported_languages": ["en", "es"],
                "supported_language_pairs": [["en", "en"]],
            },
        )

        with self.assertRaisesRegex(ModelManifestError, "language pairs do not match"):
            nar_stage_module._validate_grpo_checkpoint_capabilities(checkpoint, manifest)

    def test_grpo_rejects_speaker_summary_topology_mismatch(self):
        checkpoint, manifest = matching_grpo_capability_contract()
        checkpoint.infer_speaker_num_summary_tokens.return_value = 8

        with self.assertRaisesRegex(ModelManifestError, "summary-token topology"):
            nar_stage_module._validate_grpo_checkpoint_capabilities(checkpoint, manifest)

    def test_grpo_rejects_duration_capability_mismatches(self):
        mismatches = {
            "enabled": DurationCheckpointInfo(enabled=False),
            "hidden_size": DurationCheckpointInfo(True, 128, 2, False),
            "num_layers": DurationCheckpointInfo(True, 256, 3, False),
            "uses_speaker": DurationCheckpointInfo(True, 256, 2, True),
        }
        for name, observed in mismatches.items():
            with self.subTest(field=name):
                checkpoint, manifest = matching_grpo_capability_contract()
                checkpoint.duration_capability.return_value = observed

                with self.assertRaisesRegex(ModelManifestError, "duration-predictor"):
                    nar_stage_module._validate_grpo_checkpoint_capabilities(
                        checkpoint,
                        manifest,
                    )

    def test_grpo_rejects_monotonic_alignment_capability_mismatches(self):
        mismatches = {
            "enabled": MonotonicAlignmentCheckpointInfo(enabled=False),
            "hidden_size": MonotonicAlignmentCheckpointInfo(
                True,
                32,
                MONOTONIC_ALIGNMENT_VERSION,
            ),
            "version": MonotonicAlignmentCheckpointInfo(True, 64, 999),
        }
        for name, observed in mismatches.items():
            with self.subTest(field=name):
                checkpoint, manifest = matching_grpo_capability_contract()
                checkpoint.monotonic_alignment_capability.return_value = observed

                with self.assertRaisesRegex(ModelManifestError, "monotonic-alignment"):
                    nar_stage_module._validate_grpo_checkpoint_capabilities(
                        checkpoint,
                        manifest,
                    )

    def test_runtime_revalidates_sft_bytes_immediately_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, parent_manifest = write_sft_parent(root)
            config = stage_config(root, parent)

            def reward(audio, batch):
                del audio, batch
                return {}

            bind_reward_evaluator_manifest(reward, config.reward_evaluators)
            original_validate = nar_stage_module.validate_manifest_weight
            original_load = torch.load
            events = []

            def validate(manifest, path, *, selected_filename=None):
                events.append(("validate", Path(path).resolve(), selected_filename))
                return original_validate(
                    manifest,
                    path,
                    selected_filename=selected_filename,
                )

            def deserialize(*args, **kwargs):
                events.append(("deserialize", Path(args[0]).resolve()))
                return original_load(*args, **kwargs)

            with (
                patch.object(
                    nar_stage_module,
                    "_new_model_from_manifest",
                    side_effect=(ToyVelocity(), ToyVelocity()),
                ),
                patch.object(
                    nar_stage_module,
                    "_validate_grpo_checkpoint_capabilities",
                ),
                patch.object(nar_stage_module, "validate_manifest_weight", side_effect=validate),
                patch.object(nar_stage_module, "validate_loaded_codec"),
                patch("nar_vae.checkpoint.torch.load", side_effect=deserialize),
            ):
                runtime = nar_stage_module.build_nar_vae_grpo_runtime(
                    config,
                    parent_manifest=parent_manifest,
                    reward=reward,
                    device=torch.device("cpu"),
                    codec=nn.Identity(),
                )

            self.assertIsInstance(runtime, GRPOStageRuntime)
            self.assertEqual(
                events,
                [
                    ("validate", parent.resolve(), parent.name),
                    ("deserialize", parent.resolve()),
                ],
            )

    def test_frozen_grpo_collator_uses_authenticated_provider_pad(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, parent_manifest = write_sft_parent(
                root,
                export_config=frozen_model_config(),
            )
            config = stage_config(root, parent)

            def reward(audio, batch):
                del audio, batch
                return {"quality": torch.zeros(1, config.group_size)}

            bind_reward_evaluator_manifest(reward, config.reward_evaluators)
            checkpoint = Mock()
            checkpoint.generative_objective.return_value = RECTIFIED_FLOW_OBJECTIVE
            checkpoint.load_into.side_effect = lambda model: None
            with (
                patch.object(
                    nar_stage_module,
                    "_new_model_from_manifest",
                    side_effect=(ToyVelocity(), ToyVelocity()),
                ),
                patch.object(
                    nar_stage_module.FlowCheckpoint,
                    "load",
                    return_value=checkpoint,
                ),
                patch.object(nar_stage_module, "_validate_grpo_checkpoint_capabilities"),
                patch.object(nar_stage_module, "validate_loaded_codec"),
            ):
                runtime = nar_stage_module.build_nar_vae_grpo_runtime(
                    config,
                    parent_manifest=parent_manifest,
                    reward=reward,
                    device=torch.device("cpu"),
                    codec=nn.Identity(),
                )

            english = language_id("en")
            rows = []
            for utterance, token_ids in (("a", [0, 5, 2]), ("b", [0, 6, 7, 2])):
                rows.append(
                    {
                        "utterance_id": utterance,
                        "latents": torch.zeros(128, 3),
                        "conditioning_ids": token_ids,
                        "conditioning_features": torch.zeros(
                            len(token_ids), 4, dtype=torch.float16
                        ),
                        "conditioning_feature_dtype": "float16",
                        "token_language_ids": [0, *([english] * (len(token_ids) - 2)), 0],
                        "alignment_mask": [False, *([True] * (len(token_ids) - 2)), False],
                        "language": "en",
                    }
                )
            batch = runtime.collate_fn(rows)
            self.assertEqual(batch["model_inputs"]["conditioning_ids"][0, -1].item(), 1)

            with self.assertRaisesRegex(ValueError, "authenticated parent"):
                nar_stage_module.build_nar_vae_grpo_runtime(
                    config,
                    parent_manifest=parent_manifest,
                    reward=reward,
                    device=torch.device("cpu"),
                    codec=nn.Identity(),
                    pad_token=0,
                )

    def test_runtime_rejects_parent_or_sparse_ema_base_tampering_before_torch_load(self):
        for tamper_base in (False, True):
            with self.subTest(tamper_base=tamper_base), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                parent, parent_manifest = write_sft_parent(root)
                selected = parent
                if tamper_base:
                    selected = parent.with_name("ema_model.bin")
                    torch.save({"scale": torch.tensor(0.2)}, selected)
                    parent_manifest = write_model_manifest(
                        parent.parent,
                        model_config(),
                        stage="sft",
                        checkpoint_files=(parent.name, selected.name),
                        parent_manifest=load_model_manifest(
                            root / "pretrain" / MODEL_MANIFEST_FILENAME
                        ),
                    )
                config = stage_config(root, selected)

                def reward(audio, batch):
                    del audio, batch
                    return {}

                bind_reward_evaluator_manifest(reward, config.reward_evaluators)
                (parent if tamper_base else selected).write_bytes(b"tampered after startup check")
                with (
                    patch.object(
                        nar_stage_module,
                        "_new_model_from_manifest",
                        side_effect=(ToyVelocity(), ToyVelocity()),
                    ),
                    patch("nar_vae.checkpoint.torch.load") as deserialize,
                    self.assertRaisesRegex(ModelManifestError, "does not match"),
                ):
                    nar_stage_module.build_nar_vae_grpo_runtime(
                        config,
                        parent_manifest=parent_manifest,
                        reward=reward,
                        device=torch.device("cpu"),
                        codec=nn.Identity(),
                    )
                deserialize.assert_not_called()

    def test_public_stage_initializes_distributed_before_propagating_startup_failure(self):
        events = []
        context = DistributedContext(0, 0, 1, 1)

        def initialize():
            events.append("initialize")
            return context

        def load_config(path):
            del path
            events.append("config")
            raise ValueError("bad config")

        def propagate(process, error, *, description):
            self.assertIs(process, context)
            events.append(f"propagate:{description}")
            if error is not None:
                raise error

        with (
            patch.object(nar_stage_module, "initialize_distributed", side_effect=initialize),
            patch.object(nar_stage_module, "load_grpo_stage_config", side_effect=load_config),
            patch.object(nar_stage_module, "propagate_distributed_error", side_effect=propagate),
            self.assertRaisesRegex(ValueError, "bad config"),
        ):
            nar_stage_module._grpo_post_train("bad.yaml", reward=lambda audio, batch: {})
        self.assertEqual(
            events,
            [
                "initialize",
                "propagate:GRPO device initialization",
                "config",
                "propagate:GRPO startup validation",
            ],
        )

    def test_wandb_finish_failure_does_not_mask_training_failure(self):
        class FailingLogger:
            def log(self, metrics, *, step):
                del metrics, step

            def finish(self):
                raise RuntimeError("finish failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            parent, parent_manifest = write_sft_parent(root)
            reference = grpo_reference_identity(parent, parent_manifest)
            runtime = toy_runtime(parent_manifest)

            def fail_reward(audio, batch):
                del audio, batch
                raise RuntimeError("training failure")

            runtime.reward = fail_reward
            with (
                patch.object(stage_module, "_WandbLogger", return_value=FailingLogger()),
                self.assertRaisesRegex(RuntimeError, "training failure"),
            ):
                run_grpo_stage(
                    stage_config(root, parent),
                    runtime=runtime,
                    dataset=ToyPromptDataset(),
                    dataset_identity=dataset_identity(root / "dataset"),
                    reference_identity=reference,
                    context=DistributedContext(0, 0, 1, 1),
                    device=torch.device("cpu"),
                )

    def test_packaged_config_and_public_reward_binding(self):
        config_path = Path(DEFAULT_GRPO_CONFIG_PATH)
        self.assertTrue(config_path.is_file())
        config_text = config_path.read_text(encoding="utf-8")
        self.assertIn(
            "final/flow_model/pytorch_model.bin",
            config_text,
        )
        self.assertIn('optimizer: "adamw"', config_text)
        self.assertIn('# optimizer: "muon"', config_text)
        self.assertIn('report_to: "wandb"', config_text)

        def reward(audio, batch):
            del audio, batch
            return {}

        evaluators = {
            "quality": {
                "implementation": "tests.quality",
                "revision": "v1",
                "sha256": "a" * 64,
            }
        }
        self.assertIs(bind_reward_evaluator_manifest(reward, evaluators), reward)
        self.assertEqual(reward.nar_vae_reward_evaluators, evaluators)

    def test_collective_verdict_precedes_reward_statistics_and_each_next_forward(self):
        events = []
        policy = ToyVelocity()
        reference = copy.deepcopy(policy)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.05)
        original_combine = grpo_module.combine_reward_components

        def velocity(model, state, time, conditioning):
            events.append("forward")
            return model(state, time, conditioning)

        def synchronize(error, description):
            events.append(f"sync:{description}")
            if error is not None:
                raise error

        def combine(*args, **kwargs):
            events.append("combine")
            return original_combine(*args, **kwargs)

        trainer = FlowGRPOTrainer(
            policy=policy,
            reference_policy=reference,
            optimizer=optimizer,
            velocity_adapter=velocity,
            decode=lambda state, batch: state,
            reward=lambda audio, batch: {"quality": audio[..., 0]},
            reward_weights={"quality": 1.0},
            config=FlowGRPOConfig(
                num_steps=4,
                group_size=2,
                sde_window_start=1,
                sde_window_size=2,
                policy_update_epochs=2,
            ),
            distributed_error_synchronizer=synchronize,
        )
        with patch("nar_vae.post_training.grpo.combine_reward_components", side_effect=combine):
            trainer.step(
                initial_state=torch.randn(1, 2, 4),
                conditioning=None,
                batch=None,
                generator=torch.Generator().manual_seed(4),
            )

        reward_verdict = events.index("sync:GRPO decode/reward validation")
        self.assertLess(reward_verdict, events.index("combine"))
        transition_verdicts = [
            event for event in events if event.startswith("sync:") and "transition step" in event
        ]
        self.assertEqual(len(transition_verdicts), events.count("forward"))
        for index, event in enumerate(events[:-1]):
            if event == "forward" and events[index + 1] == "forward":
                self.fail("A second DDP-capable forward started before a rank verdict.")

    def test_exact_length_decode_and_reward_never_observe_padding(self):
        class Codec(nn.Module):
            def __init__(self):
                super().__init__()
                self.lengths = []

            def decode(self, latents):
                self.lengths.append(latents.shape[-1])
                return torch.ones(latents.shape[0], latents.shape[-1] * 2)

        codec = Codec()
        audio, sample_lengths = _decode_exact_latent_lengths(
            codec,
            torch.randn(2, 3, 1, 5),
            torch.tensor([3, 5]),
        )
        self.assertEqual(codec.lengths, [3, 5])
        self.assertEqual(sample_lengths.tolist(), [6, 10])
        observed = []

        def reward(exact_audio, batch):
            observed.append(exact_audio.shape[-1])
            return {"quality": torch.ones(1, 3)}

        components = _evaluate_exact_audio_lengths(
            reward,
            audio,
            {
                "prompt_ids": ("a", "b"),
                "reward_rows": ({}, {}),
                "latent_lengths": torch.tensor([3, 5]),
                "sample_lengths": sample_lengths,
                "model_inputs": {"latents": torch.randn(2, 1, 5)},
            },
            component_names=("quality",),
            group_size=3,
        )
        self.assertEqual(observed, [6, 10])
        self.assertEqual(components["quality"].shape, (2, 3))

    def test_collator_keeps_each_prompt_id_paired_with_its_reward_row(self):
        collator = NARVAEGRPOCollator(
            pad_token=0,
            speaker_patch_size=1,
            prompt_id_column="utterance_id",
        )
        collator.base = lambda rows: {
            "latent_mask": torch.tensor([[True, False], [True, True]]),
        }
        batch = collator(
            [
                {
                    "utterance_id": "prompt-a",
                    "transcript": "first",
                    "latents": [[1.0]],
                },
                {
                    "utterance_id": "prompt-b",
                    "transcript": "second",
                    "latents": [[2.0], [3.0]],
                },
            ]
        )

        self.assertEqual(batch["prompt_ids"], ("prompt-a", "prompt-b"))
        self.assertEqual(
            tuple(row["transcript"] for row in batch["reward_rows"]),
            ("first", "second"),
        )
        self.assertEqual(len(batch["prompt_ids"]), len(batch["reward_rows"]))
        self.assertNotIn("latents", batch["reward_rows"][0])
        self.assertEqual(batch["latent_lengths"].tolist(), [1, 2])

    def test_collator_preserves_full_v2_text_metadata_and_separate_global_language(self):
        collator = NARVAEGRPOCollator(
            pad_token=0,
            speaker_patch_size=1,
            prompt_id_column="utterance_id",
        )
        english = language_id("en")
        turkish = language_id("tr")
        batch = collator(
            [
                {
                    "utterance_id": "prompt-a",
                    "latents": torch.zeros(1, 3),
                    "conditioning_ids": [10, 11, 12, 13, 14],
                    "token_language_ids": [0, english, turkish, english, 0],
                    "alignment_mask": [False, True, True, True, False],
                    "language": "en",
                },
                {
                    "utterance_id": "prompt-b",
                    "latents": torch.zeros(1, 4),
                    "conditioning_ids": [20, 21, 22],
                    "token_language_ids": [0, turkish, 0],
                    "alignment_mask": [False, True, False],
                    "language": "tr",
                },
            ]
        )

        inputs = batch["model_inputs"]
        self.assertEqual(inputs["conditioning_ids"].shape, (2, 5))
        self.assertEqual(inputs["token_language_ids"].shape, (2, 5))
        self.assertEqual(inputs["alignment_mask"].shape, (2, 5))
        self.assertEqual(inputs["language_ids"].shape, (2,))
        torch.testing.assert_close(inputs["language_ids"], torch.tensor([english, turkish]))
        torch.testing.assert_close(
            inputs["token_language_ids"][0],
            torch.tensor([0, english, turkish, english, 0]),
        )
        self.assertTrue((inputs["token_language_ids"][1, 3:] == 0).all())
        self.assertFalse(inputs["alignment_mask"][1, 3:].any())

    def test_flow_only_trainability_and_mas_replay_use_fixed_durations(self):
        class DummyDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.flow = nn.Linear(1, 1)
                self.prefix_trainable = True

            def set_latent_prefix_trainable(self, enabled):
                self.prefix_trainable = enabled

        policy = nn.Module()
        policy.duration_predictor = nn.Linear(1, 1)
        policy.duration_alignment = nn.Linear(1, 1)
        policy.dit = DummyDiT()
        _preserve_flow_only_trainability(policy)
        self.assertFalse(any(p.requires_grad for p in policy.duration_predictor.parameters()))
        self.assertFalse(any(p.requires_grad for p in policy.duration_alignment.parameters()))
        self.assertTrue(all(p.requires_grad for p in policy.dit.flow.parameters()))
        self.assertFalse(policy.dit.prefix_trainable)

        class CaptureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.token_durations = None
                self.last_call = None

            def forward(self, **kwargs):
                self.token_durations = kwargs.get("token_durations")
                self.last_call = kwargs
                return kwargs["latents"] * self.weight

        model = CaptureModel()
        english = language_id("en")
        turkish = language_id("tr")
        durations = torch.tensor([[0, 2, 0, 3, 0], [0, 1, 2, 0, 2]])
        token_language_ids = torch.tensor(
            [[0, english, 0, turkish, 0], [0, turkish, english, 0, english]]
        )
        alignment_mask = torch.tensor(
            [[False, True, False, True, False], [False, True, True, False, True]]
        )
        global_language_ids = torch.tensor([english, turkish])
        conditioning = {
            "conditioning_ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
            "conditioning_mask": torch.ones(2, 5, dtype=torch.bool),
            "token_language_ids": token_language_ids,
            "alignment_mask": alignment_mask,
            "language_ids": global_language_ids,
            "token_durations": durations,
        }
        _velocity_adapter(
            model,
            torch.randn(2, 3, 1, 5),
            torch.rand(2, 3),
            conditioning,
        )
        self.assertEqual(model.token_durations.shape, (6, 5))
        torch.testing.assert_close(model.token_durations[::3], durations)
        self.assertEqual(model.last_call["language_ids"].shape, (6,))
        self.assertEqual(model.last_call["token_language_ids"].shape, (6, 5))
        self.assertEqual(model.last_call["alignment_mask"].shape, (6, 5))
        torch.testing.assert_close(model.last_call["language_ids"][::3], global_language_ids)
        torch.testing.assert_close(model.last_call["token_language_ids"][::3], token_language_ids)
        torch.testing.assert_close(model.last_call["alignment_mask"][::3], alignment_mask)

        loss = FlowMatchingLoss(timestep_distribution="uniform")(
            model,
            torch.randn(2, 1, 5),
            conditioning["conditioning_ids"],
            token_durations=durations,
            language_ids=global_language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
        )
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(model.token_durations, durations)
        torch.testing.assert_close(model.last_call["language_ids"], global_language_ids)
        torch.testing.assert_close(model.last_call["token_language_ids"], token_language_ids)
        torch.testing.assert_close(model.last_call["alignment_mask"], alignment_mask)

        padded = _pad_token_durations_to_batch_frames(
            torch.tensor([[0, 1, 0, 2, 0], [0, 1, 1, 0, 2]]),
            torch.tensor([[False, True, False, True, False], [False, True, True, False, True]]),
            torch.tensor([3, 4]),
            padded_frames=5,
        )
        torch.testing.assert_close(
            padded,
            torch.tensor([[0, 1, 0, 4, 0], [0, 1, 1, 0, 3]]),
        )
        torch.testing.assert_close(padded.sum(dim=1), torch.tensor([5, 5]))
        with self.assertRaisesRegex(ValueError, "Non-alignable"):
            _pad_token_durations_to_batch_frames(
                torch.tensor([[1, 1, 0]]),
                torch.tensor([[False, True, False]]),
                torch.tensor([2]),
                padded_frames=2,
            )

    def test_runtime_threads_v2_metadata_through_fixed_duration_and_replay(self):
        class CapturingModel(nn.Module):
            def __init__(self, duration_plan):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))
                self.duration_plan = duration_plan
                self.duration_call = None
                self.forward_calls = []

            def predict_token_duration_frames(self, conditioning_ids, **kwargs):
                self.duration_call = (conditioning_ids, kwargs)
                return self.duration_plan.to(conditioning_ids.device)

            def forward(self, **kwargs):
                self.forward_calls.append(kwargs)
                return kwargs["latents"] * self.weight

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, parent_manifest = write_sft_parent(root)
            config = replace(
                stage_config(root, parent),
                supervised_replay_weight=0.2,
            )

            def reward(audio, batch):
                del audio, batch
                return {"quality": torch.zeros(1, config.group_size)}

            bind_reward_evaluator_manifest(reward, config.reward_evaluators)
            duration_plan = torch.tensor(
                [[0, 1, 0, 2, 0], [0, 2, 2, 0, 0]],
                dtype=torch.long,
            )
            policy = CapturingModel(duration_plan)
            reference = CapturingModel(duration_plan)
            checkpoint = Mock()
            checkpoint.generative_objective.return_value = RECTIFIED_FLOW_OBJECTIVE
            checkpoint.load_into.side_effect = lambda model: None
            with (
                patch.object(
                    nar_stage_module,
                    "_new_model_from_manifest",
                    side_effect=(policy, reference),
                ),
                patch.object(
                    nar_stage_module.FlowCheckpoint,
                    "load",
                    return_value=checkpoint,
                ),
                patch.object(nar_stage_module, "_validate_grpo_checkpoint_capabilities"),
                patch.object(nar_stage_module, "validate_loaded_codec"),
            ):
                runtime = nar_stage_module.build_nar_vae_grpo_runtime(
                    config,
                    parent_manifest=parent_manifest,
                    reward=reward,
                    device=torch.device("cpu"),
                    codec=nn.Identity(),
                )

            english = language_id("en")
            turkish = language_id("tr")
            batch = runtime.collate_fn(
                [
                    {
                        "utterance_id": "prompt-a",
                        "latents": torch.zeros(1, 3),
                        "conditioning_ids": [10, 11, 12, 13, 14],
                        "token_language_ids": [0, english, 0, turkish, 0],
                        "alignment_mask": [False, True, False, True, False],
                        "language": "en",
                    },
                    {
                        "utterance_id": "prompt-b",
                        "latents": torch.zeros(1, 4),
                        "conditioning_ids": [20, 21, 22, 23],
                        "token_language_ids": [0, turkish, english, 0],
                        "alignment_mask": [False, True, True, False],
                        "language": "tr",
                    },
                ]
            )
            prepared = runtime.prepare_batch(
                batch,
                torch.device("cpu"),
                config.group_size,
                torch.Generator().manual_seed(7),
            )

            duration_ids, duration_kwargs = reference.duration_call
            inputs = prepared.conditioning
            torch.testing.assert_close(duration_ids, inputs["conditioning_ids"])
            torch.testing.assert_close(
                duration_kwargs["token_language_ids"],
                inputs["token_language_ids"],
            )
            torch.testing.assert_close(
                duration_kwargs["alignment_mask"],
                inputs["alignment_mask"],
            )
            torch.testing.assert_close(duration_kwargs["language_ids"], inputs["language_ids"])
            self.assertEqual(duration_kwargs["language_ids"].shape, (2,))
            self.assertEqual(duration_kwargs["token_language_ids"].shape, (2, 5))
            torch.testing.assert_close(
                inputs["token_durations"],
                torch.tensor([[0, 1, 0, 3, 0], [0, 2, 2, 0, 0]]),
            )

            runtime.velocity_adapter(
                policy,
                prepared.initial_state,
                torch.rand(2, config.group_size),
                prepared.conditioning,
            )
            rollout_call = policy.forward_calls[-1]
            self.assertEqual(rollout_call["language_ids"].shape, (4,))
            self.assertEqual(rollout_call["token_language_ids"].shape, (4, 5))
            self.assertEqual(rollout_call["alignment_mask"].shape, (4, 5))
            torch.testing.assert_close(
                rollout_call["token_language_ids"][:: config.group_size],
                inputs["token_language_ids"],
            )
            torch.testing.assert_close(
                rollout_call["alignment_mask"][:: config.group_size],
                inputs["alignment_mask"],
            )

            self.assertIsNotNone(runtime.supervised_loss)
            replay_loss = runtime.supervised_loss(policy, prepared.trainer_batch)
            self.assertTrue(torch.isfinite(replay_loss))
            replay_call = policy.forward_calls[-1]
            torch.testing.assert_close(
                replay_call["token_language_ids"],
                inputs["token_language_ids"],
            )
            torch.testing.assert_close(replay_call["alignment_mask"], inputs["alignment_mask"])
            torch.testing.assert_close(replay_call["language_ids"], inputs["language_ids"])

    def test_content_bound_prompt_ids_are_stable_and_duplicates_fail(self):
        row = {
            "latents": torch.arange(6, dtype=torch.float32).reshape(2, 3).numpy(),
            "conditioning_ids": [1, 2],
            "language": "en",
            "text": "hello",
        }
        first = _content_bound_utterance_id(row)
        self.assertEqual(first, _content_bound_utterance_id(copy.deepcopy(row)))
        with self.assertRaisesRegex(ValueError, "unique"):
            _validate_unique_prompt_ids([{"utterance_id": first}, {"utterance_id": first}])

    def test_fixed_rollout_is_reused_after_policy_changes(self):
        policy = ToyVelocity()
        reference = copy.deepcopy(policy)
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        optimizer_steps = []
        trainer = FlowGRPOTrainer(
            policy=policy,
            reference_policy=reference,
            optimizer=optimizer,
            velocity_adapter=lambda model, state, time, conditioning: model(
                state, time, conditioning
            ),
            decode=lambda state, batch: state,
            reward=lambda audio, batch: {"quality": audio[..., 0]},
            reward_weights={"quality": 1.0},
            config=FlowGRPOConfig(
                num_steps=4,
                group_size=2,
                sde_window_start=1,
                sde_window_size=2,
                policy_update_epochs=2,
            ),
            optimizer_step_callback=lambda: optimizer_steps.append(True),
        )

        metrics = trainer.step(
            initial_state=torch.randn(2, 2, 4),
            conditioning=None,
            batch=None,
            generator=torch.Generator().manual_seed(2),
        )

        self.assertEqual(len(optimizer_steps), 2)
        self.assertGreater(metrics.mean_abs_log_ratio, 0.0)

    def test_sampler_keeps_groups_whole_and_disjoint(self):
        first = RankPromptSampler(10, rank=0, world_size=2, seed=7)
        second = RankPromptSampler(10, rank=1, world_size=2, seed=7)
        self.assertFalse(set(first).intersection(second))
        self.assertEqual(set(first).union(second), set(range(10)))
        first.set_epoch(1)
        self.assertNotEqual(list(first), list(RankPromptSampler(10, rank=0, world_size=2, seed=7)))

    def test_config_requires_multiple_policy_epochs_and_exact_reward_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.bin"
            with self.assertRaisesRegex(ValueError, "policy_update_epochs"):
                GRPOStageConfig(
                    parent_checkpoint=parent.resolve(),
                    prompt_dataset_local=(root / "data").resolve(),
                    save_folder=(root / "run").resolve(),
                    reward_weights={"quality": 1.0},
                    reward_evaluators={
                        "quality": {
                            "implementation": "toy",
                            "revision": "v1",
                            "sha256": "a" * 64,
                        }
                    },
                    policy_update_epochs=1,
                )
            with self.assertRaisesRegex(ValueError, "identify every"):
                GRPOStageConfig(
                    parent_checkpoint=parent.resolve(),
                    prompt_dataset_local=(root / "data").resolve(),
                    save_folder=(root / "run").resolve(),
                    reward_weights={"quality": 1.0},
                    reward_evaluators={},
                )

            with self.assertRaisesRegex(ValueError, "W&B logging is mandatory"):
                GRPOStageConfig(
                    parent_checkpoint=parent.resolve(),
                    prompt_dataset_local=(root / "data").resolve(),
                    save_folder=(root / "run").resolve(),
                    reward_weights={"quality": 1.0},
                    reward_evaluators={
                        "quality": {
                            "implementation": "toy",
                            "revision": "v1",
                            "sha256": "a" * 64,
                        }
                    },
                    report_to="none",
                )

    def test_config_strictly_validates_grpo_muon_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = stage_config(root, root / "parent.bin")
            self.assertEqual(base.optimizer, "adamw")
            self.assertIsNone(base.muon_learning_rate)
            self.assertIsNone(base.muon_weight_decay)
            self.assertEqual(base.muon_momentum, 0.95)
            self.assertTrue(base.muon_nesterov)
            self.assertEqual(base.muon_ns_steps, 5)
            self.assertEqual(base.muon_epsilon, 1e-7)
            self.assertEqual(base.muon_adjust_lr_fn, "match_rms_adamw")

            configured = replace(
                base,
                optimizer="muon",
                muon_learning_rate=2e-6,
                muon_weight_decay=0.02,
                muon_momentum=0.9,
                muon_nesterov=False,
                muon_ns_steps=4,
                muon_epsilon=2e-7,
                muon_adjust_lr_fn="original",
            )
            self.assertEqual(configured.optimizer, "muon")
            self.assertEqual(configured.muon_learning_rate, 2e-6)
            self.assertNotEqual(configured.sha256, base.sha256)
            restored = GRPOStageConfig.from_mapping(configured.manifest_payload())
            self.assertEqual(restored.sha256, configured.sha256)
            with self.assertRaisesRegex(ValueError, "require optimizer: muon"):
                replace(base, muon_momentum=0.9)

            invalid_options = (
                ("optimizer", "sgd", "optimizer"),
                ("muon_learning_rate", 0.0, "muon_learning_rate"),
                ("muon_learning_rate", float("inf"), "finite"),
                ("muon_weight_decay", -0.01, "muon_weight_decay"),
                ("muon_momentum", 1.0, "muon_momentum"),
                ("muon_momentum", True, "finite"),
                ("muon_nesterov", 1, "muon_nesterov"),
                ("muon_ns_steps", 0, "muon_ns_steps"),
                ("muon_ns_steps", 100, "muon_ns_steps"),
                ("muon_ns_steps", True, "muon_ns_steps"),
                ("muon_epsilon", 0.0, "muon_epsilon"),
                ("muon_adjust_lr_fn", "none", "muon_adjust_lr_fn"),
            )
            for field_name, value, message in invalid_options:
                validation_base = (
                    base if field_name == "optimizer" else replace(base, optimizer="muon")
                )
                with (
                    self.subTest(field=field_name, value=value),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    replace(validation_base, **{field_name: value})

    def test_grpo_muon_builder_receives_explicit_options_and_single_scheduler_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            parent, parent_manifest = write_sft_parent(root)
            reference = grpo_reference_identity(parent, parent_manifest)
            runtime = toy_runtime(parent_manifest)
            config = replace(
                stage_config(root, parent),
                optimizer="muon",
                learning_rate=4e-5,
                weight_decay=0.03,
                adam_beta1=0.8,
                adam_beta2=0.9,
                adam_epsilon=2e-8,
                muon_learning_rate=3e-5,
                muon_weight_decay=0.02,
                muon_momentum=0.9,
                muon_nesterov=False,
                muon_ns_steps=4,
                muon_epsilon=2e-7,
                muon_adjust_lr_fn="original",
            )
            captured = {}

            def build_muon(model, mapping):
                self.assertIs(model, runtime.policy)
                captured.update(mapping)
                return torch.optim.AdamW(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    lr=mapping["learning_rate"],
                    weight_decay=mapping["weight_decay"],
                )

            with patch.object(
                stage_module,
                "build_muon_optimizer",
                side_effect=build_muon,
            ) as builder:
                final = run_grpo_stage(
                    config,
                    runtime=runtime,
                    dataset=ToyPromptDataset(),
                    dataset_identity=dataset_identity(root / "dataset"),
                    reference_identity=reference,
                    context=DistributedContext(0, 0, 1, 1),
                    device=torch.device("cpu"),
                )

            builder.assert_called_once()
            self.assertTrue((final / "pytorch_model.bin").is_file())
            self.assertTrue((root / "run" / "checkpoint-1" / "optimizer.pt").is_file())
            self.assertEqual(
                captured,
                {
                    "optimizer": "muon",
                    "learning_rate": 4e-5,
                    "weight_decay": 0.03,
                    "adam_beta1": 0.8,
                    "adam_beta2": 0.9,
                    "adam_epsilon": 2e-8,
                    "muon_learning_rate": 3e-5,
                    "muon_weight_decay": 0.02,
                    "muon_momentum": 0.9,
                    "muon_nesterov": False,
                    "muon_ns_steps": 4,
                    "muon_epsilon": 2e-7,
                    "muon_adjust_lr_fn": "original",
                },
            )

    def test_full_cpu_stage_saves_atomic_resume_state_and_loadable_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            parent, parent_manifest = write_sft_parent(root)
            reference = grpo_reference_identity(parent, parent_manifest)
            config = stage_config(root, parent)
            final = run_grpo_stage(
                config,
                runtime=toy_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )

            self.assertEqual(final, root / "run" / "final")
            exported = load_model_manifest(final / MODEL_MANIFEST_FILENAME)
            self.assertEqual(exported.stage, "grpo")
            self.assertEqual(exported.parent["stage"], "sft")
            checkpoint = root / "run" / "checkpoint-1"
            for name in (
                "pytorch_model.bin",
                "reference_model.bin",
                "optimizer.pt",
                "scheduler.pt",
                "rng.pt",
                "state.json",
                "dataset_manifest.json",
                "reference_manifest.json",
                "grpo_checkpoint_manifest.json",
            ):
                self.assertTrue((checkpoint / name).is_file(), name)
            run_manifest = json.loads(
                (root / "run" / "grpo_run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["reference"]["checkpoint_filename"], parent.name)
            self.assertIsNone(run_manifest["reference"]["base_checkpoint_filename"])
            self.assertEqual(exported.parent["selected_weight_filename"], parent.name)
            self.assertEqual(
                exported.parent["selected_weight_sha256"],
                run_manifest["reference"]["checkpoint_sha256"],
            )

            # Simulate interruption after a sealed checkpoint but before final publication.
            shutil.rmtree(final)
            resumed = run_grpo_stage(
                stage_config(root, parent, resume=True),
                runtime=toy_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )
            self.assertTrue((resumed / "pytorch_model.bin").is_file())

    def test_interrupted_resume_matches_uninterrupted_stochastic_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            parent, parent_manifest = write_sft_parent(root)
            reference = grpo_reference_identity(parent, parent_manifest)
            baseline = run_grpo_stage(
                stage_config(
                    root,
                    parent,
                    max_steps=2,
                    save_folder="baseline",
                    supervised_replay_weight=0.2,
                ),
                runtime=stochastic_replay_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )
            baseline_state = torch.load(
                baseline / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )

            interrupted_config = stage_config(
                root,
                parent,
                max_steps=2,
                save_folder="interrupted",
                supervised_replay_weight=0.2,
            )
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_grpo_stage(
                    interrupted_config,
                    runtime=stochastic_replay_runtime(
                        parent_manifest,
                        fail_on_reward_call=2,
                    ),
                    dataset=ToyPromptDataset(),
                    dataset_identity=dataset_identity(root / "dataset"),
                    reference_identity=reference,
                    context=DistributedContext(0, 0, 1, 1),
                    device=torch.device("cpu"),
                )
            resumed = run_grpo_stage(
                stage_config(
                    root,
                    parent,
                    resume=True,
                    max_steps=2,
                    save_folder="interrupted",
                    supervised_replay_weight=0.2,
                ),
                runtime=stochastic_replay_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )
            resumed_state = torch.load(
                resumed / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(set(baseline_state), set(resumed_state))
            for name in baseline_state:
                torch.testing.assert_close(
                    baseline_state[name], resumed_state[name], rtol=0, atol=0
                )

    def test_resume_rejects_tampered_optimizer_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            parent, parent_manifest = write_sft_parent(root)
            reference = grpo_reference_identity(parent, parent_manifest)
            run_grpo_stage(
                stage_config(root, parent),
                runtime=toy_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )
            shutil.rmtree(root / "run" / "final")
            with (root / "run" / "checkpoint-1" / "optimizer.pt").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(GRPOStageError, "SHA-256"):
                run_grpo_stage(
                    stage_config(
                        root,
                        parent,
                        resume=(root / "run" / "checkpoint-1").resolve(),
                    ),
                    runtime=toy_runtime(parent_manifest),
                    dataset=ToyPromptDataset(),
                    dataset_identity=dataset_identity(root / "dataset"),
                    reference_identity=reference,
                    context=DistributedContext(0, 0, 1, 1),
                    device=torch.device("cpu"),
                )

    def test_auto_resume_skips_newest_invalid_seal_but_explicit_resume_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dataset").mkdir()
            parent, parent_manifest = write_sft_parent(root)
            reference = grpo_reference_identity(parent, parent_manifest)
            run_grpo_stage(
                stage_config(root, parent, max_steps=2),
                runtime=toy_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )
            newest = root / "run" / "checkpoint-2"
            with (newest / "optimizer.pt").open("ab") as handle:
                handle.write(b"tampered")
            run_manifest = json.loads(
                (root / "run" / "grpo_run_manifest.json").read_text(encoding="utf-8")
            )

            selected = stage_module._resolve_resume_checkpoint(
                stage_config(root, parent, resume=True, max_steps=2),
                run_manifest=run_manifest,
            )
            self.assertEqual(selected, (root / "run" / "checkpoint-1").resolve())

            with self.assertRaisesRegex(GRPOStageError, "SHA-256"):
                stage_module._resolve_resume_checkpoint(
                    stage_config(root, parent, resume=newest.resolve(), max_steps=2),
                    run_manifest=run_manifest,
                )

            # The executable recovery path preserves the rejected bytes out of the checkpoint
            # namespace, replays from checkpoint-1, and publishes a new valid checkpoint-2.
            shutil.rmtree(root / "run" / "final")
            final = run_grpo_stage(
                stage_config(root, parent, resume=True, max_steps=2),
                runtime=toy_runtime(parent_manifest),
                dataset=ToyPromptDataset(),
                dataset_identity=dataset_identity(root / "dataset"),
                reference_identity=reference,
                context=DistributedContext(0, 0, 1, 1),
                device=torch.device("cpu"),
            )
            self.assertTrue((final / "pytorch_model.bin").is_file())
            self.assertEqual(len(list((root / "run").glob(".rejected-checkpoint-2.*"))), 1)
            stage_module._validate_checkpoint_manifest(newest, run_manifest=run_manifest)

    def test_grpo_parent_and_export_reject_lineage_downgrades(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, sft_manifest = write_sft_parent(root)
            self.assertEqual(validate_grpo_parent_manifest(parent).sha256, sft_manifest.sha256)
            with self.assertRaisesRegex(ModelManifestError, "SFT model manifest"):
                validate_grpo_parent_manifest(root / "pretrain" / "pytorch_model.bin")
            output = root / "bad-grpo"
            output.mkdir()
            (output / "pytorch_model.bin").write_bytes(b"grpo")
            pretrain_manifest = load_model_manifest(root / "pretrain" / MODEL_MANIFEST_FILENAME)
            with self.assertRaisesRegex(ModelManifestError, "SFT reference"):
                write_model_manifest(
                    output,
                    model_config(),
                    stage="grpo",
                    checkpoint_files=("pytorch_model.bin",),
                    parent_manifest=pretrain_manifest,
                )

    def test_sparse_ema_reference_binds_selected_and_full_base_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, _ = write_sft_parent(root)
            ema = parent.with_name("ema_model.bin")
            torch.save({"scale": torch.tensor(0.2)}, ema)
            # Re-seal the SFT manifest with both the full base and sparse EMA artifact.
            sft_manifest = write_model_manifest(
                parent.parent,
                model_config(),
                stage="sft",
                checkpoint_files=(parent.name, ema.name),
                parent_manifest=load_model_manifest(root / "pretrain" / MODEL_MANIFEST_FILENAME),
            )
            identity = grpo_reference_identity(ema, sft_manifest)
            self.assertEqual(identity["checkpoint_filename"], ema.name)
            self.assertEqual(identity["base_checkpoint_filename"], parent.name)

            output = root / "grpo-export"
            output.mkdir()
            (output / "pytorch_model.bin").write_bytes(b"grpo")
            manifest = write_model_manifest(
                output,
                model_config(),
                stage="grpo",
                checkpoint_files=("pytorch_model.bin",),
                parent_manifest=sft_manifest,
                parent_checkpoint_path=ema,
                parent_base_checkpoint_path=parent,
            )
            self.assertEqual(manifest.parent["selected_weight_filename"], ema.name)
            self.assertEqual(manifest.parent["base_weight_filename"], parent.name)
            with self.assertRaisesRegex(ModelManifestError, "full SFT base"):
                write_model_manifest(
                    output,
                    model_config(),
                    stage="grpo",
                    checkpoint_files=("pytorch_model.bin",),
                    parent_manifest=sft_manifest,
                    parent_checkpoint_path=ema,
                )

    def test_model_manifest_rejects_nonpositive_norm_and_nonboolean_capabilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "pytorch_model.bin"
            weights.write_bytes(b"weights")
            invalid = model_config()
            for norm_eps, message in ((0.0, "norm_eps must be positive"), (True, "finite number")):
                invalid["norm_eps"] = norm_eps
                with (
                    self.subTest(norm_eps=norm_eps),
                    self.assertRaisesRegex(ModelManifestError, message),
                ):
                    write_model_manifest(
                        root,
                        invalid,
                        stage="pretrain",
                        checkpoint_files=(weights.name,),
                    )
            invalid = model_config()
            invalid["use_language_conditioning"] = 1
            with self.assertRaisesRegex(ModelManifestError, "must be a boolean"):
                write_model_manifest(
                    root,
                    invalid,
                    stage="pretrain",
                    checkpoint_files=(weights.name,),
                )


if __name__ == "__main__":
    unittest.main()
