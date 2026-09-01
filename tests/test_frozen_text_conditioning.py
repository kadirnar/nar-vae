"""Frozen text-feature boundary and token-axis contract tests."""

import unittest

import torch

from nar_vae.dataset.data_collator import FlowMatchingDataCollator
from nar_vae.losses.flow_matching_loss import FlowMatchingLoss
from nar_vae.models.flow_matching import create_flow_matching_echodit
from nar_vae.models.text_conditioning import (
    FROZEN_FEATURE_ADAPTER_VERSION,
    FROZEN_FEATURE_TEXT_CONDITIONING,
    SCRATCH_TOKEN_TEXT_CONDITIONING,
    TEXT_CONDITIONING_ADAPTER_VERSION_KEY,
    TEXT_CONDITIONING_FEATURE_SIZE_KEY,
    TEXT_CONDITIONING_MODE_KEY,
    TEXT_CONDITIONING_VERSION,
    TEXT_CONDITIONING_VERSION_KEY,
    FrozenTextFeatureAdapter,
    resolve_text_conditioning_metadata,
)


def _sample(token_ids: list[int], feature_size: int | None = None) -> dict:
    sample = {
        "latents": torch.zeros(4, 5),
        "conditioning_ids": token_ids,
        "token_language_ids": [1] * len(token_ids),
        "alignment_mask": [True] * len(token_ids),
        "language": "en",
    }
    if feature_size is not None:
        sample["conditioning_features"] = torch.arange(
            len(token_ids) * feature_size,
            dtype=torch.float32,
        ).reshape(len(token_ids), feature_size)
        sample["conditioning_feature_dtype"] = "float32"
    return sample


def _tiny_model_kwargs() -> dict:
    return {
        "latent_size": 4,
        "model_size": 8,
        "num_layers": 1,
        "num_heads": 2,
        "intermediate_size": 16,
        "text_vocab_size": 32,
        "text_model_size": 8,
        "text_num_heads": 2,
        "text_intermediate_size": 16,
        "speaker_patch_size": 2,
        "speaker_model_size": 8,
        "speaker_num_layers": 1,
        "speaker_num_heads": 2,
        "speaker_intermediate_size": 16,
        "timestep_embed_size": 8,
        "adaln_rank": 4,
        "cfg_dropout": 0.0,
    }


def _tiny_frozen_model(**overrides):
    options = _tiny_model_kwargs()
    options.update(
        text_num_layers=0,
        text_conditioning_mode=FROZEN_FEATURE_TEXT_CONDITIONING,
        conditioning_feature_size=6,
    )
    options.update(overrides)
    return create_flow_matching_echodit(**options)


def _tiny_scratch_model():
    options = _tiny_model_kwargs()
    options.update(
        text_num_layers=1,
        text_conditioning_mode=SCRATCH_TOKEN_TEXT_CONDITIONING,
    )
    return create_flow_matching_echodit(**options)


class FrozenTextFeatureAdapterTest(unittest.TestCase):
    def test_metadata_is_versioned_and_rejects_ambiguous_topologies(self):
        scratch = resolve_text_conditioning_metadata(
            SCRATCH_TOKEN_TEXT_CONDITIONING,
            None,
        )
        frozen = resolve_text_conditioning_metadata(
            FROZEN_FEATURE_TEXT_CONDITIONING,
            12,
        )

        self.assertNotEqual(scratch.mode_code, frozen.mode_code)
        self.assertEqual(scratch.feature_size, 0)
        self.assertEqual(frozen.feature_size, 12)
        self.assertEqual(frozen.adapter_version, FROZEN_FEATURE_ADAPTER_VERSION)
        with self.assertRaisesRegex(ValueError, "positive conditioning_feature_size"):
            resolve_text_conditioning_metadata(FROZEN_FEATURE_TEXT_CONDITIONING, None)
        with self.assertRaisesRegex(ValueError, "only valid"):
            resolve_text_conditioning_metadata(SCRATCH_TOKEN_TEXT_CONDITIONING, 12)

    def test_adapter_detaches_frozen_states_but_trains_projection_and_language_rows(self):
        adapter = FrozenTextFeatureAdapter(feature_size=3, model_size=4, num_languages=2)
        with torch.no_grad():
            adapter.feature_projection.weight.zero_()
            adapter.feature_projection.bias.zero_()
            adapter.language_embedding.weight[1].fill_(1.0)
            adapter.language_embedding.weight[2].fill_(2.0)

        features = torch.randn(1, 2, 3, requires_grad=True)
        output = adapter(features, torch.tensor([[1, 2]]))

        torch.testing.assert_close(output[0, 0], torch.ones(4))
        torch.testing.assert_close(output[0, 1], torch.full((4,), 2.0))
        output.sum().backward()
        self.assertIsNone(features.grad)
        self.assertIsNotNone(adapter.feature_projection.weight.grad)
        self.assertIsNotNone(adapter.language_embedding.weight.grad)
        self.assertFalse(any("backbone" in name for name, _ in adapter.named_modules()))


class FrozenFeatureCollatorTest(unittest.TestCase):
    def test_pads_aligned_features_on_the_existing_conditioning_axis(self):
        batch = FlowMatchingDataCollator()(
            [
                _sample([7, 8], feature_size=3),
                _sample([9, 10, 11], feature_size=3),
            ]
        )

        self.assertEqual(batch["conditioning_features"].shape, (2, 3, 3))
        self.assertEqual(batch["conditioning_ids"].shape, (2, 3))
        self.assertTrue((batch["conditioning_features"][0, 2] == 0).all())
        torch.testing.assert_close(
            batch["conditioning_features"][1],
            _sample([9, 10, 11], feature_size=3)["conditioning_features"],
        )

    def test_rejects_partial_or_misaligned_feature_batches(self):
        with self.assertRaisesRegex(ValueError, "every sample"):
            FlowMatchingDataCollator()([_sample([7, 8], feature_size=3), _sample([9, 10, 11])])

        malformed = _sample([7, 8], feature_size=3)
        malformed["conditioning_features"] = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "conditioning_tokens"):
            FlowMatchingDataCollator()([malformed])


class FrozenFeatureTrainingBoundaryTest(unittest.TestCase):
    def test_legacy_scratch_checkpoint_schema_has_no_new_text_metadata(self):
        state = _tiny_scratch_model().state_dict()
        for key in (
            TEXT_CONDITIONING_VERSION_KEY,
            TEXT_CONDITIONING_MODE_KEY,
            TEXT_CONDITIONING_FEATURE_SIZE_KEY,
            TEXT_CONDITIONING_ADAPTER_VERSION_KEY,
        ):
            self.assertNotIn(key, state)

    def test_model_training_forward_consumes_features_without_a_text_backbone(self):
        model = _tiny_frozen_model()
        state = model.state_dict()
        self.assertEqual(int(state[TEXT_CONDITIONING_VERSION_KEY]), TEXT_CONDITIONING_VERSION)
        self.assertEqual(int(state[TEXT_CONDITIONING_MODE_KEY]), 1)
        self.assertEqual(int(state[TEXT_CONDITIONING_FEATURE_SIZE_KEY]), 6)
        self.assertEqual(
            int(state[TEXT_CONDITIONING_ADAPTER_VERSION_KEY]),
            FROZEN_FEATURE_ADAPTER_VERSION,
        )
        self.assertFalse(
            any(
                name.startswith("dit.text_encoder.text_embedding")
                for name, _ in model.named_parameters()
            )
        )
        self.assertFalse(
            any(name.startswith("dit.text_encoder.blocks") for name, _ in model.named_parameters())
        )

        conditioning_ids = torch.tensor([[7, 8, 9], [10, 11, 0]])
        conditioning_mask = torch.tensor([[True, True, True], [True, True, False]])
        conditioning_features = torch.randn(2, 3, 6)
        prediction = model(
            latents=torch.randn(2, 4, 5),
            conditioning_ids=conditioning_ids,
            conditioning_features=conditioning_features,
            timesteps=torch.rand(2),
            attention_mask=conditioning_mask,
            use_cfg_dropout=False,
        )
        self.assertEqual(prediction.shape, (2, 4, 5))

        differentiable_features = conditioning_features.clone().requires_grad_(True)
        encoded = model.dit.encode_text(
            conditioning_ids,
            conditioning_mask,
            conditioning_features=differentiable_features,
        )
        encoded.sum().backward()
        self.assertIsNone(differentiable_features.grad)
        self.assertIsNotNone(model.dit.text_encoder.feature_projection.weight.grad)

        with self.assertRaisesRegex(ValueError, "are required"):
            model(
                latents=torch.randn(2, 4, 5),
                conditioning_ids=conditioning_ids,
                timesteps=torch.rand(2),
                attention_mask=conditioning_mask,
                use_cfg_dropout=False,
            )

    def test_cached_features_reach_inference_cfg_and_duration_helpers(self):
        model = _tiny_frozen_model()
        conditioning_ids = torch.tensor([[7, 8, 9]])
        conditioning_mask = torch.ones_like(conditioning_ids, dtype=torch.bool)
        conditioning_features = torch.randn(1, 3, 6)

        prepared = model.prepare_inference_conditioning(
            conditioning_ids,
            conditioning_mask,
            conditioning_features=conditioning_features,
        )
        self.assertEqual(len(prepared.kv_cache_text), 1)
        fused = model.prepare_fused_cfg_conditioning(
            conditioning_ids,
            conditioning_mask,
            cfg_mode="joint",
            conditioning_features=conditioning_features,
        )
        self.assertEqual(fused.branch_count, 2)
        guided = model.forward_with_cfg(
            latents=torch.randn(1, 4, 5),
            conditioning_ids=conditioning_ids,
            conditioning_features=conditioning_features,
            timesteps=torch.rand(1),
            cfg_scale=1.5,
            attention_mask=conditioning_mask,
            fuse_cfg_branches=True,
        )
        self.assertEqual(guided.shape, (1, 4, 5))

        duration_model = _tiny_frozen_model(
            use_duration_predictor=True,
            use_mas_duration=True,
            duration_predictor_hidden_size=8,
            duration_predictor_num_layers=1,
            duration_alignment_hidden_size=4,
        )
        log_duration = duration_model.predict_log_duration(
            conditioning_ids,
            conditioning_mask,
            conditioning_features=conditioning_features,
        )
        expected = duration_model.predict_expected_token_durations(
            conditioning_ids,
            conditioning_mask,
            conditioning_features=conditioning_features,
        )
        allocated = duration_model.predict_token_duration_frames(
            conditioning_ids,
            conditioning_mask,
            conditioning_features=conditioning_features,
            total_frames=6,
        )
        self.assertEqual(log_duration.shape, (1,))
        self.assertEqual(expected.shape, conditioning_ids.shape)
        self.assertEqual(allocated.shape, conditioning_ids.shape)
        self.assertEqual(int(allocated.sum()), 6)

    def test_loss_validates_and_threads_cached_features(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conditioning_features = None

            def forward(self, **kwargs):
                self.conditioning_features = kwargs.get("conditioning_features")
                return torch.zeros_like(kwargs["latents"])

        model = RecordingModel()
        features = torch.randn(1, 2, 6)
        loss = FlowMatchingLoss(timestep_distribution="uniform")(
            model=model,
            latents=torch.randn(1, 4, 3),
            conditioning_ids=torch.tensor([[7, 8]]),
            conditioning_features=features,
        )
        self.assertEqual(loss.ndim, 0)
        self.assertIs(model.conditioning_features, features)

        with self.assertRaisesRegex(ValueError, "conditioning_token"):
            FlowMatchingLoss(timestep_distribution="uniform")(
                model=model,
                latents=torch.randn(1, 4, 3),
                conditioning_ids=torch.tensor([[7, 8]]),
                conditioning_features=torch.randn(1, 3, 6),
            )


if __name__ == "__main__":
    unittest.main()
