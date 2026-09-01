import unittest
from pathlib import Path

import torch

from nar_vae.configuration import validate_pretraining_config
from nar_vae.dataset.data_collator import FlowMatchingDataCollator
from nar_vae.languages import language_id
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.text_frontend import FrozenTextFrontendSpec
from nar_vae.train import _load_pretraining_yaml

ROOT = Path(__file__).resolve().parents[1]


def tiny_frozen_model(
    *,
    latent_patch_size: int = 4,
    speaker_conditioning: bool = False,
    language_conditioning: bool = True,
    supported_languages: tuple[str, ...] = ("en", "tr"),
) -> FlowMatchingEchoDiT:
    model = FlowMatchingEchoDiT(
        latent_size=2,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=32,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        timestep_embed_size=8,
        adaln_rank=4,
        use_speaker_conditioning=speaker_conditioning,
        use_language_conditioning=language_conditioning,
        supported_languages=supported_languages,
        latent_patch_size=latent_patch_size,
        text_encoder_type="frozen_features",
        frozen_text_input_size=6,
        text_adapter_bottleneck_ratio=2,
    )
    torch.nn.init.normal_(model.dit.out_proj.weight, std=0.1)
    return model


class FrozenMultilingualArchitectureTest(unittest.TestCase):
    def test_frozen_adapter_has_no_scratch_embedding_and_requires_aligned_features(self):
        model = tiny_frozen_model()
        self.assertFalse(hasattr(model.dit.text_encoder, "text_embedding"))
        ids = torch.tensor([[1, 2, 3]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        languages = torch.tensor([language_id("tr")])
        with self.assertRaisesRegex(ValueError, "requires conditioning_features"):
            model.dit.encode_text(ids, mask, languages)
        state = model.dit.encode_text(
            ids,
            mask,
            languages,
            torch.randn(1, 3, 6),
        )
        self.assertEqual(tuple(state.shape), (1, 3, 8))

    def test_internal_patching_preserves_raw_shape_and_blocks_padding_leakage(self):
        torch.manual_seed(7)
        model = tiny_frozen_model(latent_patch_size=4).eval()
        ids = torch.tensor([[1, 2, 3]])
        features = torch.randn(1, 3, 6)
        language_ids = torch.tensor([language_id("en")])
        mask = torch.tensor([[True, True, True, False, False]])
        first = torch.randn(1, 2, 5, requires_grad=True)
        second = first.detach().clone()
        second[..., 3:] = 10_000

        kwargs = dict(
            conditioning_ids=ids,
            timesteps=torch.tensor([0.5]),
            language_ids=language_ids,
            latent_mask=mask,
            conditioning_features=features,
            use_cfg_dropout=False,
        )
        output_a = model(latents=first, **kwargs)
        output_b = model(latents=second, **kwargs)
        self.assertEqual(tuple(output_a.shape), (1, 2, 5))
        torch.testing.assert_close(output_a[..., :3], output_b[..., :3])
        torch.testing.assert_close(output_a[..., 3:], torch.zeros_like(output_a[..., 3:]))

        output_a[..., :3].sum().backward()
        torch.testing.assert_close(first.grad[..., 3:], torch.zeros_like(first.grad[..., 3:]))

    def test_cfg_uses_the_adapter_null_state_instead_of_token_zero(self):
        model = tiny_frozen_model().eval()
        with torch.no_grad():
            model.dit.text_encoder.null_state.fill_(0.25)
        encoded = model.encode_inference_conditioning(
            torch.tensor([[4, 5]]),
            torch.ones(1, 2, dtype=torch.bool),
            cfg_mode="joint",
            language_ids=torch.tensor([language_id("en")]),
            conditioning_features=torch.randn(1, 2, 6),
        )
        conditional, unconditional = encoded.variants[0].text_state.chunk(2)
        self.assertFalse(torch.equal(conditional, unconditional))
        self.assertGreater(float(unconditional.abs().sum()), 0.0)
        torch.testing.assert_close(unconditional[:, 1], torch.zeros_like(unconditional[:, 1]))

    def test_collator_pads_frozen_states_on_the_token_axis(self):
        batch = FlowMatchingDataCollator(
            pad_token=0,
            conditioning_feature_dtype="float16",
        )(
            [
                {
                    "latents": torch.zeros(2, 3),
                    "conditioning_ids": [1, 2],
                    "conditioning_features": [[1.0, 2.0], [3.0, 4.0]],
                    "representation_contract": {"text_frontend": {"feature_dtype": "float16"}},
                },
                {
                    "latents": torch.zeros(2, 2),
                    "conditioning_ids": [3],
                    "conditioning_features": [[5.0, 6.0]],
                    "representation_contract": {"text_frontend": {"feature_dtype": "float16"}},
                },
            ]
        )
        self.assertEqual(tuple(batch["conditioning_features"].shape), (2, 2, 2))
        self.assertEqual(batch["conditioning_features"].dtype, torch.float16)
        torch.testing.assert_close(
            batch["conditioning_features"][1, 1],
            torch.zeros(2, dtype=torch.float16),
        )

    def test_frontend_contract_is_content_addressed_and_rejects_mutable_revisions(self):
        spec = FrozenTextFrontendSpec(
            model_id="jhu-clsp/mmBERT-small",
            revision="abc32620dd4f6ab06f5fbe905dc25f310618e09f",
            hidden_size=384,
            max_length=512,
            input_mode="raw_text",
            feature_dtype="float16",
            license="MIT",
        )
        self.assertIn(spec.fingerprint, spec.contract_name)
        with self.assertRaisesRegex(ValueError, "40-character"):
            FrozenTextFrontendSpec(
                model_id="mutable/model",
                revision="main",
                hidden_size=8,
                max_length=8,
                license="MIT",
            )

    def test_packaged_frozen_raw_text_recipes_validate(self):
        expected = {
            "multilingual_frozen_config.yaml": ("raw_text", 384, 256000, 0),
            "turkish_frozen_config.yaml": ("raw_text", 384, 256000, 0),
        }
        for filename, contract in expected.items():
            with self.subTest(filename=filename):
                config = _load_pretraining_yaml(ROOT / "nar_vae" / "configs" / filename)
                validate_pretraining_config(config)
                spec = FrozenTextFrontendSpec.from_config(config)
                self.assertEqual(
                    (
                        spec.input_mode,
                        spec.hidden_size,
                        config["text_vocab_size"],
                        config["pad_token"],
                    ),
                    contract,
                )

    def test_turkish_recipe_is_strictly_monolingual_without_a_learned_language_bias(self):
        config = _load_pretraining_yaml(ROOT / "nar_vae" / "configs" / "turkish_frozen_config.yaml")
        validate_pretraining_config(config)
        self.assertFalse(config["use_language_conditioning"])
        self.assertEqual(config["supported_languages"], ["tr"])
        self.assertIsNone(config["supported_language_pairs"])
        self.assertEqual(config["latent_patch_size"], 2)
        self.assertFalse(config["use_mas_duration"])

        rank_deficient = dict(config, latent_patch_size=4)
        with self.assertRaisesRegex(ValueError, "rank-deficient acoustic projection"):
            validate_pretraining_config(rank_deficient)

    def test_turkish_monolingual_model_accepts_only_turkish_metadata(self):
        model = tiny_frozen_model(
            language_conditioning=False,
            supported_languages=("tr",),
        )
        self.assertIsNone(
            model._prepare_language_ids(None, batch_size=1, device=torch.device("cpu"))
        )
        self.assertIsNone(
            model._prepare_language_ids(
                torch.tensor([language_id("tr")]),
                batch_size=1,
                device=torch.device("cpu"),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "monolingual checkpoint"):
            model._prepare_language_ids(
                torch.tensor([language_id("en")]),
                batch_size=1,
                device=torch.device("cpu"),
            )

    def test_cfg_null_speaker_prediction_does_not_leak_reference_length(self):
        torch.manual_seed(11)
        model = tiny_frozen_model(
            speaker_conditioning=True,
            language_conditioning=False,
            supported_languages=("tr",),
        ).eval()
        ids = torch.tensor([[4, 5]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        features = torch.randn(1, 2, 6)
        latents = torch.randn(1, 2, 4)
        timesteps = torch.tensor([0.5])

        unconditional_predictions = []
        for reference_frames in (4, 8):
            prepared = model.prepare_fused_cfg_conditioning(
                ids,
                mask,
                torch.randn(1, 2, reference_frames),
                cfg_mode="joint",
                language_ids=torch.tensor([language_id("tr")]),
                conditioning_features=features,
            )
            unconditional = prepared.variants[0].slice_batch(1, 2)
            unconditional_predictions.append(
                model.forward_prepared(latents, timesteps, unconditional)
            )

        torch.testing.assert_close(
            unconditional_predictions[0],
            unconditional_predictions[1],
        )


if __name__ == "__main__":
    unittest.main()
