"""Tests for shared training helpers."""

import unittest

import torch

from nar_vae.dataset.representation import (
    REPRESENTATION_CONTRACT_COLUMN,
    REPRESENTATION_CONTRACT_VERSION,
    TEXT_FRONTEND_NAME,
    TEXT_FRONTEND_VERSION,
)
from nar_vae.languages import LanguagePair
from nar_vae.training_utils import (
    freeze_layers,
    resolve_duration_training_options,
    resolve_language_training_options,
    resolve_reference_language_training_options,
    resolve_speaker_training_options,
    validate_tts_dataset,
)


class TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Module()
        self.dit.text_encoder = torch.nn.Linear(2, 2)
        self.dit.speaker_encoder = torch.nn.Linear(2, 2)
        self.dit.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
        self.dit.out_proj = torch.nn.Linear(2, 2)


class FreezeLayersTest(unittest.TestCase):
    @staticmethod
    def representation_contract(*, codec_source: str = "codec/v1") -> dict[str, object]:
        return {
            "contract_version": REPRESENTATION_CONTRACT_VERSION,
            "text_frontend_name": TEXT_FRONTEND_NAME,
            "text_frontend_version": TEXT_FRONTEND_VERSION,
            "codec_source": codec_source,
            "codec_backend": "bundled",
            "codec_revision": None,
            "codec_filename": None,
            "codec_sha256": "f" * 64,
            "sample_rate": 44100,
            "hop_length": 512,
            "latent_width": 2,
        }

    def test_freezes_requested_encoders_and_leading_dit_blocks(self):
        model = TinyTrainingModel()
        freeze_layers(
            model,
            {
                "freeze_text_encoder": True,
                "freeze_speaker_encoder": False,
                "freeze_first_n_layers": 2,
            },
        )

        states = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        self.assertFalse(states["dit.text_encoder.weight"])
        self.assertTrue(states["dit.speaker_encoder.weight"])
        self.assertFalse(states["dit.blocks.0.weight"])
        self.assertFalse(states["dit.blocks.1.weight"])
        self.assertTrue(states["dit.blocks.2.weight"])
        self.assertTrue(states["dit.out_proj.weight"])

    def test_negative_block_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            freeze_layers(TinyTrainingModel(), {"freeze_first_n_layers": -1})

    def test_speaker_encoder_is_trainable_by_default(self):
        model = TinyTrainingModel()
        freeze_layers(model, {})
        self.assertTrue(model.dit.speaker_encoder.weight.requires_grad)

    def test_layer_options_do_not_reenable_architecture_frozen_parameters(self):
        model = TinyTrainingModel()
        model.dit.out_proj.weight.requires_grad_(False)

        summary = freeze_layers(model, {})

        self.assertFalse(model.dit.out_proj.weight.requires_grad)
        self.assertGreaterEqual(summary["frozen_params"], model.dit.out_proj.weight.numel())

    def test_speaker_training_options_reject_frozen_random_encoder(self):
        with self.assertRaisesRegex(ValueError, "from-scratch"):
            resolve_speaker_training_options(
                {
                    "use_speaker_conditioning": True,
                    "freeze_speaker_encoder": True,
                }
            )

    def test_speaker_initialization_requires_text_only_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "pretrained_checkpoint"):
            resolve_speaker_training_options(
                {
                    "use_speaker_conditioning": True,
                    "initialize_speaker_conditioning": True,
                }
            )

        self.assertEqual(
            resolve_speaker_training_options(
                {
                    "use_speaker_conditioning": True,
                    "initialize_speaker_conditioning": True,
                    "freeze_speaker_encoder": False,
                    "speaker_patch_size": 8,
                },
                pretrained_checkpoint="text-only.bin",
            ),
            (True, True, 8),
        )

    def test_duration_head_must_be_trained_and_explicitly_initialized(self):
        with self.assertRaisesRegex(ValueError, "positive duration_loss_weight"):
            resolve_duration_training_options(
                {"use_duration_predictor": True},
                use_speaker_conditioning=False,
            )
        with self.assertRaisesRegex(ValueError, "pretrained_checkpoint"):
            resolve_duration_training_options(
                {
                    "use_duration_predictor": True,
                    "initialize_duration_predictor": True,
                    "duration_loss_weight": 0.1,
                },
                use_speaker_conditioning=False,
            )

        options = resolve_duration_training_options(
            {
                "use_duration_predictor": True,
                "initialize_duration_predictor": True,
                "duration_predictor_hidden_size": 64,
                "duration_predictor_num_layers": 3,
                "duration_loss_weight": 0.2,
            },
            use_speaker_conditioning=False,
            pretrained_checkpoint="legacy.bin",
        )
        self.assertTrue(options.enabled)
        self.assertTrue(options.initialize)
        self.assertEqual((options.hidden_size, options.num_layers), (64, 3))

    def test_speaker_duration_requires_speaker_conditioning(self):
        with self.assertRaisesRegex(ValueError, "both duration and speaker"):
            resolve_duration_training_options(
                {
                    "use_duration_predictor": True,
                    "duration_predictor_use_speaker": True,
                    "duration_loss_weight": 0.1,
                },
                use_speaker_conditioning=False,
            )

    def test_mas_duration_options_are_explicit_and_fail_closed(self):
        config = {
            "dacvae_latent_dim": 8,
            "use_duration_predictor": True,
            "duration_loss_weight": 0.1,
            "use_mas_duration": True,
            "duration_alignment_hidden_size": 4,
            "mas_duration_loss_weight": 0.2,
            "mas_alignment_loss_weight": 0.03,
        }
        options = resolve_duration_training_options(
            config,
            use_speaker_conditioning=False,
        )
        self.assertTrue(options.uses_mas)
        self.assertEqual(options.alignment_hidden_size, 4)
        self.assertEqual(options.mas_duration_loss_weight, 0.2)
        self.assertEqual(options.mas_alignment_loss_weight, 0.03)

        # SFT may retain a matching parent capability, but cannot initialize MAS
        # together with the legacy duration-head migration path.
        sft_options = resolve_duration_training_options(
            config,
            use_speaker_conditioning=False,
            pretrained_checkpoint="mas-parent.bin",
        )
        self.assertTrue(sft_options.uses_mas)
        with self.assertRaisesRegex(ValueError, "cannot be added"):
            resolve_duration_training_options(
                {**config, "initialize_duration_predictor": True},
                use_speaker_conditioning=False,
                pretrained_checkpoint="legacy-parent.bin",
            )

        invalid_configs = (
            ({**config, "use_mas_duration": "true"}, "must be a boolean"),
            ({**config, "use_mas_duration": False}, "must be zero"),
            ({**config, "mas_duration_loss_weight": 0.0}, "positive"),
            ({**config, "duration_alignment_hidden_size": 9}, "cannot exceed"),
        )
        for invalid, message in invalid_configs:
            with self.subTest(invalid=invalid, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    resolve_duration_training_options(
                        invalid,
                        use_speaker_conditioning=False,
                    )

    def test_speaker_training_requires_valid_references_in_every_row(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
            }
        ]

        with self.assertRaisesRegex(ValueError, "no speaker_latents column"):
            validate_tts_dataset(rows, latent_size=2, use_speaker_conditioning=True)

        rows[0]["speaker_latents"] = torch.zeros(2, 4)
        validate_tts_dataset(rows, latent_size=2, use_speaker_conditioning=True)

    def test_invalid_speaker_reference_is_rejected(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                "speaker_latents": torch.empty(2, 0),
            }
        ]

        with self.assertRaisesRegex(ValueError, "invalid speaker_latents"):
            validate_tts_dataset(rows, latent_size=2, use_speaker_conditioning=True)

    def test_persisted_latent_frame_count_must_match_the_array(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "latent_num_frames": 3,
                "conditioning_ids": [1],
            }
        ]
        validate_tts_dataset(rows, latent_size=2, use_speaker_conditioning=False)

        rows[0]["latent_num_frames"] = 4
        with self.assertRaisesRegex(ValueError, "latents contain 3 frames"):
            validate_tts_dataset(rows, latent_size=2, use_speaker_conditioning=False)

    def test_legacy_dataset_without_frame_metadata_remains_valid(self):
        validate_tts_dataset(
            [{"latents": torch.zeros(2, 3), "conditioning_ids": [1]}],
            latent_size=2,
            use_speaker_conditioning=False,
        )

    def test_server_training_requires_one_consistent_representation_contract(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                REPRESENTATION_CONTRACT_COLUMN: self.representation_contract(),
            },
            {
                "latents": torch.zeros(2, 4),
                "conditioning_ids": [2],
                REPRESENTATION_CONTRACT_COLUMN: self.representation_contract(),
            },
        ]
        validate_tts_dataset(
            rows,
            latent_size=2,
            use_speaker_conditioning=False,
            allow_legacy_representation=False,
            expected_codec_source="codec/v1",
            expected_codec_backend="bundled",
            expected_sample_rate=44100,
            expected_hop_length=512,
        )

        legacy = [{"latents": torch.zeros(2, 3), "conditioning_ids": [1]}]
        with self.assertRaisesRegex(ValueError, "representation_contract"):
            validate_tts_dataset(
                legacy,
                latent_size=2,
                use_speaker_conditioning=False,
                allow_legacy_representation=False,
            )

        rows[1][REPRESENTATION_CONTRACT_COLUMN] = self.representation_contract(
            codec_source="codec/v2"
        )
        with self.assertRaisesRegex(ValueError, "different codec/frontend"):
            validate_tts_dataset(
                rows,
                latent_size=2,
                use_speaker_conditioning=False,
                allow_legacy_representation=False,
            )

    def test_training_configuration_must_match_the_prepared_codec_contract(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                REPRESENTATION_CONTRACT_COLUMN: self.representation_contract(),
            }
        ]

        with self.assertRaisesRegex(ValueError, "codec_source.*does not match"):
            validate_tts_dataset(
                rows,
                latent_size=2,
                use_speaker_conditioning=False,
                allow_legacy_representation=False,
                expected_codec_source="codec/other",
            )

    def test_mas_dataset_requires_one_frame_per_conditioning_token(self):
        valid = [{"latents": torch.zeros(2, 3), "conditioning_ids": [1, 2, 3]}]
        validate_tts_dataset(
            valid,
            latent_size=2,
            use_speaker_conditioning=False,
            use_mas_duration=True,
        )

        invalid = [{"latents": torch.zeros(2, 2), "conditioning_ids": [1, 2, 3]}]
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            validate_tts_dataset(
                invalid,
                latent_size=2,
                use_speaker_conditioning=False,
                use_mas_duration=True,
            )
        # The invariant is specific to the optional MAS objective.
        validate_tts_dataset(
            invalid,
            latent_size=2,
            use_speaker_conditioning=False,
            use_mas_duration=False,
        )

    def test_language_training_options_normalize_one_or_many_codes(self):
        self.assertEqual(
            resolve_language_training_options(
                {
                    "use_language_conditioning": True,
                    "supported_languages": "Turkish",
                }
            ),
            (True, False, ("tr",)),
        )
        self.assertEqual(
            resolve_language_training_options(
                {
                    "use_language_conditioning": True,
                    "supported_languages": ["en-US", "Spanish", "en"],
                }
            ),
            (True, False, ("en", "es")),
        )

    def test_language_initialization_requires_a_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "pretrained_checkpoint"):
            resolve_language_training_options(
                {
                    "use_language_conditioning": True,
                    "initialize_language_conditioning": True,
                }
            )

        with self.assertRaisesRegex(ValueError, "beyond English"):
            resolve_language_training_options({"supported_languages": ["en", "es"]})

    def test_language_training_validates_target_and_reference_independently(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                "language": "es",
                "speaker_latents": torch.zeros(2, 4),
                "speaker_language": "en",
            }
        ]

        validate_tts_dataset(
            rows,
            latent_size=2,
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=("en", "es"),
            supported_reference_languages=("en",),
        )
        with self.assertRaisesRegex(ValueError, "unsupported language"):
            validate_tts_dataset(
                rows,
                latent_size=2,
                use_speaker_conditioning=True,
                use_language_conditioning=True,
                supported_languages=("en",),
                supported_reference_languages=("en",),
            )

    def test_reference_language_options_require_explicit_conditioning(self):
        with self.assertRaisesRegex(ValueError, "both speaker and language"):
            resolve_reference_language_training_options(
                {"supported_reference_languages": ["en"]},
                use_speaker_conditioning=True,
                use_language_conditioning=False,
            )

        self.assertEqual(
            resolve_reference_language_training_options(
                {"supported_reference_languages": ["English", "Japanese"]},
                use_speaker_conditioning=True,
                use_language_conditioning=True,
            ),
            (
                ("en", "ja"),
                (LanguagePair("en", "en"), LanguagePair("en", "ja")),
                False,
            ),
        )

        with self.assertRaisesRegex(ValueError, "requires exact supported_language_pairs"):
            resolve_reference_language_training_options(
                {},
                use_speaker_conditioning=True,
                use_language_conditioning=True,
                supported_languages=("en", "es"),
            )

    def test_exact_language_pairs_are_authoritative_over_legacy_shorthand(self):
        references, pairs, initialize = resolve_reference_language_training_options(
            {
                "supported_reference_languages": ["Japanese"],
                "supported_language_pairs": [["Spanish", "English"], ["en", "ja"]],
            },
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=("en", "es"),
        )

        self.assertEqual(references, ("en", "ja"))
        self.assertEqual(
            pairs,
            (LanguagePair("es", "en"), LanguagePair("en", "ja")),
        )
        self.assertFalse(initialize)

    def test_dataset_rejects_untrained_language_pair_and_pair_overclaim(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                "language": "es",
                "speaker_latents": torch.zeros(2, 4),
                "speaker_language": "en",
            }
        ]

        validate_tts_dataset(
            rows,
            latent_size=2,
            use_speaker_conditioning=True,
            use_language_conditioning=True,
            supported_languages=("en", "es"),
            supported_language_pairs=(("es", "en"),),
        )
        with self.assertRaisesRegex(ValueError, "unsupported target/reference language pair"):
            validate_tts_dataset(
                rows,
                latent_size=2,
                use_speaker_conditioning=True,
                use_language_conditioning=True,
                supported_languages=("en", "es"),
                supported_language_pairs=(("es", "ja"),),
            )
        with self.assertRaisesRegex(ValueError, "language-pair coverage"):
            validate_tts_dataset(
                rows
                + [
                    {
                        "latents": torch.zeros(2, 3),
                        "conditioning_ids": [1],
                        "language": "en",
                        "speaker_latents": torch.zeros(2, 4),
                        "speaker_language": "ja",
                    }
                ],
                latent_size=2,
                use_speaker_conditioning=True,
                use_language_conditioning=True,
                supported_languages=("en", "es"),
                supported_language_pairs=(("es", "en"), ("en", "ja"), ("es", "ja")),
                require_language_coverage=True,
            )

    def test_strict_language_coverage_prevents_checkpoint_overclaiming(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                "language": "es",
            }
        ]

        with self.assertRaisesRegex(ValueError, "target-language coverage"):
            validate_tts_dataset(
                rows,
                latent_size=2,
                use_speaker_conditioning=False,
                use_language_conditioning=True,
                supported_languages=("en", "es"),
                require_language_coverage=True,
            )

    def test_strict_reference_coverage_prevents_checkpoint_overclaiming(self):
        rows = [
            {
                "latents": torch.zeros(2, 3),
                "conditioning_ids": [1],
                "language": "es",
                "speaker_latents": torch.zeros(2, 4),
                "speaker_language": "en",
            }
        ]

        with self.assertRaisesRegex(ValueError, "reference-language coverage"):
            validate_tts_dataset(
                rows,
                latent_size=2,
                use_speaker_conditioning=True,
                use_language_conditioning=True,
                supported_languages=("es",),
                supported_reference_languages=("en", "ja"),
                require_language_coverage=True,
            )


if __name__ == "__main__":
    unittest.main()
