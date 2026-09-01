"""Tests for packaged EchoDiT model-size configurations."""

import unittest
from pathlib import Path

import torch
import yaml

from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.model_presets import (
    ARCHITECTURE_FIELDS,
    ModelPreset,
    get_model_preset,
    list_model_presets,
    resolve_model_architecture,
)
from nar_vae.models.flow_matching import create_flow_matching_echodit

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PARAMETER_COUNTS = {
    "nano": (3_495_425, 4_034_753),
    "tiny": (14_970_113, 17_166_977),
    "small": (45_281_025, 51_193_089),
    "medium": (118_516_481, 135_152_129),
    "large": (290_071_297, 324_046_209),
    "xlarge": (570_432_257, 613_846_785),
}

EXPECTED_CANONICAL_FROZEN_PARAMETER_COUNTS = {
    "nano": 3_604_113,
    "tiny": 15_316_385,
    "small": 44_854_209,
    "medium": 109_660_097,
    "large": 284_411_521,
    "xlarge": 556_217_217,
}


class ModelPresetTest(unittest.TestCase):
    def test_expected_presets_are_ordered_from_nano_to_xlarge(self):
        self.assertEqual(
            list_model_presets(),
            ("nano", "tiny", "small", "medium", "large", "xlarge"),
        )

        widths = [get_model_preset(name).model_size for name in list_model_presets()]
        layers = [get_model_preset(name).num_layers for name in list_model_presets()]
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(layers, sorted(layers))

    def test_nano_is_the_cheapest_valid_architecture(self):
        nano = get_model_preset("nano")

        self.assertEqual(
            nano.model_kwargs(),
            {
                "model_size": 128,
                "num_layers": 6,
                "num_heads": 4,
                "intermediate_size": 512,
                "text_model_size": 128,
                "text_num_layers": 2,
                "text_num_heads": 4,
                "text_intermediate_size": 512,
                "speaker_model_size": 96,
                "speaker_num_layers": 2,
                "speaker_num_heads": 3,
                "speaker_intermediate_size": 384,
                "timestep_embed_size": 64,
                "adaln_rank": 32,
            },
        )

    def test_every_preset_has_valid_attention_dimensions(self):
        for name in list_model_presets():
            preset = get_model_preset(name)
            for prefix in ("", "text_", "speaker_"):
                width = getattr(preset, f"{prefix}model_size")
                heads = getattr(preset, f"{prefix}num_heads")
                self.assertEqual(width % heads, 0, name)
                self.assertEqual((width // heads) % 2, 0, name)
            self.assertEqual(set(preset.model_kwargs()), set(ARCHITECTURE_FIELDS))

    def test_named_preset_rejects_silent_shape_overrides(self):
        with self.assertRaisesRegex(ValueError, "conflicts with model_preset"):
            resolve_model_architecture({"model_preset": "tiny", "model_size": 1024})

        self.assertEqual(resolve_model_architecture({"model_preset": "tiny"}).name, "tiny")

    def test_custom_architecture_requires_every_field(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            resolve_model_architecture({"model_size": 256})

        tiny = get_model_preset("tiny")
        custom = resolve_model_architecture(tiny.model_kwargs())
        self.assertEqual(custom.name, "custom")
        self.assertEqual(custom.model_kwargs(), tiny.model_kwargs())

    def test_invalid_rotary_head_width_is_rejected(self):
        values = get_model_preset("tiny").model_kwargs()
        values["model_size"] = 120
        values["num_heads"] = 8

        with self.assertRaisesRegex(ValueError, "head width must be even"):
            ModelPreset(name="invalid", description="test", **values)

    def test_unknown_preset_has_actionable_choices(self):
        with self.assertRaisesRegex(ValueError, "nano, tiny, small, medium, large, xlarge"):
            get_model_preset("micro")

    def test_packaged_training_configs_use_one_switch_for_every_preset(self):
        for filename in ("echodit_config.yaml", "finetune_config.yaml"):
            path = ROOT / "nar_vae" / "configs" / filename
            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)

            self.assertFalse(set(ARCHITECTURE_FIELDS).intersection(config), filename)
            for name in list_model_presets():
                with self.subTest(filename=filename, preset=name):
                    switched = dict(config, model_preset=name)
                    resolved = resolve_model_architecture(switched)
                    expected = get_model_preset(name)
                    self.assertEqual(resolved.text_num_layers, 0)
                    for field in ARCHITECTURE_FIELDS:
                        if field != "text_num_layers":
                            self.assertEqual(getattr(resolved, field), getattr(expected, field))

    def test_every_preset_constructs_with_documented_parameter_counts(self):
        for name, (expected_base, expected_full) in EXPECTED_PARAMETER_COUNTS.items():
            preset_kwargs = get_model_preset(name).model_kwargs()
            with self.subTest(preset=name, topology="base"), torch.device("meta"):
                model = create_flow_matching_echodit(
                    latent_size=128,
                    text_vocab_size=530,
                    target_patch_size=1,
                    use_duration_predictor=True,
                    use_mas_duration=True,
                    **preset_kwargs,
                )
                self.assertEqual(model.get_num_params()["total"], expected_base)

            with self.subTest(preset=name, topology="fully_conditioned"), torch.device("meta"):
                model = create_flow_matching_echodit(
                    latent_size=128,
                    text_vocab_size=530,
                    target_patch_size=1,
                    use_speaker_conditioning=True,
                    use_language_conditioning=True,
                    supported_languages=("en", "tr"),
                    supported_language_pairs=(("en", "en"), ("tr", "tr"), ("tr", "en")),
                    use_duration_predictor=True,
                    duration_predictor_use_speaker=True,
                    use_mas_duration=True,
                    **preset_kwargs,
                )
                self.assertEqual(model.get_num_params()["total"], expected_full)

    def test_canonical_frozen_multilingual_clone_counts_match_documentation(self):
        for name, expected in EXPECTED_CANONICAL_FROZEN_PARAMETER_COUNTS.items():
            preset_kwargs = get_model_preset(name).model_kwargs()
            # The external provider replaces the preset's scratch text stack.
            preset_kwargs["text_num_layers"] = 0
            with self.subTest(preset=name), torch.device("meta"):
                model = create_flow_matching_echodit(
                    latent_size=128,
                    text_vocab_size=1969,
                    text_conditioning_mode="frozen_features",
                    conditioning_feature_size=768,
                    target_patch_size=2,
                    use_speaker_conditioning=True,
                    speaker_num_summary_tokens=8,
                    use_language_conditioning=True,
                    supported_languages=("en", "tr"),
                    supported_language_pairs=(("en", "en"), ("tr", "tr"), ("tr", "en")),
                    use_duration_predictor=True,
                    duration_predictor_use_speaker=True,
                    use_mas_duration=True,
                    **preset_kwargs,
                )

            self.assertEqual(model.get_num_params()["total"], expected)

    def test_inference_factory_passes_the_complete_preset(self):
        class RecordingInference(FlowMatchingTTSInference):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        runtime = RecordingInference.from_preset("small", flow_model_path="small.bin")

        self.assertEqual(runtime.kwargs["flow_model_path"], "small.bin")
        for name, value in get_model_preset("small").model_kwargs().items():
            self.assertEqual(runtime.kwargs[name], value)

        with self.assertRaisesRegex(ValueError, "conflict with model preset"):
            RecordingInference.from_preset("small", model_size=1024)


if __name__ == "__main__":
    unittest.main()
