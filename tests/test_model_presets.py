"""Tests for packaged EchoDiT model-size configurations."""

import unittest

from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.model_presets import (
    ARCHITECTURE_FIELDS,
    ModelPreset,
    get_model_preset,
    list_model_presets,
    resolve_model_architecture,
)


class ModelPresetTest(unittest.TestCase):
    def test_expected_presets_are_ordered_from_tiny_to_xlarge(self):
        self.assertEqual(
            list_model_presets(),
            ("tiny", "small", "medium", "large", "xlarge"),
        )

        widths = [get_model_preset(name).model_size for name in list_model_presets()]
        layers = [get_model_preset(name).num_layers for name in list_model_presets()]
        self.assertEqual(widths, sorted(widths))
        self.assertEqual(layers, sorted(layers))

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
        with self.assertRaisesRegex(ValueError, "tiny, small, medium, large, xlarge"):
            get_model_preset("micro")

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
