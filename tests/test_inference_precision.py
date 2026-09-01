"""Inference precision contracts that preserve codec and model-buffer identity."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from nar_vae.checkpoint import (
    DurationCheckpointInfo,
    LanguageCheckpointInfo,
    MonotonicAlignmentCheckpointInfo,
    ReferenceLanguageCheckpointInfo,
    TextConditioningCheckpointInfo,
)
from nar_vae.inference import (
    FlowMatchingTTSInference,
    _cast_acoustic_parameters,
    _normalize_acoustic_dtype,
)
from nar_vae.inference_realtime import RealtimeTTSInference
from nar_vae.model_manifest import text_conditioning_from_config
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.objectives import VP_DIFFUSION_OBJECTIVE


def _tiny_vp_model() -> FlowMatchingEchoDiT:
    return FlowMatchingEchoDiT(
        latent_size=4,
        model_size=8,
        num_layers=1,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=20,
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
        generative_objective=VP_DIFFUSION_OBJECTIVE,
    )


class _FP32Codec(torch.nn.Module):
    sample_rate = 16
    hop_length = 4

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.last_latents = None

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.last_latents = latents
        return latents


def _checkpoint() -> Mock:
    checkpoint = Mock()
    checkpoint.path = Path("precision.bin")
    checkpoint.provenance = None
    checkpoint.generative_objective.return_value = VP_DIFFUSION_OBJECTIVE
    checkpoint.diffusion_schedule_shift.return_value = 1.0
    checkpoint.text_conditioning.return_value = TextConditioningCheckpointInfo("scratch_tokens")
    checkpoint.infer_target_patch_size.return_value = 1
    checkpoint.infer_text_vocab_size.return_value = 20
    checkpoint.infer_speaker_conditioning.return_value = False
    checkpoint.infer_speaker_num_summary_tokens.return_value = 0
    checkpoint.language_capability.return_value = LanguageCheckpointInfo(False)
    checkpoint.reference_language_capability.return_value = ReferenceLanguageCheckpointInfo(False)
    checkpoint.duration_capability.return_value = DurationCheckpointInfo(False)
    checkpoint.monotonic_alignment_capability.return_value = MonotonicAlignmentCheckpointInfo(False)
    return checkpoint


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        text_conditioning=text_conditioning_from_config({}),
        representation={
            "codec_source": "codec.pth",
            "codec_revision": None,
            "codec_filename": "codec.pth",
            "codec_sha256": "a" * 64,
        },
    )


def test_dtype_aliases_are_narrow_and_invalid_values_fail_before_checkpoint_loading() -> None:
    assert _normalize_acoustic_dtype("fp32") is torch.float32
    assert _normalize_acoustic_dtype("float32") is torch.float32
    assert _normalize_acoustic_dtype("BF16") is torch.bfloat16
    assert _normalize_acoustic_dtype("bfloat16") is torch.bfloat16
    assert _normalize_acoustic_dtype(torch.float32) is torch.float32
    assert _normalize_acoustic_dtype(torch.bfloat16) is torch.bfloat16

    for invalid in ("float16", torch.float16, None, 16):
        with (
            patch("nar_vae.inference.FlowCheckpoint.load") as load_checkpoint,
            pytest.raises(ValueError, match="acoustic_dtype"),
        ):
            FlowMatchingTTSInference("checkpoint.bin", acoustic_dtype=invalid)
        load_checkpoint.assert_not_called()


def test_parameter_only_cast_preserves_every_buffer_exactly() -> None:
    model = _tiny_vp_model()
    before = {
        name: (buffer.detach().clone(), buffer.dtype) for name, buffer in model.named_buffers()
    }

    _cast_acoustic_parameters(model, torch.bfloat16)

    assert all(parameter.dtype is torch.bfloat16 for parameter in model.parameters())
    after = dict(model.named_buffers())
    assert after.keys() == before.keys()
    for name, buffer in after.items():
        expected, expected_dtype = before[name]
        assert buffer.dtype is expected_dtype
        torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
    rope_caches = [buffer for name, buffer in after.items() if name.endswith("_rope_cache")]
    assert rope_caches
    assert all(buffer.is_complex() for buffer in rope_caches)
    assert after["diffusion_schedule_shift_metadata"].dtype is torch.float64


def test_public_runtime_casts_only_acoustic_parameters_and_decodes_in_codec_dtype() -> None:
    model = _tiny_vp_model()
    before_buffers = {
        name: (buffer.detach().clone(), buffer.dtype) for name, buffer in model.named_buffers()
    }
    codec = _FP32Codec()
    checkpoint = _checkpoint()
    events = []
    checkpoint.load_into.side_effect = lambda loaded_model: events.append(("load", loaded_model))
    original_to = model.to

    def record_cast(loaded_model, dtype):
        events.append(("cast", loaded_model))
        _cast_acoustic_parameters(loaded_model, dtype)

    def record_move(*args, **kwargs):
        events.append(("move", args[0] if args else kwargs.get("device")))
        return original_to(*args, **kwargs)

    with (
        patch("nar_vae.inference.FlowCheckpoint.load", return_value=checkpoint),
        patch("nar_vae.inference.create_flow_matching_echodit", return_value=model),
        patch("nar_vae.inference._cast_acoustic_parameters", side_effect=record_cast),
        patch.object(model, "to", side_effect=record_move),
        patch("nar_vae.inference.load_model_manifest", return_value=_manifest()),
        patch("nar_vae.inference.validate_inference_manifest"),
        patch("nar_vae.inference.load_dacvae", return_value=codec),
        patch("nar_vae.inference.validate_loaded_codec"),
    ):
        runtime = FlowMatchingTTSInference(
            "precision.bin",
            dacvae_model="codec.pth",
            device="cpu",
            latent_size=4,
            acoustic_dtype="bf16",
        )

    assert [name for name, _ in events[:3]] == ["load", "cast", "move"]
    assert runtime.acoustic_dtype is torch.bfloat16
    assert all(parameter.dtype is torch.bfloat16 for parameter in runtime.flow_model.parameters())
    for name, buffer in runtime.flow_model.named_buffers():
        expected, expected_dtype = before_buffers[name]
        assert buffer.dtype is expected_dtype
        torch.testing.assert_close(buffer, expected, rtol=0, atol=0)
    assert next(codec.parameters()).dtype is torch.float32

    decoded = runtime._decode_generated_latents(torch.ones(1, 4, 2, dtype=torch.bfloat16))
    assert codec.last_latents is not None
    assert codec.last_latents.dtype is torch.float32
    assert decoded.dtype is torch.float32


def test_realtime_and_preset_factories_thread_the_public_dtype_option() -> None:
    received = {}

    def fake_base_init(runtime, **kwargs):
        received.update(kwargs)
        runtime.device = torch.device("cpu")

    with patch(
        "nar_vae.inference_realtime.FlowMatchingTTSInference.__init__",
        new=fake_base_init,
    ):
        RealtimeTTSInference(
            "checkpoint.bin",
            device="cpu",
            acoustic_dtype="bfloat16",
        )
    assert received["acoustic_dtype"] == "bfloat16"

    class RecordingInference(FlowMatchingTTSInference):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    runtime = RecordingInference.from_preset(
        "nano",
        flow_model_path="checkpoint.bin",
        acoustic_dtype="bf16",
    )
    assert runtime.kwargs["acoustic_dtype"] == "bf16"
