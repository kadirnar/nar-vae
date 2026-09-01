"""Compatibility tests for existing complete-waveform benchmark output."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nar_vae.benchmark import run_benchmark
from nar_vae.checkpoint import CheckpointProvenance
from nar_vae.tokenization import TextSpan


class DummyFlowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class DummyRealtimeTTS:
    def __init__(self, checkpoint: Path, **kwargs):
        del kwargs
        self.device = torch.device("cpu")
        self.flow_model = DummyFlowModel()
        self.supports_voice_cloning = False
        self.supports_multilingual = False
        self.supports_cross_lingual = False
        self.uses_learned_duration = False
        self.supported_languages = ("en",)
        self.supported_reference_languages = ()
        self.checkpoint_provenance = CheckpointProvenance(
            kind="local",
            source=str(checkpoint),
            requested_revision=None,
            resolved_revision=None,
            base_filename=checkpoint.name,
            ema_filename=None,
            selected_filename=checkpoint.name,
            path=checkpoint,
            base_path=checkpoint,
        )
        self.sample_rate = 16
        self.hop_length = 4
        self.last_cache_stats = SimpleNamespace(
            version=None,
            cached_steps=0,
            executed_steps=0,
            cache_ratio=0.0,
            baseline_block_calls=0,
            estimated_block_calls=0,
            block_work_reduction=0.0,
        )
        self.closed = False
        self.calls = []

    @staticmethod
    def _effective_cfg(**kwargs):
        return (
            kwargs["cfg_scale"],
            kwargs["cfg_mode"],
            kwargs["cfg_scale_text"],
            kwargs["cfg_scale_speaker"],
        )

    def synthesize_fast(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return torch.zeros(16), {
            "ttft": 0.1,
            "ttfa": 0.7,
            "conditioning": 0.1,
            "ode_sampling": 0.4,
            "decoding": 0.15,
            "output_transfer": 0.05,
            "total": 0.7,
            "audio_duration": 1.0,
        }

    def close(self):
        self.closed = True


class BenchmarkCompatibilityTest(unittest.TestCase):
    def test_run_benchmark_preserves_legacy_fields_and_adds_non_streaming_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "result.json"
            runtime = DummyRealtimeTTS(checkpoint)
            with (
                patch("nar_vae.benchmark.RealtimeTTSInference", return_value=runtime),
                patch("nar_vae.benchmark.package_source_hashes", return_value={}),
                patch("nar_vae.benchmark.environment", return_value={"cuda_available": False}),
            ):
                result = run_benchmark(
                    checkpoint=checkpoint,
                    dacvae_model=root / "codec",
                    device="cpu",
                    warmup_runs=1,
                    runs=2,
                    output=output,
                    language="tr",
                    language_spans=(TextSpan(text="", language="tr", phonemes=("m", "e", "r")),),
                )

        measurement = result["measurements"][0]
        self.assertEqual(measurement["ttft"], 0.1)
        self.assertEqual(measurement["ttfa"], 0.7)
        self.assertEqual(measurement["total"], 0.7)
        self.assertEqual(measurement["stages"]["generation_s"], 0.4)
        self.assertEqual(result["summary"]["ttfa"]["median_s"], 0.7)
        self.assertEqual(result["summary"]["stages"]["ttfa"]["p99_s"], 0.7)
        self.assertFalse(result["definitions"]["streaming"])
        self.assertEqual(result["evidence"]["result_kind"], "complete_waveform_benchmark")
        self.assertFalse(result["evidence"]["named_gpu_streaming_evidence"])
        self.assertEqual(
            result["configuration"]["text_conditioning"],
            {
                "phonemes": None,
                "language_spans": [
                    {
                        "text": "",
                        "language": "tr",
                        "normalized_text": None,
                        "phonemes": ["m", "e", "r"],
                    }
                ],
            },
        )
        self.assertEqual(len(runtime.calls), 3)
        for _, kwargs in runtime.calls:
            self.assertEqual(kwargs["language"], "tr")
            self.assertEqual(
                kwargs["language_spans"],
                (TextSpan(text="", language="tr", phonemes=("m", "e", "r")),),
            )
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
