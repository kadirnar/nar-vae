"""Formatting tests for the solver-duration comparison."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from nar_vae.benchmark_solvers import _markdown_table, compare_solvers
from nar_vae.checkpoint import CheckpointProvenance, HubCheckpointSource
from nar_vae.configuration import SOLVER_NFE_PER_STEP, SOLVERS


class DummyFlowModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))


class DummyRealtimeTTS:
    def __init__(self, provenance):
        self.checkpoint_provenance = provenance
        self.device = torch.device("cpu")
        self.flow_model = DummyFlowModel()
        self.supports_voice_cloning = False
        self.supports_multilingual = True
        self.supports_cross_lingual = False
        self.uses_learned_duration = True
        self.uses_mas_duration = True
        self.supported_languages = ("en", "tr")
        self.supported_reference_languages = ()
        self.sample_rate = 16
        self.hop_length = 4
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
        }

    @staticmethod
    def save_audio(audio, path):
        del audio
        Path(path).write_bytes(b"wave")

    def close(self):
        self.closed = True


class SolverBenchmarkTest(unittest.TestCase):
    def test_table_contains_every_solver(self):
        result = {
            "configuration": {"num_steps": 16, "measured_runs": 3},
            "solvers": {
                solver: {
                    "num_steps": 16,
                    "nfe": 16,
                    "summary": {
                        "ttft": {"median_s": 0.1},
                        "ttfa": {"median_s": 1.0, "p95_s": 1.1},
                    },
                    "median_rtf": 0.5,
                    "quality": {"word_error_rate": 0.0, "passed": None},
                }
                for solver in SOLVERS
            },
        }

        table = _markdown_table(result)

        for solver in SOLVERS:
            self.assertIn(f"| {solver} |", table)
        self.assertIn("| NFE |", table)
        self.assertIn("| WER | WER gate |", table)
        self.assertIn("| 0.0% | threshold not set |", table)

    def test_nfe_costs_cover_every_solver(self):
        self.assertEqual(set(SOLVER_NFE_PER_STEP), set(SOLVERS))
        self.assertEqual(SOLVER_NFE_PER_STEP["euler"], 1)
        self.assertEqual(SOLVER_NFE_PER_STEP["midpoint"], 2)
        self.assertEqual(SOLVER_NFE_PER_STEP["heun"], 2)
        self.assertEqual(SOLVER_NFE_PER_STEP["rk4"], 4)

    def test_nfe_budget_must_be_unambiguous_and_evenly_divisible(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            compare_solvers(checkpoint="unused.bin", num_steps=16, nfe_budget=16)
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            compare_solvers(checkpoint="unused.bin", nfe_budget=3)

    def test_wer_threshold_requires_asr_evaluation(self):
        with self.assertRaisesRegex(ValueError, "requires evaluate_asr"):
            compare_solvers(checkpoint="unused.bin", maximum_wer=0.2)

    def test_pinned_source_and_runtime_provenance_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "ema.bin"
            selected.write_bytes(b"ema")
            base = root / "base.bin"
            base.write_bytes(b"base")
            revision = "a" * 40
            source = HubCheckpointSource(
                repo_id="owner/model",
                revision=revision,
                base_filename="weights/base.bin",
                ema_filename="weights/ema.bin",
            )
            provenance = CheckpointProvenance(
                kind="huggingface_hub",
                source=source.repo_id,
                requested_revision=revision,
                resolved_revision=revision,
                base_filename=source.base_filename,
                ema_filename=source.ema_filename,
                selected_filename=source.ema_filename,
                path=selected,
                base_path=base,
            )
            runtime = DummyRealtimeTTS(provenance)
            output = root / "result.json"
            markdown = root / "result.md"
            audio_dir = root / "audio"

            with (
                patch(
                    "nar_vae.benchmark_solvers.RealtimeTTSInference",
                    return_value=runtime,
                ) as runtime_factory,
                patch("nar_vae.benchmark_solvers.package_source_hashes", return_value={}),
                patch(
                    "nar_vae.benchmark_solvers.environment",
                    return_value={"cuda_available": False},
                ),
            ):
                result = compare_solvers(
                    checkpoint=source,
                    dacvae_model=root / "codec",
                    device="cpu",
                    warmup_runs=0,
                    runs=1,
                    output=output,
                    markdown=markdown,
                    audio_dir=audio_dir,
                    language="es",
                    reference_audio=torch.ones(32),
                    reference_sample_rate=16,
                    reference_language="en",
                    phonemes=("o", "l", "a"),
                )

        self.assertIs(runtime_factory.call_args.kwargs["flow_model_path"], source)
        self.assertEqual(result["schema_version"], 5)
        self.assertEqual(result["model"]["source_kind"], "huggingface_hub")
        self.assertEqual(result["model"]["hf_id"], "owner/model")
        self.assertEqual(result["model"]["requested_revision"], revision)
        self.assertEqual(result["model"]["revision"], revision)
        self.assertEqual(result["model"]["selected_filename"], "weights/ema.bin")
        self.assertEqual(result["model"]["artifacts"]["selected"]["path"], "weights/ema.bin")
        self.assertEqual(result["model"]["artifacts"]["base"]["path"], "weights/base.bin")
        self.assertTrue(result["model"]["capabilities"]["monotonic_alignment"])
        self.assertEqual(
            result["configuration"]["language_pair"],
            {"target": "es", "reference": "en", "cross_lingual": True},
        )
        self.assertEqual(
            result["configuration"]["text_conditioning"],
            {"phonemes": ["o", "l", "a"], "language_spans": None},
        )
        self.assertEqual(len(runtime.calls), len(SOLVERS))
        for _, kwargs in runtime.calls:
            self.assertEqual(kwargs["language"], "es")
            self.assertEqual(kwargs["reference_language"], "en")
            self.assertEqual(kwargs["phonemes"], ("o", "l", "a"))
            torch.testing.assert_close(kwargs["reference_audio"], torch.ones(32))
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
