"""Tests for dependency-free serving metadata and timing records."""

import json
import subprocess
import sys
import unittest
from dataclasses import replace

from nar_vae.serving import (
    RequestMetadata,
    ShapeBucketKey,
    StageTiming,
    percentile,
    summarize_percentiles,
)


def bucket_key(**overrides):
    values = {
        "checkpoint": "checkpoint@sha256:abc",
        "generation_profile": "fast",
        "precision": "bf16",
        "text_bucket": 64,
        "reference_bucket": 128,
        "latent_bucket": 512,
        "block_bucket": 32,
        "solver": "euler",
        "step_index": 0,
        "step_count": 16,
        "cfg_layout": "independent-fused",
    }
    values.update(overrides)
    return ShapeBucketKey(**values)


class ServingMetadataTest(unittest.TestCase):
    def test_bucket_identity_contains_every_compatibility_dimension(self):
        original = bucket_key()
        variants = (
            replace(original, checkpoint="other"),
            replace(original, generation_profile="quality"),
            replace(original, precision="fp16"),
            replace(original, text_bucket=128),
            replace(original, reference_bucket=256),
            replace(original, latent_bucket=1024),
            replace(original, block_bucket=64),
            replace(original, solver="heun"),
            replace(original, step_index=1),
            replace(original, step_count=32),
            replace(original, cfg_layout="joint"),
        )

        self.assertEqual(len({original, *variants}), len(variants) + 1)
        self.assertEqual(json.loads(original.stable_id), original.to_dict())

    def test_request_keeps_target_and_reference_languages_independent(self):
        request = RequestMetadata(
            request_id="request-1",
            client_id="client-1",
            arrival_time_s=1.0,
            first_audio_deadline_s=1.1,
            bucket_key=bucket_key(),
            target_language="Turkish",
            reference_language="Spanish",
        )

        record = request.to_dict()
        self.assertEqual(record["target_language"], "tr")
        self.assertEqual(record["reference_language"], "es")
        self.assertEqual(
            record["language_pair"],
            {"target": "tr", "reference": "es", "cross_lingual": True},
        )
        json.dumps(record, allow_nan=False)

    def test_serving_package_import_does_not_load_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, nar_vae.serving; assert 'torch' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


class ServingTimingTest(unittest.TestCase):
    def test_percentiles_use_linear_interpolation_and_include_p99(self):
        values = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(percentile(values, 0.5), 2.5)
        self.assertAlmostEqual(percentile(values, 0.95), 3.85)
        summary = summarize_percentiles(values)
        self.assertAlmostEqual(summary["p99_s"], 3.97)

    def test_complete_waveform_adapter_preserves_legacy_values(self):
        legacy = {
            "ttft": 0.01,
            "ttfa": 0.50,
            "conditioning": 0.02,
            "ode_sampling": 0.40,
            "decoding": 0.06,
            "output_transfer": 0.02,
            "total": 0.50,
        }

        stages = StageTiming.from_complete_waveform_timings(legacy)

        self.assertEqual(stages.queue_s, 0.0)
        self.assertEqual(stages.generation_s, legacy["ode_sampling"])
        self.assertEqual(stages.decode_s, legacy["decoding"])
        self.assertEqual(stages.transfer_s, legacy["output_transfer"])
        self.assertEqual(stages.ttfa_s, legacy["ttfa"])
        self.assertEqual(stages.result_kind, "complete_waveform")
        self.assertEqual(legacy["ttfa"], legacy["total"])


if __name__ == "__main__":
    unittest.main()
