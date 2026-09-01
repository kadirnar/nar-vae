"""Synthetic load report, clock injection, and serialization tests."""

import json
import tempfile
import unittest
from pathlib import Path

from nar_vae.serving import (
    DEFAULT_CLIENT_COUNTS,
    ArrivalPattern,
    ManualClock,
    SchedulerConfig,
    SyntheticLoadHarness,
    SyntheticServiceTimes,
    SyntheticWorkload,
    run_synthetic_load_suite,
    write_json_result,
)


class SyntheticLoadHarnessTest(unittest.TestCase):
    def test_standard_suite_is_deterministic_json_and_keeps_language_roles(self):
        first = run_synthetic_load_suite(target_language="tr", reference_language="es")
        second = run_synthetic_load_suite(target_language="tr", reference_language="es")

        self.assertEqual(first, second)
        self.assertEqual(first["client_counts"], list(DEFAULT_CLIENT_COUNTS))
        self.assertEqual(
            set(first["scenarios"]),
            {"synchronized_burst", "steady_stream"},
        )
        for scenario in first["scenarios"].values():
            self.assertEqual(set(scenario), {str(value) for value in DEFAULT_CLIENT_COUNTS})
            for report in scenario.values():
                self.assertTrue(report["evidence"]["synthetic"])
                self.assertFalse(report["evidence"]["named_gpu_streaming_evidence"])
                self.assertEqual(report["workload"]["target_language"], "tr")
                self.assertEqual(report["workload"]["reference_language"], "es")
                self.assertIn("p99_s", report["latency"]["ttfa"])
                self.assertIn("size_distribution", report["batches"])
        json.dumps(first, allow_nan=False)

    def test_injected_clock_runs_without_sleep_and_respects_batch_cap(self):
        clock = ManualClock(5.0)
        harness = SyntheticLoadHarness(
            clock=clock,
            scheduler_config=SchedulerConfig(max_batch_size=3, max_queue_delay_s=1.0),
        )
        report = harness.run(
            SyntheticWorkload(
                pattern=ArrivalPattern.SYNCHRONIZED_BURST,
                clients=8,
                blocks_per_request=1,
                first_audio_budget_s=1.0,
                target_language="ja",
                reference_language="en",
            )
        )

        self.assertGreater(clock(), 5.0)
        self.assertEqual(report["counts"]["completed"], 8)
        self.assertLessEqual(report["batches"]["maximum_size"], 3)
        for row in report["request_results"]:
            self.assertEqual(row["target_language"], "ja")
            self.assertEqual(row["reference_language"], "en")

    def test_report_counts_rejections_and_deadline_failures(self):
        capacity_report = SyntheticLoadHarness(
            clock=ManualClock(),
            scheduler_config=SchedulerConfig(
                max_batch_size=1,
                max_active_requests=1,
                max_queue_delay_s=1.0,
            ),
        ).run(
            SyntheticWorkload(
                pattern=ArrivalPattern.SYNCHRONIZED_BURST,
                clients=2,
                blocks_per_request=1,
                first_audio_budget_s=1.0,
            )
        )
        timeout_report = SyntheticLoadHarness(
            clock=ManualClock(),
            scheduler_config=SchedulerConfig(max_batch_size=1, max_queue_delay_s=0.01),
            service_times=SyntheticServiceTimes(first_generation_s=0.02),
        ).run(
            SyntheticWorkload(
                pattern=ArrivalPattern.SYNCHRONIZED_BURST,
                clients=2,
                blocks_per_request=1,
                first_audio_budget_s=0.01,
            )
        )

        self.assertEqual(capacity_report["rejections"]["total"], 1)
        self.assertEqual(capacity_report["counts"]["completed"], 1)
        self.assertEqual(timeout_report["counts"]["timed_out"], 2)
        self.assertEqual(timeout_report["failures"]["total"], 2)

    def test_json_writer_round_trips_report(self):
        report = run_synthetic_load_suite(client_counts=(1,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "load.json"
            returned = write_json_result(report, output)

            self.assertEqual(returned, output)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
