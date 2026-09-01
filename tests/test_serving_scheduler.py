"""Deterministic admission, deadline, batching, and priority tests."""

import unittest
from dataclasses import replace

from vyvotts.serving import (
    DeadlineBatchScheduler,
    ManualClock,
    RequestMetadata,
    RequestStatus,
    SchedulerConfig,
    ShapeBucketKey,
    WorkKind,
)


def bucket_key(**overrides):
    values = {
        "checkpoint": "checkpoint-a",
        "generation_profile": "fast",
        "precision": "bf16",
        "text_bucket": 64,
        "reference_bucket": 0,
        "latent_bucket": 256,
        "block_bucket": 32,
        "solver": "euler",
        "step_index": 0,
        "step_count": 16,
        "cfg_layout": "joint",
    }
    values.update(overrides)
    return ShapeBucketKey(**values)


def request(
    request_id,
    key,
    *,
    arrival=0.0,
    deadline=1.0,
    blocks=1,
    interval=None,
):
    return RequestMetadata(
        request_id=request_id,
        client_id=f"client-{request_id}",
        arrival_time_s=arrival,
        first_audio_deadline_s=deadline,
        bucket_key=key,
        total_blocks=blocks,
        continuation_interval_s=interval,
    )


class DeadlineBatchSchedulerTest(unittest.TestCase):
    def test_batches_only_exact_keys_and_honors_batch_limit(self):
        clock = ManualClock()
        scheduler = DeadlineBatchScheduler(
            SchedulerConfig(max_batch_size=2, max_queue_delay_s=1.0),
            clock=clock,
        )
        key_a = bucket_key()
        key_b = replace(key_a, precision="fp16")
        for request_id, key in (("a-1", key_a), ("a-2", key_a), ("a-3", key_a), ("b", key_b)):
            scheduler.admit(request(request_id, key))

        first = scheduler.next_batch()
        second = scheduler.next_batch()
        third = scheduler.next_batch()

        self.assertEqual([item.request_id for item in first.items], ["a-1", "a-2"])
        self.assertEqual([item.request_id for item in second.items], ["a-3"])
        self.assertEqual([item.request_id for item in third.items], ["b"])
        self.assertTrue(all(item.bucket_key == first.bucket_key for item in first.items))
        self.assertLessEqual(len(first.items), scheduler.config.max_batch_size)

    def test_queue_delay_limit_expires_waiting_first_block(self):
        clock = ManualClock()
        scheduler = DeadlineBatchScheduler(
            SchedulerConfig(max_queue_delay_s=0.01),
            clock=clock,
        )
        scheduler.admit(request("waiting", bucket_key(), deadline=1.0))

        clock.advance(0.01)
        expired = scheduler.expire()

        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].status, RequestStatus.TIMED_OUT)
        self.assertEqual(expired[0].reason, "queue_delay_limit_expired")
        self.assertEqual(scheduler.pending_count, 0)

    def test_expired_continuation_deadline_times_out_stream(self):
        clock = ManualClock()
        scheduler = DeadlineBatchScheduler(
            SchedulerConfig(max_queue_delay_s=1.0),
            clock=clock,
        )
        scheduler.admit(request("stream", bucket_key(), blocks=2, interval=0.01))
        scheduler.complete_batch(scheduler.next_batch())

        clock.advance(0.01)
        expired = scheduler.expire()

        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].status, RequestStatus.TIMED_OUT)
        self.assertEqual(expired[0].reason, "continuation_deadline_expired")

    def test_expired_deadline_and_capacity_are_rejected_at_admission(self):
        clock = ManualClock(0.1)
        scheduler = DeadlineBatchScheduler(
            SchedulerConfig(max_active_requests=1, max_queue_delay_s=1.0),
            clock=clock,
        )
        expired = scheduler.admit(request("expired", bucket_key(), arrival=0.0, deadline=0.05))
        accepted = scheduler.admit(request("accepted", bucket_key(), arrival=0.1, deadline=1.0))
        overloaded = scheduler.admit(request("overloaded", bucket_key(), arrival=0.1, deadline=1.0))

        self.assertFalse(expired.admitted)
        self.assertEqual(expired.reason, "first_audio_deadline_expired")
        self.assertTrue(accepted.admitted)
        self.assertFalse(overloaded.admitted)
        self.assertEqual(overloaded.reason, "capacity_exhausted")

    def test_first_blocks_have_priority_and_continuations_make_progress(self):
        clock = ManualClock()
        scheduler = DeadlineBatchScheduler(
            SchedulerConfig(
                max_batch_size=1,
                max_queue_delay_s=1.0,
                max_first_block_batches=2,
            ),
            clock=clock,
        )
        scheduler.admit(request("stream", bucket_key(), blocks=2, interval=0.5))
        stream_first = scheduler.next_batch()
        scheduler.complete_batch(stream_first)
        scheduler.admit(request("new-1", bucket_key()))
        scheduler.admit(request("new-2", bucket_key()))

        prioritized_first = scheduler.next_batch()
        fair_continuation = scheduler.next_batch()

        self.assertEqual(prioritized_first.kind, WorkKind.FIRST_BLOCK)
        self.assertEqual(prioritized_first.items[0].request_id, "new-1")
        self.assertEqual(fair_continuation.kind, WorkKind.CONTINUATION)
        self.assertEqual(fair_continuation.items[0].request_id, "stream")

    def test_equal_deadlines_use_stable_admission_order(self):
        clock = ManualClock()
        scheduler = DeadlineBatchScheduler(
            SchedulerConfig(max_batch_size=1, max_queue_delay_s=1.0),
            clock=clock,
        )
        scheduler.admit(request("z-first", bucket_key()))
        scheduler.admit(request("a-second", bucket_key()))

        self.assertEqual(scheduler.next_batch().items[0].request_id, "z-first")
        self.assertEqual(scheduler.next_batch().items[0].request_id, "a-second")


if __name__ == "__main__":
    unittest.main()
