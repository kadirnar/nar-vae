"""CPU-only tests for deterministic variable-length batch sampling."""

import json
import unittest

from vyvotts.dataset import (
    LATENT_NUM_FRAMES_COLUMN,
    FrameBudgetBatchSampler,
    read_dataset_frame_lengths,
)


def flatten(batches):
    return [index for batch in batches for index in batch]


class FrameBudgetBatchSamplerTest(unittest.TestCase):
    def test_persisted_frame_column_does_not_load_latent_rows(self):
        class MetadataOnlyDataset:
            column_names = (LATENT_NUM_FRAMES_COLUMN, "latents")

            def __len__(self):
                return 3

            def __getitem__(self, key):
                if key == LATENT_NUM_FRAMES_COLUMN:
                    return [11, 7, 5]
                raise AssertionError("The preferred path must not deserialize latent rows")

        self.assertEqual(read_dataset_frame_lengths(MetadataOnlyDataset()), [11, 7, 5])

    def test_legacy_frame_inference_requires_an_explicit_opt_in(self):
        legacy = [
            {"latents": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]},
            {"latents": [[0.0, 0.0], [0.0, 0.0]]},
        ]

        with self.assertRaisesRegex(ValueError, "allow_legacy_inference=True"):
            read_dataset_frame_lengths(legacy)
        self.assertEqual(
            read_dataset_frame_lengths(legacy, allow_legacy_inference=True),
            [3, 2],
        )

    def test_invalid_persisted_frame_metadata_is_rejected(self):
        class InvalidMetadataDataset:
            column_names = (LATENT_NUM_FRAMES_COLUMN,)

            def __len__(self):
                return 2

            def __getitem__(self, key):
                if key == LATENT_NUM_FRAMES_COLUMN:
                    return [4, 0]
                raise KeyError(key)

        with self.assertRaisesRegex(ValueError, r"latent_num_frames\[1\]"):
            read_dataset_frame_lengths(InvalidMetadataDataset())

    def test_batches_respect_frame_and_example_budgets(self):
        lengths = [9, 2, 5, 7, 3, 8, 4, 6, 1]
        sampler = FrameBudgetBatchSampler(
            lengths,
            max_frames=12,
            max_examples=3,
            bucket_size=4,
            seed=17,
        )

        batches = list(sampler)

        self.assertEqual(sorted(flatten(batches)), list(range(len(lengths))))
        self.assertEqual(len(flatten(batches)), len(set(flatten(batches))))
        for batch in batches:
            self.assertLessEqual(sum(lengths[index] for index in batch), 12)
            self.assertLessEqual(len(batch), 3)

    def test_accepts_one_pass_length_iterables(self):
        sampler = FrameBudgetBatchSampler(
            (length for length in [2, 3, 4]),
            max_frames=5,
            shuffle=False,
        )

        self.assertEqual(list(sampler), [[0, 1], [2]])

    def test_seed_and_epoch_produce_reproducible_sortish_plans(self):
        lengths = [(index * 7) % 19 + 1 for index in range(40)]

        def plan(seed, epoch):
            return list(
                FrameBudgetBatchSampler(
                    lengths,
                    max_frames=38,
                    bucket_size=8,
                    seed=seed,
                    epoch=epoch,
                )
            )

        self.assertEqual(plan(123, 4), plan(123, 4))
        self.assertNotEqual(plan(123, 4), plan(123, 5))
        self.assertNotEqual(plan(123, 4), plan(124, 4))

    def test_set_epoch_rewinds_the_dispatch_cursor(self):
        sampler = FrameBudgetBatchSampler(
            [4, 5, 6, 7, 8, 9],
            max_frames=12,
            seed=9,
        )
        original = list(sampler)
        self.assertEqual(sampler.remaining_batches, 0)

        sampler.set_epoch(0)

        self.assertEqual(sampler.next_batch, 0)
        self.assertEqual(list(sampler), original)

    def test_distributed_shards_are_disjoint_without_padding(self):
        lengths = [5] * 9
        rank_batches = [
            list(
                FrameBudgetBatchSampler(
                    lengths,
                    max_frames=10,
                    shuffle=False,
                    rank=rank,
                    world_size=3,
                )
            )
            for rank in range(3)
        ]
        rank_indices = [set(flatten(batches)) for batches in rank_batches]

        self.assertEqual(set.union(*rank_indices), set(range(len(lengths))))
        for left in range(3):
            for right in range(left + 1, 3):
                self.assertTrue(rank_indices[left].isdisjoint(rank_indices[right]))
        self.assertEqual([len(batches) for batches in rank_batches], [2, 2, 1])

    def test_drop_last_makes_distributed_batch_counts_equal_without_duplicates(self):
        lengths = [5] * 5
        samplers = [
            FrameBudgetBatchSampler(
                lengths,
                max_frames=5,
                shuffle=False,
                rank=rank,
                world_size=2,
                drop_last=True,
            )
            for rank in range(2)
        ]
        batches = [list(sampler) for sampler in samplers]
        rank_indices = [set(flatten(items)) for items in batches]

        self.assertEqual([len(items) for items in batches], [2, 2])
        self.assertTrue(rank_indices[0].isdisjoint(rank_indices[1]))
        self.assertEqual(len(set.union(*rank_indices)), 4)

    def test_explicit_padding_equalizes_ranks_by_repeating_batches(self):
        lengths = [5] * 5
        batches = [
            list(
                FrameBudgetBatchSampler(
                    lengths,
                    max_frames=5,
                    shuffle=False,
                    rank=rank,
                    world_size=2,
                    pad_to_world_size=True,
                )
            )
            for rank in range(2)
        ]
        flattened = [flatten(items) for items in batches]

        self.assertEqual([len(items) for items in batches], [3, 3])
        self.assertEqual(set(flattened[0]) | set(flattened[1]), set(range(5)))
        self.assertEqual(len(flattened[0]) + len(flattened[1]), 6)
        self.assertFalse(set(flattened[0]).isdisjoint(flattened[1]))

    def test_state_dict_resumes_at_the_exact_next_batch(self):
        kwargs = {
            "max_frames": 15,
            "max_examples": 3,
            "bucket_size": 5,
            "seed": 42,
            "epoch": 7,
        }
        sampler = FrameBudgetBatchSampler([3, 8, 5, 6, 2, 9, 7, 4], **kwargs)
        iterator = iter(sampler)
        first_batch = next(iterator)
        state = sampler.state_dict()
        expected_remaining = list(iterator)

        resumed = FrameBudgetBatchSampler([3, 8, 5, 6, 2, 9, 7, 4], **kwargs)
        resumed.load_state_dict(json.loads(json.dumps(state)))

        self.assertTrue(first_batch)
        self.assertEqual(resumed.next_batch, 1)
        self.assertEqual(resumed.remaining_batches, len(expected_remaining))
        self.assertEqual(list(resumed), expected_remaining)

    def test_state_rejects_changed_configuration_or_plan(self):
        sampler = FrameBudgetBatchSampler([3, 4, 5, 6], max_frames=8, seed=1)
        state = sampler.state_dict()

        changed = FrameBudgetBatchSampler([3, 4, 5, 6], max_frames=9, seed=1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            changed.load_state_dict(state)

        before_tamper = sampler.state_dict()
        tampered = dict(state, plan_digest="not-the-plan")
        with self.assertRaisesRegex(ValueError, "plan does not match"):
            sampler.load_state_dict(tampered)
        self.assertEqual(sampler.state_dict(), before_tamper)

    def test_invalid_inputs_fail_before_iteration(self):
        invalid_cases = (
            ([], {"max_frames": 10}, "at least one"),
            ([1, 0], {"max_frames": 10}, r"lengths\[1\]"),
            ([11], {"max_frames": 10}, "exceeds"),
            ([1], {"max_frames": 0}, "max_frames"),
            ([1], {"max_frames": 10, "max_examples": 0}, "max_examples"),
            ([1], {"max_frames": 10, "bucket_size": 0}, "bucket_size"),
            ([1], {"max_frames": 10, "rank": 2, "world_size": 2}, "rank"),
        )
        for lengths, kwargs, message in invalid_cases:
            with self.subTest(lengths=lengths, kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    FrameBudgetBatchSampler(lengths, **kwargs)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            FrameBudgetBatchSampler(
                [1],
                max_frames=10,
                drop_last=True,
                pad_to_world_size=True,
            )


if __name__ == "__main__":
    unittest.main()
