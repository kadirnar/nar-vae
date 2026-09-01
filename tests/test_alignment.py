"""CPU tests for monotonic alignment and exact duration allocation."""

import itertools
import unittest

import torch

from nar_vae.models import (
    allocate_integer_durations,
    durations_from_alignment,
    monotonic_alignment_search,
)


class MonotonicAlignmentSearchTest(unittest.TestCase):
    def test_matches_brute_force_optimum(self):
        torch.manual_seed(17)
        token_count = 3
        frame_count = 6
        scores = torch.randn(token_count, frame_count)

        alignment = monotonic_alignment_search(scores)
        durations = durations_from_alignment(alignment)
        selected_score = scores[alignment].sum()

        candidate_scores = []
        for boundaries in itertools.combinations(range(1, frame_count), token_count - 1):
            edges = (0, *boundaries, frame_count)
            token_path = []
            for token_index in range(token_count):
                token_path.extend([token_index] * (edges[token_index + 1] - edges[token_index]))
            candidate_scores.append(
                scores[
                    torch.tensor(token_path),
                    torch.arange(frame_count),
                ].sum()
            )

        torch.testing.assert_close(selected_score, torch.stack(candidate_scores).max())
        self.assertEqual(durations.sum().item(), frame_count)
        self.assertTrue((durations > 0).all())
        self.assertEqual(alignment.dtype, torch.bool)

    def test_masks_padded_tokens_frames_and_extracts_expected_durations(self):
        scores = torch.full((2, 4, 6), -20.0)
        first_path = [0, 0, 1, 2, 2, 2]
        second_path = [0, 1, 1, 1]
        scores[0, first_path, torch.arange(6)] = 20.0
        scores[1, second_path, torch.arange(4)] = 20.0
        scores[1, 2:, :] = 1_000.0
        scores[1, :, 4:] = 1_000.0
        token_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, False, False],
            ]
        )
        frame_mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
            ]
        )

        alignment = monotonic_alignment_search(scores, token_mask, frame_mask)
        durations = durations_from_alignment(alignment, token_mask, frame_mask)

        torch.testing.assert_close(
            durations,
            torch.tensor([[2, 1, 3, 0], [1, 3, 0, 0]]),
        )
        self.assertEqual(alignment[1, 2:].count_nonzero().item(), 0)
        self.assertEqual(alignment[1, :, 4:].count_nonzero().item(), 0)

    def test_rejects_infeasible_or_nonfinite_paths(self):
        with self.assertRaisesRegex(ValueError, "at least one frame per token"):
            monotonic_alignment_search(torch.zeros(3, 2))

        scores = torch.tensor([[0.0, 0.0], [float("-inf"), float("-inf")]])
        with self.assertRaisesRegex(ValueError, "no finite monotonic path"):
            monotonic_alignment_search(scores)

        with self.assertRaisesRegex(ValueError, "NaN or positive-infinite"):
            monotonic_alignment_search(torch.tensor([[0.0, float("nan")]]))

    def test_rejects_non_prefix_masks(self):
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            monotonic_alignment_search(
                torch.zeros(2, 3),
                token_mask=torch.tensor([True, False]),
                frame_mask=torch.tensor([True, False, True]),
            )

    def test_rejects_empty_score_dimensions(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            monotonic_alignment_search(torch.empty(0, 2))


class AlignmentDurationTest(unittest.TestCase):
    def test_rejects_nonbinary_duplicate_and_nonmonotonic_alignments(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            durations_from_alignment(torch.tensor([[0.5, 0.0], [0.5, 1.0]]))

        duplicate = torch.tensor(
            [
                [True, True, False],
                [False, True, True],
            ]
        )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            durations_from_alignment(duplicate)

        nonmonotonic = torch.tensor(
            [
                [True, False, True],
                [False, True, False],
            ]
        )
        with self.assertRaisesRegex(ValueError, "monotonic"):
            durations_from_alignment(nonmonotonic)

    def test_rejects_assignments_in_padding(self):
        alignment = torch.tensor(
            [
                [True, False, False],
                [False, True, True],
            ]
        )
        with self.assertRaisesRegex(ValueError, "padded"):
            durations_from_alignment(
                alignment,
                token_mask=torch.tensor([True, False]),
                frame_mask=torch.tensor([True, True, False]),
            )

    def test_rejects_skipped_tokens_and_incomplete_endpoints(self):
        skipped = torch.tensor(
            [
                [True, False, False],
                [False, False, False],
                [False, True, True],
            ]
        )
        with self.assertRaisesRegex(ValueError, "advance by one"):
            durations_from_alignment(skipped)

        incomplete = torch.tensor(
            [
                [True, True, False],
                [False, False, True],
                [False, False, False],
            ]
        )
        with self.assertRaisesRegex(ValueError, "final token"):
            durations_from_alignment(incomplete)


class IntegerDurationAllocationTest(unittest.TestCase):
    def test_exact_largest_remainder_allocation_is_masked_and_stable(self):
        weights = torch.tensor(
            [
                [1.0, 1.0, 1.0, 999.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        token_mask = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        )

        allocation = allocate_integer_durations(
            weights,
            torch.tensor([5, 3]),
            token_mask,
        )

        torch.testing.assert_close(
            allocation,
            torch.tensor([[2, 2, 1, 0], [1, 1, 1, 0]]),
        )
        torch.testing.assert_close(allocation.sum(dim=1), torch.tensor([5, 3]))
        self.assertEqual(allocation.dtype, torch.long)
        self.assertTrue((allocation >= 0).all())

    def test_preserves_proportions_and_allows_zero_duration_tokens(self):
        exact = allocate_integer_durations(torch.tensor([0.2, 0.3, 0.5]), 10)
        scarce = allocate_integer_durations(torch.tensor([1.0, 1.0, 1.0]), 2)

        torch.testing.assert_close(exact, torch.tensor([2, 3, 5]))
        torch.testing.assert_close(scarce, torch.tensor([1, 1, 0]))

    def test_ignores_masked_weights_but_rejects_invalid_valid_values(self):
        masked_nan = allocate_integer_durations(
            torch.tensor([1.0, float("nan")]),
            3,
            torch.tensor([True, False]),
        )
        torch.testing.assert_close(masked_nan, torch.tensor([3, 0]))

        with self.assertRaisesRegex(ValueError, "finite"):
            allocate_integer_durations(torch.tensor([1.0, float("nan")]), 3)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            allocate_integer_durations(torch.tensor([1.0, -0.1]), 3)

    def test_rejects_invalid_totals(self):
        weights = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            allocate_integer_durations(weights, -1)
        with self.assertRaisesRegex(ValueError, "integral"):
            allocate_integer_durations(weights, torch.tensor(1.5))
        with self.assertRaisesRegex(TypeError, "not bool"):
            allocate_integer_durations(weights, True)


if __name__ == "__main__":
    unittest.main()
