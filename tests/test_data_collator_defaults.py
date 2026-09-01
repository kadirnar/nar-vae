"""Shared-token defaults for dataset collation."""

import unittest

import torch

from vyvotts.dataset.data_collator import (
    FlowMatchingDataCollator,
    SimpleTTSCollator,
    create_data_collator,
)
from vyvotts.tokenization import PAD_TOKEN


class DataCollatorDefaultsTest(unittest.TestCase):
    def test_every_collator_uses_shared_pad_token(self):
        self.assertEqual(FlowMatchingDataCollator().pad_token, PAD_TOKEN)
        self.assertEqual(SimpleTTSCollator().pad_token, PAD_TOKEN)
        self.assertEqual(create_data_collator().pad_token, PAD_TOKEN)

    def test_text_only_and_mixed_batches_are_rejected(self):
        collator = FlowMatchingDataCollator()
        with self.assertRaisesRegex(ValueError, "text-QA/OPT mixing support was removed"):
            collator([{"input_ids": [1, 2], "labels": [1, 2]}])
        with self.assertRaisesRegex(ValueError, "Rows without latents: \\[1\\]"):
            collator(
                [
                    {"latents": torch.zeros(2, 3), "conditioning_ids": [1]},
                    {"input_ids": [1, 2]},
                ]
            )

    def test_empty_batch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one feature"):
            FlowMatchingDataCollator()([])

    def test_speaker_references_are_patch_aligned_and_masked(self):
        collator = FlowMatchingDataCollator(speaker_patch_size=4)
        batch = collator._collate_tts(
            [
                {
                    "latents": torch.zeros(2, 3),
                    "conditioning_ids": [1],
                    "speaker_latents": torch.ones(2, 4),
                },
                {
                    "latents": torch.zeros(2, 3),
                    "conditioning_ids": [1],
                    "speaker_latents": torch.ones(2, 9),
                },
            ]
        )

        self.assertEqual(batch["speaker_latents"].shape, (2, 2, 12))
        torch.testing.assert_close(
            batch["speaker_mask"],
            torch.tensor([[True, False, False], [True, True, True]]),
        )
        self.assertTrue((batch["speaker_latents"][0, :, 4:] == 0).all())

    def test_empty_speaker_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            FlowMatchingDataCollator()._collate_tts(
                [
                    {
                        "latents": torch.zeros(2, 3),
                        "conditioning_ids": [1],
                        "speaker_latents": torch.empty(2, 0),
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
