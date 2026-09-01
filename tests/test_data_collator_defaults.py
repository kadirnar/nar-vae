"""Shared-token defaults for dataset collation."""

import unittest

import torch

from nar_vae.dataset.data_collator import (
    FlowMatchingDataCollator,
    SimpleTTSCollator,
    create_data_collator,
)
from nar_vae.languages import language_id
from nar_vae.tokenization import PAD_TOKEN, encode_tts_conditioning


class DataCollatorDefaultsTest(unittest.TestCase):
    def test_every_collator_uses_shared_pad_token(self):
        self.assertEqual(FlowMatchingDataCollator().pad_token, PAD_TOKEN)
        self.assertEqual(SimpleTTSCollator().pad_token, PAD_TOKEN)
        self.assertEqual(create_data_collator().pad_token, PAD_TOKEN)

    def test_text_only_and_mixed_batches_are_rejected(self):
        collator = FlowMatchingDataCollator()
        with self.assertRaisesRegex(ValueError, "require acoustic latent rows"):
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

    def test_v2_text_language_and_alignment_fields_are_padded_independently(self):
        english = encode_tts_conditioning("hello", language="en")
        switched = encode_tts_conditioning(
            None,
            language_spans=[
                {"text": "hello", "language": "en"},
                {"text": "dünya", "language": "tr"},
            ],
        )
        batch = FlowMatchingDataCollator()(
            [
                {
                    "latents": torch.zeros(2, 8),
                    "conditioning_ids": english.conditioning_ids,
                    "token_language_ids": english.token_language_ids,
                    "alignment_mask": english.alignment_mask,
                    "language": "en",
                },
                {
                    "latents": torch.zeros(2, 8),
                    "conditioning_ids": switched.conditioning_ids,
                    "token_language_ids": switched.token_language_ids,
                    "alignment_mask": switched.alignment_mask,
                    "language": "en",
                },
            ]
        )

        self.assertEqual(batch["token_language_ids"].shape, batch["conditioning_ids"].shape)
        self.assertEqual(batch["alignment_mask"].shape, batch["conditioning_ids"].shape)
        self.assertTrue((batch["token_language_ids"][~batch["conditioning_mask"]] == 0).all())
        self.assertFalse(batch["alignment_mask"][0, 0])
        self.assertFalse(batch["alignment_mask"][0, len(english.conditioning_ids) - 1])
        self.assertIn(language_id("tr"), batch["token_language_ids"][1].tolist())

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
