"""One-latent utterance store and dynamic voice-reference tests."""

import unittest

import numpy as np

from nar_vae.dataset.representation import (
    PREPARED_ROW_VERSION,
    PREPARED_ROW_VERSION_COLUMN,
    REPRESENTATION_CONTRACT_COLUMN,
)
from nar_vae.dataset.utterance_store import (
    AUDIO_SHA256_COLUMN,
    CONDITIONING_NUM_TOKENS_COLUMN,
    SPEAKER_NUM_FRAMES_COLUMN,
    DynamicReferenceDataset,
    validate_utterance_store,
)
from nar_vae.languages import LanguagePair, language_id
from nar_vae.tokenization import encode_tts_conditioning


class Rows:
    def __init__(self, values):
        self.values = values
        self.column_names = list(values[0])

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.values]
        return self.values[key]


def row(index, language, *, speaker="corpus:speaker-a", frames=30):
    text = encode_tts_conditioning(f"sample{index}", language=language)
    return {
        "latents": np.stack(
            [np.arange(frames, dtype=np.float32) + index * 100],
            axis=0,
        ),
        "latent_num_frames": frames,
        "conditioning_ids": text.conditioning_ids,
        "token_language_ids": text.token_language_ids,
        "alignment_mask": text.alignment_mask,
        CONDITIONING_NUM_TOKENS_COLUMN: len(text.conditioning_ids),
        SPEAKER_NUM_FRAMES_COLUMN: frames,
        "language": language,
        "speaker_id": speaker,
        "utterance_id": f"corpus:utt-{index}",
        AUDIO_SHA256_COLUMN: f"{index + 1:064x}",
        PREPARED_ROW_VERSION_COLUMN: PREPARED_ROW_VERSION,
        REPRESENTATION_CONTRACT_COLUMN: {
            "sample_rate": 10,
            "hop_length": 1,
        },
    }


class DynamicReferenceDatasetTest(unittest.TestCase):
    def test_cross_language_pair_selection_is_same_speaker_and_not_target(self):
        dataset = Rows([row(0, "en"), row(1, "en"), row(2, "tr"), row(3, "tr")])
        dynamic = DynamicReferenceDataset(
            dataset,
            supported_language_pairs=(("en", "tr"), ("tr", "en")),
            seed=19,
            min_reference_seconds=1.0,
            short_reference_max_seconds=1.5,
            max_reference_seconds=2.0,
            speaker_patch_size=2,
        )

        english_target = dynamic[0]
        turkish_target = dynamic[2]

        self.assertEqual(
            dynamic.reference_pair_coverage(),
            (LanguagePair("en", "tr"), LanguagePair("tr", "en")),
        )
        self.assertEqual(english_target["speaker_language"], "tr")
        self.assertEqual(turkish_target["speaker_language"], "en")
        self.assertNotEqual(english_target["reference_utterance_id"], "corpus:utt-0")
        self.assertNotEqual(turkish_target["reference_utterance_id"], "corpus:utt-2")
        self.assertNotIn("reference_audio_sha256", dataset[0])

    def test_declared_pair_coverage_fails_before_random_epoch_sampling(self):
        dataset = Rows([row(0, "en"), row(1, "en"), row(2, "tr"), row(3, "tr")])

        with self.assertRaisesRegex(ValueError, "Declared target/reference language pairs"):
            DynamicReferenceDataset(
                dataset,
                supported_language_pairs=(("en", "tr"), ("tr", "ja")),
                min_reference_seconds=1.0,
                short_reference_max_seconds=1.5,
                max_reference_seconds=2.0,
            )

    def test_reference_and_crop_are_reproducible_within_an_epoch(self):
        dataset = Rows([row(0, "en"), row(1, "en"), row(2, "en")])
        dynamic = DynamicReferenceDataset(
            dataset,
            seed=7,
            min_reference_seconds=1.0,
            short_reference_max_seconds=1.5,
            max_reference_seconds=2.0,
            speaker_patch_size=2,
        )

        first = dynamic[0]
        second = dynamic[0]
        self.assertEqual(first["reference_utterance_id"], second["reference_utterance_id"])
        np.testing.assert_array_equal(first["speaker_latents"], second["speaker_latents"])

        self.assertEqual(
            dynamic.get_length_metadata(SPEAKER_NUM_FRAMES_COLUMN)[0],
            first["speaker_latents"].shape[1],
        )
        dynamic.set_epoch(3)
        self.assertEqual(dynamic.state_dict(), {"version": 1, "seed": 7, "epoch": 3})

    def test_duplicate_audio_and_missing_peer_fail_closed(self):
        duplicated = [row(0, "en"), row(1, "en")]
        duplicated[1][AUDIO_SHA256_COLUMN] = duplicated[0][AUDIO_SHA256_COLUMN]
        with self.assertRaisesRegex(ValueError, "Duplicate audio"):
            validate_utterance_store(Rows(duplicated))

        with self.assertRaisesRegex(ValueError, "no same-speaker"):
            DynamicReferenceDataset(Rows([row(0, "en")]))

    def test_language_ids_are_parallel_frontend_metadata_not_speaker_ids(self):
        prepared = row(0, "tr")
        aligned_languages = {
            token_language
            for token_language, aligned in zip(
                prepared["token_language_ids"],
                prepared["alignment_mask"],
            )
            if aligned
        }
        self.assertEqual(aligned_languages, {language_id("tr")})
        self.assertNotIn(prepared["speaker_id"], prepared["conditioning_ids"])


if __name__ == "__main__":
    unittest.main()
