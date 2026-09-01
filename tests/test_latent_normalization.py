"""Synthetic tests for the versioned latent-normalization contract."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError

import torch

from nar_vae.latent_normalization import (
    LATENT_NORMALIZATION_SCHEMA_VERSION,
    LatentNormalizationContract,
    LatentNormalizationError,
    LatentStatisticsAccumulator,
)

_CODEC_SHA256 = "a" * 64


def _contract(**overrides) -> LatentNormalizationContract:
    values = {
        "mode": "per_channel_v1",
        "codec_source": "facebook/dacvae-watermarked",
        "codec_revision": "b" * 40,
        "codec_sha256": _CODEC_SHA256,
        "sample_rate": 44100,
        "hop_length": 512,
        "latent_dim": 2,
        "frame_count": 8,
        "dataset_fingerprint": "synthetic-train-v1",
        "mean": (1.0, -2.0),
        "std": (2.0, 4.0),
    }
    values.update(overrides)
    return LatentNormalizationContract(**values)


class LatentStatisticsAccumulatorTest(unittest.TestCase):
    def test_streaming_valid_prefix_and_parallel_merge_match_population_statistics(self):
        first = torch.tensor(
            [[1.0e8 + 1.0, 1.0e8 + 2.0, 1.0e8 + 4.0, -999.0], [1.0, 3.0, 8.0, -999.0]],
            dtype=torch.float64,
        )
        second = torch.tensor([[1.0e8 + 8.0, 1.0e8 + 16.0], [13.0, 21.0]], dtype=torch.float64)
        expected = torch.cat((first[:, :3], second), dim=1)

        left = LatentStatisticsAccumulator(2).update(first, valid_frames=3)
        right = LatentStatisticsAccumulator(2).update(second)
        left.merge(right)
        contract = left.finalize(
            mode="per_channel_v1",
            codec_source="facebook/dacvae-watermarked",
            codec_revision="b" * 40,
            codec_sha256=_CODEC_SHA256,
            sample_rate=44100,
            hop_length=512,
            dataset_fingerprint="synthetic-train-v1",
        )

        self.assertEqual(contract.frame_count, expected.shape[1])
        torch.testing.assert_close(
            torch.tensor(contract.mean, dtype=torch.float64),
            expected.mean(dim=1),
            rtol=2e-16,
            atol=1e-12,
        )
        torch.testing.assert_close(
            torch.tensor(contract.std, dtype=torch.float64),
            expected.std(dim=1, correction=0),
            rtol=1e-9,
            atol=1e-12,
        )

    def test_updates_validate_shape_valid_count_and_finite_values(self):
        accumulator = LatentStatisticsAccumulator(2)
        self.assertIs(accumulator.update(torch.empty(2, 3), valid_frames=0), accumulator)
        self.assertEqual(accumulator.frame_count, 0)

        with self.assertRaisesRegex(LatentNormalizationError, r"\[C, T\]"):
            accumulator.update(torch.zeros(1, 2, 3))
        with self.assertRaisesRegex(LatentNormalizationError, "channel dimension"):
            accumulator.update(torch.zeros(3, 2))
        with self.assertRaisesRegex(LatentNormalizationError, "between 0 and 2"):
            accumulator.update(torch.zeros(2, 2), valid_frames=3)
        with self.assertRaisesRegex(LatentNormalizationError, "finite"):
            accumulator.update(torch.tensor([[0.0, float("nan")], [1.0, 2.0]]))

    def test_constant_channel_is_rejected_instead_of_silently_clamped(self):
        accumulator = LatentStatisticsAccumulator(2).update(torch.ones(2, 4))
        with self.assertRaisesRegex(LatentNormalizationError, "strictly positive"):
            accumulator.finalize(
                mode="per_channel_v1",
                codec_source="local/codec",
                codec_revision=None,
                codec_sha256=_CODEC_SHA256,
                sample_rate=44100,
                hop_length=512,
                dataset_fingerprint="constant-data",
            )


class LatentNormalizationContractTest(unittest.TestCase):
    def test_normalize_denormalize_exactly_round_trip_ct_and_bct(self):
        contract = _contract()
        ct = torch.tensor([[1.0, 3.0, -1.0], [-2.0, 2.0, -6.0]], dtype=torch.float32)
        expected = torch.tensor([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]], dtype=torch.float32)

        normalized = contract.normalize(ct)
        self.assertTrue(torch.equal(normalized, expected))
        self.assertTrue(torch.equal(contract.denormalize(normalized), ct))
        self.assertEqual(normalized.dtype, ct.dtype)
        self.assertEqual(normalized.device, ct.device)

        bct = torch.stack((ct, ct + torch.tensor([[2.0], [4.0]])))
        self.assertTrue(torch.equal(contract.denormalize(contract.normalize(bct)), bct))

    def test_none_is_explicit_serializable_and_an_exact_no_op(self):
        accumulator = LatentStatisticsAccumulator(2).update(torch.tensor([[1.0], [2.0]]))
        contract = accumulator.finalize(
            mode="none",
            codec_source="local/codec",
            codec_revision=None,
            codec_sha256=_CODEC_SHA256,
            sample_rate=44100,
            hop_length=512,
            dataset_fingerprint="disabled-normalization",
        )
        latents = torch.randn(3, 2, 5)

        self.assertIs(contract.normalize(latents), latents)
        self.assertIs(contract.denormalize(latents), latents)
        self.assertEqual(contract.mean, ())
        self.assertEqual(contract.std, ())
        self.assertEqual(LatentNormalizationContract.from_json(contract.to_json()), contract)

    def test_json_and_checksum_are_deterministic_and_tamper_evident(self):
        contract = _contract()
        encoded = contract.to_json()

        self.assertEqual(encoded, contract.to_json())
        self.assertEqual(LatentNormalizationContract.from_json(encoded), contract)
        self.assertEqual(len(contract.checksum), 64)
        self.assertEqual(json.loads(encoded)["schema_version"], LATENT_NORMALIZATION_SCHEMA_VERSION)

        tampered = json.loads(encoded)
        tampered["mean"][0] = 123.0
        with self.assertRaisesRegex(LatentNormalizationError, "checksum does not match"):
            LatentNormalizationContract.from_json(json.dumps(tampered))

        duplicate = encoded[:-1] + ',"mode":"none"}'
        with self.assertRaisesRegex(LatentNormalizationError, "Duplicate JSON field"):
            LatentNormalizationContract.from_json(duplicate)
        with self.assertRaisesRegex(LatentNormalizationError, "Non-finite JSON number"):
            LatentNormalizationContract.from_json(encoded.replace('"mean":[1.0', '"mean":[NaN'))

    def test_contract_is_immutable_and_validates_metadata_statistics_and_shapes(self):
        contract = _contract()
        with self.assertRaises(FrozenInstanceError):
            contract.mode = "none"
        with self.assertRaisesRegex(LatentNormalizationError, "lowercase SHA-256"):
            _contract(codec_sha256="not-a-digest")
        with self.assertRaisesRegex(LatentNormalizationError, "lengths must equal"):
            _contract(mean=(0.0,))
        with self.assertRaisesRegex(LatentNormalizationError, "strictly positive"):
            _contract(std=(1.0, 0.0))
        with self.assertRaisesRegex(LatentNormalizationError, "must not contain"):
            _contract(mode="none")
        with self.assertRaisesRegex(LatentNormalizationError, "channel dimension"):
            contract.normalize(torch.zeros(3, 4))
        with self.assertRaisesRegex(LatentNormalizationError, r"\[C, T\].*\[B, C, T\]"):
            contract.normalize(torch.zeros(2))
        with self.assertRaisesRegex(LatentNormalizationError, "floating-point"):
            contract.normalize(torch.zeros(2, 4, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
