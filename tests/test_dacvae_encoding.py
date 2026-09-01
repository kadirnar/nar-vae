"""Focused tests for external, reproducible DACVAE posterior sampling."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from nar_vae.dacvae_encoding import (
    DACVAE_POSTERIOR_SAMPLING_POLICY,
    TORCH_UINT64_SEED_MAX,
    DACVAEEncodingError,
    canonical_mono_float32_pcm,
    derive_dacvae_posterior_seed,
    encode_dacvae_posterior_legacy_global_rng,
    encode_dacvae_posterior_seeded,
    validate_torch_seed,
)

_CODEC_SHA256 = "a" * 64


class _Projection:
    def __call__(self, encoded: torch.Tensor) -> torch.Tensor:
        mean = encoded * 0.25 + 0.1
        scale = encoded * -0.125 - 0.2
        return torch.cat((mean, scale), dim=1)


class _Codec:
    def __init__(self) -> None:
        self.quantizer = SimpleNamespace(in_proj=_Projection())

    @staticmethod
    def _pad(audio: torch.Tensor) -> torch.Tensor:
        return F.pad(audio, (0, 1), mode="reflect")

    @staticmethod
    def encoder(audio: torch.Tensor) -> torch.Tensor:
        return torch.cat((audio, audio * 2.0), dim=1)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self._pad(audio))
        mean, scale = self.quantizer.in_proj(encoded).chunk(2, dim=1)
        stdev = F.softplus(scale) + 1e-4
        return torch.randn_like(mean) * stdev + mean


class SeedContractTest(unittest.TestCase):
    def test_seed_accepts_full_unsigned_torch_range_and_rejects_ambiguous_values(self):
        self.assertEqual(validate_torch_seed(0), 0)
        self.assertEqual(validate_torch_seed(np.uint64(TORCH_UINT64_SEED_MAX)), 2**64 - 1)
        for invalid in (-1, 2**64, True, 1.0, "1"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(DACVAEEncodingError, "unsigned 64-bit"):
                    validate_torch_seed(invalid)

    def test_content_seed_is_canonical_and_bound_to_codec_and_pcm(self):
        mono = np.array([0.0, -0.0, 0.25, -0.5], dtype=np.float64)
        stereo = np.stack((mono.astype(np.float32), mono.astype(np.float32)))
        first = derive_dacvae_posterior_seed(mono, codec_sha256=_CODEC_SHA256)
        second = derive_dacvae_posterior_seed(torch.from_numpy(stereo), codec_sha256=_CODEC_SHA256)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, TORCH_UINT64_SEED_MAX)
        self.assertNotEqual(
            first,
            derive_dacvae_posterior_seed(mono, codec_sha256="b" * 64),
        )
        changed = mono.copy()
        changed[-1] += 1e-3
        self.assertNotEqual(
            first,
            derive_dacvae_posterior_seed(changed, codec_sha256=_CODEC_SHA256),
        )
        self.assertEqual(DACVAE_POSTERIOR_SAMPLING_POLICY, "posterior_sample_seeded_v1")

    def test_canonical_pcm_rejects_invalid_shape_empty_and_nonfinite(self):
        audio = np.asarray([-0.0, 0.5], dtype="<f4")
        original_bits = audio.view(np.uint32).copy()
        expected = np.asarray([0.0, 0.5], dtype="<f4").tobytes()
        self.assertEqual(canonical_mono_float32_pcm(audio), expected)
        np.testing.assert_array_equal(audio.view(np.uint32), original_bits)

        tensor = torch.tensor([-0.0, 0.5], dtype=torch.float32)
        tensor_bits = tensor.view(torch.int32).clone()
        self.assertEqual(canonical_mono_float32_pcm(tensor), expected)
        torch.testing.assert_close(tensor.view(torch.int32), tensor_bits)
        for invalid in ([], np.zeros((1, 1, 2)), [0.0, float("nan")]):
            with self.subTest(shape=np.asarray(invalid).shape):
                with self.assertRaises(DACVAEEncodingError):
                    canonical_mono_float32_pcm(invalid)
        with self.assertRaisesRegex(DACVAEEncodingError, "lowercase SHA-256"):
            derive_dacvae_posterior_seed([0.0], codec_sha256="A" * 64)


class SeededPosteriorEncodingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.codec = _Codec()
        self.audio = torch.tensor([[[0.1, -0.2, 0.3, -0.4]]], dtype=torch.float32)

    def test_matches_the_unchanged_public_formula_for_the_same_cpu_seed(self):
        prior_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(1729)
            expected = self.codec.encode(self.audio)
        finally:
            torch.random.set_rng_state(prior_state)

        actual = encode_dacvae_posterior_seeded(self.codec, self.audio, seed=1729)
        self.assertTrue(torch.equal(actual, expected))

    def test_legacy_helper_matches_origin_global_rng_and_consumes_the_same_state(self):
        prior_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(1729)
            expected = self.codec.encode(self.audio)
            expected_state = torch.random.get_rng_state().clone()
            torch.manual_seed(1729)
            actual = encode_dacvae_posterior_legacy_global_rng(self.codec, self.audio)
            actual_state = torch.random.get_rng_state().clone()
        finally:
            torch.random.set_rng_state(prior_state)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_posterior_formula_oracle_and_seed_sensitivity(self):
        encoded = self.codec.encoder(self.codec._pad(self.audio))
        mean, scale = self.codec.quantizer.in_proj(encoded).chunk(2, dim=1)
        generator = torch.Generator(device="cpu").manual_seed(7)
        noise = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
        expected = noise * (F.softplus(scale) + 1e-4) + mean

        actual = encode_dacvae_posterior_seeded(self.codec, self.audio, seed=7)
        other = encode_dacvae_posterior_seeded(self.codec, self.audio, seed=8)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, other))

    def test_repeated_call_is_exact_and_global_rng_state_is_unchanged(self):
        torch.manual_seed(991)
        before = torch.random.get_rng_state().clone()
        first = encode_dacvae_posterior_seeded(self.codec, self.audio, seed=123)
        middle = torch.random.get_rng_state().clone()
        second = encode_dacvae_posterior_seeded(self.codec, self.audio, seed=123)
        after = torch.random.get_rng_state()

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(before, middle))
        self.assertTrue(torch.equal(before, after))

    def test_concurrent_calls_match_serial_results_without_shared_rng(self):
        seeds = [0, 1, 2, 3, 2**63, TORCH_UINT64_SEED_MAX] * 4
        serial = [
            encode_dacvae_posterior_seeded(self.codec, self.audio, seed=seed) for seed in seeds
        ]
        before = torch.random.get_rng_state().clone()
        with ThreadPoolExecutor(max_workers=6) as executor:
            concurrent = list(
                executor.map(
                    lambda seed: encode_dacvae_posterior_seeded(self.codec, self.audio, seed=seed),
                    seeds,
                )
            )
        after = torch.random.get_rng_state()

        for expected, actual in zip(serial, concurrent):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(before, after))

    def test_input_validation_fails_closed(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            encode_dacvae_posterior_seeded(self.codec, [[[0.0]]], seed=1)
        with self.assertRaisesRegex(DACVAEEncodingError, r"\[1, 1, T\]"):
            encode_dacvae_posterior_seeded(self.codec, torch.zeros(1, 2, 4), seed=1)
        with self.assertRaisesRegex(DACVAEEncodingError, r"\[1, 1, T\]"):
            encode_dacvae_posterior_seeded(self.codec, torch.zeros(2, 1, 4), seed=1)
        with self.assertRaisesRegex(DACVAEEncodingError, "floating-point"):
            encode_dacvae_posterior_seeded(
                self.codec, torch.zeros(1, 1, 4, dtype=torch.int64), seed=1
            )
        with self.assertRaisesRegex(DACVAEEncodingError, "finite"):
            encode_dacvae_posterior_seeded(
                self.codec, torch.tensor([[[float("inf"), 0.0]]]), seed=1
            )


class ProductionCallSiteTest(unittest.TestCase):
    def test_production_dacvae_encodes_use_only_versioned_boundary_helpers(self):
        package = Path(__file__).resolve().parents[1] / "nar_vae"
        seeded_calls = Counter()
        legacy_calls = Counter()
        raw_calls = []
        for path in package.rglob("*.py"):
            relative = path.relative_to(package).as_posix()
            if relative.startswith("dacvae/") or relative == "dacvae_encoding.py":
                continue
            source = path.read_text(encoding="utf-8")
            if ".dacvae.encode(" in source:
                raw_calls.append(relative)
            tree = ast.parse(source, filename=str(path))
            seeded_calls[relative] = sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "encode_dacvae_posterior_seeded"
                for node in ast.walk(tree)
            )
            legacy_calls[relative] = sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "encode_dacvae_posterior_legacy_global_rng"
                for node in ast.walk(tree)
            )

        self.assertEqual(raw_calls, [])
        self.assertEqual(
            {name: count for name, count in seeded_calls.items() if count},
            {
                "dataset/finetune_prepare.py": 2,
                "dataset/prepare.py": 2,
                "dataset/prepare_dataset.py": 3,
                "inference.py": 1,
            },
        )
        self.assertEqual(
            {name: count for name, count in legacy_calls.items() if count},
            {"inference.py": 1},
        )


if __name__ == "__main__":
    unittest.main()
