import hashlib
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

try:
    from datasets import Dataset, load_from_disk
except ImportError:  # pragma: no cover - optional in lightweight unit environments
    Dataset = None
    load_from_disk = None

from nar_vae.dataset.data_collator import FlowMatchingDataCollator
from nar_vae.frozen_text_provider import (
    FROZEN_TEXT_ALIGNMENT_POLICY,
    FrozenTextProvider,
    FrozenTextProviderSpec,
)
from nar_vae.languages import language_id
from nar_vae.tokenization import TextSpan

_COMMIT = "a" * 40


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeTokenizer:
    is_fast = True
    pad_token_id = 1
    bos_token_id = 0
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self) -> None:
        tokens = (
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "▁",
            "h",
            "ə",
            "l",
            "o",
            "m",
            "e",
            "r",
            "a",
            "b",
            "d",
            "ü",
            "n",
            "y",
            "i",
            "!",
            "Hello",
            "world",
        )
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.calls: list[str] = []

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool,
        return_special_tokens_mask: bool,
        return_offsets_mapping: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.calls.append(text)
        assert not padding
        assert not truncation
        assert return_attention_mask
        assert return_special_tokens_mask
        assert return_offsets_mapping
        assert return_tensors == "pt"
        matches = tuple(re.finditer(r"\S+", text))
        ids = [self.vocab.get(match.group(), self.unk_token_id) for match in matches]
        offsets = [(match.start(), match.end()) for match in matches]
        special = [0] * len(ids)
        if add_special_tokens:
            ids = [self.bos_token_id, *ids, self.eos_token_id]
            offsets = [(0, 0), *offsets, (0, 0)]
            special = [1, *special, 1]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "special_tokens_mask": torch.tensor([special], dtype=torch.long),
            "offset_mapping": torch.tensor([offsets], dtype=torch.long),
        }


class _FakeModel(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int = 4) -> None:
        super().__init__()
        self.sentinel = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            vocab_size=vocab_size,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=hidden_size,
            max_position_embeddings=128,
        )
        self.grad_enabled: bool | None = None
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask, **kwargs):
        self.forward_calls += 1
        self.grad_enabled = torch.is_grad_enabled()
        self.last_attention_mask = attention_mask
        self.last_kwargs = kwargs
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 1, 4)
        return SimpleNamespace(hidden_states=(base, base + 100.0, base + 200.0))


class FrozenTextProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.tokenizer = _FakeTokenizer()
        self.model = _FakeModel(vocab_size=len(self.tokenizer))
        self.config_path = self.root / "config.json"
        self.model_path = self.root / "pytorch_model.bin"
        self.tokenizer_path = self.root / "tokenizer.json"
        self.config_path.write_text(
            json.dumps(
                {
                    "vocab_size": len(self.tokenizer),
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "hidden_size": 4,
                }
            ),
            encoding="utf-8",
        )
        self.model_path.write_bytes(b"mock frozen model")
        self.tokenizer_path.write_bytes(b"mock fast tokenizer")
        self.resolver_calls: list[dict[str, str]] = []
        self.loader_calls: list[str] = []

    def _config(self, **overrides):
        config = {
            "text_conditioning_mode": "frozen_features",
            "conditioning_feature_size": 4,
            "conditioning_feature_dtype": "float16",
            "frozen_text_alignment": FROZEN_TEXT_ALIGNMENT_POLICY,
            "frozen_text_cache_version": 1,
            "frozen_text_config_sha256": _sha256(self.config_path),
            "frozen_text_encoder_id": "example/encoder",
            "frozen_text_encoder_revision": _COMMIT,
            "frozen_text_frontend": "phonemes",
            "frozen_text_hidden_layer": -1,
            "frozen_text_model_filename": self.model_path.name,
            "frozen_text_model_sha256": _sha256(self.model_path),
            "frozen_text_tokenizer_filename": self.tokenizer_path.name,
            "frozen_text_tokenizer_id": "example/tokenizer",
            "frozen_text_tokenizer_revision": _COMMIT,
            "frozen_text_tokenizer_sha256": _sha256(self.tokenizer_path),
            "text_vocab_size": len(self.tokenizer),
            "pad_token": self.tokenizer.pad_token_id,
        }
        config.update(overrides)
        return config

    def _resolver(self, **kwargs):
        self.resolver_calls.append(dict(kwargs))
        return {
            self.config_path.name: self.config_path,
            self.model_path.name: self.model_path,
            self.tokenizer_path.name: self.tokenizer_path,
        }[kwargs["filename"]]

    def _tokenizer_loader(self, spec, artifacts):
        self.loader_calls.append("tokenizer")
        self.assertEqual(artifacts.tokenizer_path, self.tokenizer_path)
        return self.tokenizer

    def _model_loader(self, spec, artifacts):
        self.loader_calls.append("model")
        self.assertEqual(artifacts.model_path, self.model_path)
        return self.model

    def _provider(self, **config_overrides) -> FrozenTextProvider:
        return FrozenTextProvider.from_config(
            self._config(**config_overrides),
            artifact_resolver=self._resolver,
            tokenizer_loader=self._tokenizer_loader,
            model_loader=self._model_loader,
        )

    def test_pins_three_artifacts_freezes_model_and_binds_contract(self):
        provider = self._provider()

        self.assertEqual(
            [(call["filename"], call["revision"]) for call in self.resolver_calls],
            [("config.json", _COMMIT), ("pytorch_model.bin", _COMMIT), ("tokenizer.json", _COMMIT)],
        )
        self.assertEqual(self.loader_calls, ["tokenizer", "model"])
        self.assertFalse(provider.model.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in provider.model.parameters())
        )
        self.assertEqual(provider.vocab_size, len(self.tokenizer))
        self.assertEqual(provider.pad_token_id, 1)
        self.assertRegex(provider.spec.contract_sha256, r"^[0-9a-f]{64}$")
        same = FrozenTextProviderSpec.from_config(self._config())
        self.assertEqual(provider.spec.contract_sha256, same.contract_sha256)

    def test_phonemes_use_exact_provider_axis_masks_and_selected_layer(self):
        provider = self._provider()
        result = provider.encode(
            "ignored source text",
            phonemes=["h", "ə", "|", "l", "o", "!"],
            language="en",
        )

        self.assertEqual(result.rendered_text, "h ə ▁ l o !")
        self.assertEqual(result.conditioning_ids.tolist(), [0, 5, 6, 4, 7, 8, 19, 2])
        self.assertEqual(result.conditioning_features.shape, (8, 4))
        self.assertEqual(result.conditioning_features.dtype, torch.float16)
        self.assertEqual(result.conditioning_mask.tolist(), [True] * 8)
        self.assertEqual(
            result.alignment_mask.tolist(),
            [False, True, True, False, True, True, False, False],
        )
        english = language_id("en")
        self.assertEqual(
            result.token_language_ids.tolist(),
            [0, english, english, 0, english, english, 0, 0],
        )
        torch.testing.assert_close(
            result.conditioning_features[:, 0].float(),
            result.conditioning_ids.float() + 200.0,
        )
        self.assertFalse(self.model.grad_enabled)
        self.assertEqual(self.model.forward_calls, 1)
        cache_row = result.to_cache_row()
        self.assertEqual(cache_row["frozen_text_contract_sha256"], provider.spec.contract_sha256)
        self.assertEqual(cache_row["frozen_text_cache_version"], 1)
        self.assertEqual(cache_row["conditioning_features"].dtype, torch.float16)
        model_inputs = result.as_model_inputs()
        self.assertEqual(model_inputs["conditioning_features"].shape, (1, 8, 4))
        self.assertEqual(model_inputs["attention_mask"].shape, (1, 8))

    def test_word_boundary_aliases_canonicalize_to_exactly_one_native_token(self):
        provider = self._provider()
        result = provider.encode(
            None,
            phonemes=["h", "i", "_", "|", "▁", "m"],
            language="en",
        )

        self.assertEqual(result.rendered_text, "h i ▁ m")
        self.assertEqual(result.conditioning_ids.tolist().count(self.tokenizer.vocab["▁"]), 1)
        self.assertEqual(result.alignment_mask.tolist(), [False, True, True, False, True, False])

    def test_phoneme_frontend_requires_supplied_native_phones_and_fails_closed(self):
        provider = self._provider()
        with self.assertRaisesRegex(ValueError, "caller-supplied phonemes"):
            provider.encode("hello", language="en")
        with self.assertRaisesRegex(ValueError, "style/control tags"):
            provider.encode(None, phonemes=["h", "<laugh>"], language="en")
        with self.assertRaisesRegex(ValueError, "unknown token"):
            provider.encode(None, phonemes=["h", "🚀"], language="en")
        self.assertEqual(self.model.forward_calls, 0)

    def test_code_switch_spans_keep_atomic_language_and_alignment_arrays(self):
        provider = self._provider()
        result = provider.encode(
            None,
            language="en",
            language_spans=(
                TextSpan(text="", language="en", phonemes=("h", "i")),
                TextSpan(text="", language="tr", phonemes=("h", "i")),
            ),
        )

        self.assertEqual(result.rendered_text, "h i ▁ h i")
        english = language_id("en")
        turkish = language_id("tr")
        self.assertEqual(
            result.token_language_ids.tolist(),
            [
                0,
                english,
                english,
                0,
                turkish,
                turkish,
                0,
            ],
        )
        self.assertFalse(result.alignment_mask[0])
        self.assertFalse(result.alignment_mask[3])
        self.assertFalse(result.alignment_mask[-1])

    def test_raw_text_mode_is_explicit_and_rejects_invented_controls(self):
        provider = self._provider(frozen_text_frontend="raw_text")
        result = provider.encode("Hello world", language="en")
        self.assertEqual(result.rendered_text, "Hello world")
        self.assertEqual(result.conditioning_ids.tolist(), [0, 20, 21, 2])
        self.assertEqual(result.alignment_mask.tolist(), [False, True, True, False])
        with self.assertRaisesRegex(ValueError, "does not invent"):
            provider.encode("Hello <laugh> world", language="en")

    def test_cache_dtype_supports_all_declared_serializations(self):
        expected = {
            "float16": torch.float16,
            "float32": torch.float32,
        }
        for configured, dtype in expected.items():
            with self.subTest(configured=configured):
                result = self._provider(conditioning_feature_dtype=configured).encode(
                    None, phonemes=["h"], language="en"
                )
                self.assertEqual(result.conditioning_features.dtype, dtype)
        with self.assertRaisesRegex(ValueError, "does not losslessly preserve bfloat16"):
            self._provider(conditioning_feature_dtype="bfloat16")
        valid = self._provider().encode(None, phonemes=["h"], language="en")
        with self.assertRaisesRegex(TypeError, "torch.float16 or torch.float32"):
            replace(valid, conditioning_features=valid.conditioning_features.bfloat16())

    @unittest.skipUnless(Dataset is not None, "datasets is not installed")
    def test_fp16_cache_row_survives_arrow_save_load_with_exact_identity(self):
        provider = self._provider(conditioning_feature_dtype="float16")
        cache_row = provider.encode(None, phonemes=["h", "i"], language="en").to_cache_row()
        cache_row["latents"] = torch.zeros(4, 3)
        cache_row["language"] = "en"

        dataset = Dataset.from_list([cache_row])
        with tempfile.TemporaryDirectory() as directory:
            dataset.save_to_disk(directory)
            loaded = load_from_disk(directory)
            feature = loaded.features["conditioning_features"].feature.feature
            self.assertEqual(feature.dtype, "float16")
            self.assertEqual(loaded[0]["conditioning_feature_dtype"], "float16")
            self.assertEqual(
                loaded[0]["frozen_text_contract_sha256"],
                provider.spec.contract_sha256,
            )
            self.assertEqual(loaded[0]["frozen_text_cache_version"], 1)
            torch.testing.assert_close(
                torch.tensor(loaded[0]["conditioning_features"], dtype=torch.float16),
                cache_row["conditioning_features"],
            )
            batch = FlowMatchingDataCollator(pad_token=provider.pad_token_id)([loaded[0]])
            self.assertEqual(batch["conditioning_features"].dtype, torch.float16)

    def test_hash_mismatch_fails_before_loading_executable_artifacts(self):
        config = self._config(frozen_text_model_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            FrozenTextProvider.from_config(
                config,
                artifact_resolver=self._resolver,
                tokenizer_loader=self._tokenizer_loader,
                model_loader=self._model_loader,
            )
        self.assertEqual(self.loader_calls, [])

    def test_manifest_mapping_and_acoustic_vocab_pad_are_validated(self):
        config = self._config()
        spec = FrozenTextProviderSpec.from_config(config)
        section = {
            "mode": spec.text_conditioning_mode,
            "provider_vocab_size": spec.text_vocab_size,
            "provider_pad_token": spec.pad_token,
            "feature_size": spec.conditioning_feature_size,
            "feature_dtype": spec.conditioning_feature_dtype,
            "alignment": spec.frozen_text_alignment,
            "cache_version": spec.frozen_text_cache_version,
            "config_sha256": spec.frozen_text_config_sha256,
            "encoder_id": spec.frozen_text_encoder_id,
            "encoder_revision": spec.frozen_text_encoder_revision,
            "frontend": spec.frozen_text_frontend,
            "hidden_layer": spec.frozen_text_hidden_layer,
            "model_filename": spec.frozen_text_model_filename,
            "model_sha256": spec.frozen_text_model_sha256,
            "tokenizer_filename": spec.frozen_text_tokenizer_filename,
            "tokenizer_id": spec.frozen_text_tokenizer_id,
            "tokenizer_revision": spec.frozen_text_tokenizer_revision,
            "tokenizer_sha256": spec.frozen_text_tokenizer_sha256,
        }
        self.assertEqual(FrozenTextProviderSpec.from_manifest(section), spec)
        self.assertEqual(FrozenTextProviderSpec.from_manifest({"text_conditioning": section}), spec)
        provider = self._provider()
        with self.assertRaisesRegex(ValueError, "text_vocab_size"):
            provider.validate_acoustic_contract({"text_vocab_size": len(self.tokenizer) + 1})
        with self.assertRaisesRegex(ValueError, "pad_token"):
            provider.validate_acoustic_contract({"pad_token": 0})

    def test_invalid_hidden_layer_and_verified_hf_metadata_are_rejected(self):
        provider = self._provider(frozen_text_hidden_layer=9)
        with self.assertRaisesRegex(ValueError, "hidden_layer=9"):
            provider.encode(None, phonemes=["h"], language="en")

        self.model.config.pad_token_id = 9
        with self.assertRaisesRegex(ValueError, "AutoModel pad_token_id"):
            self._provider()


if __name__ == "__main__":
    unittest.main()
