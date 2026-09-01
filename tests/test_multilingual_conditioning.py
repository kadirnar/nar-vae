"""Learned target-language conditioning and checkpoint metadata tests."""

import tempfile
import unittest
from pathlib import Path

import torch

from vyvotts.checkpoint import (
    FlowCheckpoint,
    LegacyCrossLingualCheckpointError,
    LegacyLanguageCheckpointError,
    load_pretrained_checkpoint,
)
from vyvotts.languages import LANGUAGE_COUNT, language_id
from vyvotts.models.dit import TextEncoder
from vyvotts.models.flow_matching import FlowMatchingEchoDiT
from vyvotts.solvers.ode_solver import ODESolver


class RecordingLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.language_ids = []

    def forward(
        self,
        *,
        latents,
        conditioning_ids,
        timesteps,
        attention_mask=None,
        language_ids=None,
    ):
        del conditioning_ids, timesteps, attention_mask
        self.language_ids.append(language_ids)
        return torch.zeros_like(latents)


class MultilingualConditioningTest(unittest.TestCase):
    @staticmethod
    def _tiny_multilingual_model(
        supported_languages,
        *,
        use_speaker_conditioning=False,
        supported_reference_languages=None,
    ):
        return FlowMatchingEchoDiT(
            latent_size=2,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=8,
            text_vocab_size=16,
            text_model_size=4,
            text_num_layers=0,
            text_num_heads=1,
            text_intermediate_size=8,
            speaker_patch_size=1,
            speaker_model_size=4,
            speaker_num_layers=0,
            speaker_num_heads=1,
            speaker_intermediate_size=8,
            timestep_embed_size=4,
            adaln_rank=2,
            use_language_conditioning=True,
            supported_languages=supported_languages,
            use_speaker_conditioning=use_speaker_conditioning,
            supported_reference_languages=supported_reference_languages,
        )

    def test_language_embedding_changes_text_state_without_changing_tokens(self):
        encoder = TextEncoder(
            vocab_size=8,
            model_size=4,
            num_layers=0,
            num_heads=1,
            intermediate_size=8,
            norm_eps=1e-6,
            num_languages=LANGUAGE_COUNT,
        )
        with torch.no_grad():
            encoder.text_embedding.weight.zero_()
            encoder.language_embedding.weight.zero_()
            encoder.language_embedding.weight[language_id("es")].fill_(1.0)
            encoder.language_embedding.weight[language_id("ja")].fill_(2.0)

        tokens = torch.tensor([[1, 2]])
        spanish = encoder(tokens, language_ids=torch.tensor([language_id("es")]))
        japanese = encoder(tokens, language_ids=torch.tensor([language_id("ja")]))

        torch.testing.assert_close(spanish, torch.ones_like(spanish))
        torch.testing.assert_close(japanese, torch.full_like(japanese, 2.0))

    def test_ode_solver_propagates_target_language(self):
        model = RecordingLanguageModel()
        ids = torch.tensor([language_id("tr")])

        ODESolver.sample(
            model=model,
            conditioning_ids=torch.ones(1, 2, dtype=torch.long),
            language_ids=ids,
            num_steps=1,
            latent_shape=(1, 2, 2),
            cfg_scale=1.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(model.language_ids), 1)
        torch.testing.assert_close(model.language_ids[0], ids)

    def test_cfg_drops_language_with_text_but_not_with_speaker(self):
        class FakeDiT(torch.nn.Module):
            speaker_patch_size = 1

            def __init__(self):
                super().__init__()
                self.language_ids = None
                self.speakers = None

            def encode_text(self, conditioning_ids, text_mask, language_ids):
                del conditioning_ids, text_mask
                self.language_ids = language_ids.clone()
                return language_ids.float()[:, None, None]

            def project_text_kv_cache(self, text_state):
                values = text_state[:, 0]
                return [(values, values)]

            def get_kv_cache_speaker(self, speaker_latent, speaker_mask=None):
                del speaker_mask
                self.speakers = speaker_latent.clone()
                values = speaker_latent.mean(dim=(1, 2), keepdim=True)
                return [(values, values)]

        model = FlowMatchingEchoDiT.__new__(FlowMatchingEchoDiT)
        torch.nn.Module.__init__(model)
        model.latent_size = 2
        model.speaker_patch_size = 1
        model.use_speaker_conditioning = True
        model.use_language_conditioning = True
        model.supported_languages = ("en", "es")
        model.register_buffer("null_speaker_embed", torch.zeros(1, 2, 1))
        model.register_buffer(
            "supported_language_ids_metadata",
            torch.tensor([language_id("en"), language_id("es")]),
        )
        model.dit = FakeDiT()

        model.prepare_fused_cfg_conditioning(
            torch.ones(1, 2, dtype=torch.long),
            speaker_latent=torch.ones(1, 2, 1),
            language_ids=torch.tensor([language_id("es")]),
            cfg_mode="independent",
        )

        torch.testing.assert_close(
            model.dit.language_ids,
            torch.tensor([language_id("es"), 0, language_id("es")]),
        )
        torch.testing.assert_close(
            model.dit.speakers.mean(dim=(1, 2)),
            torch.tensor([1.0, 1.0, 0.0]),
        )

    def test_checkpoint_records_exact_supported_languages(self):
        model = self._tiny_multilingual_model(["en", "es", "ja"])
        checkpoint = FlowCheckpoint(
            path=Path("multilingual.bin"),
            state_dict=model.state_dict(),
        )

        capability = checkpoint.language_capability()

        self.assertTrue(capability.enabled)
        self.assertEqual(capability.supported_languages, ("en", "es", "ja"))

    def test_checkpoint_records_reference_languages_separately(self):
        model = self._tiny_multilingual_model(
            ["en", "es"],
            use_speaker_conditioning=True,
            supported_reference_languages=["en", "ja"],
        )
        checkpoint = FlowCheckpoint(
            path=Path("cross-lingual.bin"),
            state_dict=model.state_dict(),
        )

        capability = checkpoint.reference_language_capability()

        self.assertTrue(capability.enabled)
        self.assertEqual(capability.supported_languages, ("en", "ja"))

    def test_partial_reference_language_metadata_is_rejected(self):
        model = self._tiny_multilingual_model(
            ["en", "es"],
            use_speaker_conditioning=True,
            supported_reference_languages=["en"],
        )
        state_dict = model.state_dict()
        del state_dict["reference_language_registry_version"]
        checkpoint = FlowCheckpoint(path=Path("partial.bin"), state_dict=state_dict)

        with self.assertRaisesRegex(LegacyCrossLingualCheckpointError, "incomplete"):
            checkpoint.reference_language_capability()

    def test_training_loader_rejects_same_shape_different_language_sets(self):
        source = self._tiny_multilingual_model(["en", "es"])
        target = self._tiny_multilingual_model(["en", "fr"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "multilingual.bin"
            torch.save(source.state_dict(), checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "different supported languages"):
                load_pretrained_checkpoint(target, checkpoint_path)

    def test_training_loader_rejects_same_shape_different_reference_sets(self):
        source = self._tiny_multilingual_model(
            ["en", "es"],
            use_speaker_conditioning=True,
            supported_reference_languages=["en"],
        )
        target = self._tiny_multilingual_model(
            ["en", "es"],
            use_speaker_conditioning=True,
            supported_reference_languages=["es"],
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cross-lingual.bin"
            torch.save(source.state_dict(), checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, "different reference languages"):
                load_pretrained_checkpoint(target, checkpoint_path)

    def test_unversioned_language_embedding_is_rejected(self):
        checkpoint = FlowCheckpoint(
            path=Path("legacy.bin"),
            state_dict={
                "dit.text_encoder.language_embedding.weight": torch.zeros(19, 4),
            },
        )

        with self.assertRaisesRegex(LegacyLanguageCheckpointError, "Missing"):
            checkpoint.language_capability()


if __name__ == "__main__":
    unittest.main()
