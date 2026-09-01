"""CPU regressions for model-side training efficiency contracts."""

import unittest
from unittest.mock import patch

import torch

import nar_vae.models.dit as dit_module
from nar_vae.models.flow_matching import FlowMatchingEchoDiT


def tiny_model(
    *,
    duration: bool = False,
    speaker: bool = False,
    duration_speaker: bool = False,
    cfg_dropout: float = 0.1,
    cfg_dropout_text: float | None = 0.0,
    cfg_dropout_speaker: float | None = 0.0,
) -> FlowMatchingEchoDiT:
    return FlowMatchingEchoDiT(
        latent_size=4,
        model_size=8,
        num_layers=2,
        num_heads=2,
        intermediate_size=16,
        text_vocab_size=20,
        text_model_size=8,
        text_num_layers=1,
        text_num_heads=2,
        text_intermediate_size=16,
        speaker_patch_size=2,
        speaker_model_size=8,
        speaker_num_layers=1,
        speaker_num_heads=2,
        speaker_intermediate_size=16,
        timestep_embed_size=8,
        adaln_rank=4,
        cfg_dropout=cfg_dropout,
        cfg_dropout_text=cfg_dropout_text,
        cfg_dropout_speaker=cfg_dropout_speaker,
        use_speaker_conditioning=speaker,
        use_duration_predictor=duration,
        duration_predictor_hidden_size=6,
        duration_predictor_num_layers=1,
        duration_predictor_use_speaker=duration_speaker,
    )


class CFGDropoutConfigurationTest(unittest.TestCase):
    def test_default_rate_supplies_both_branches_and_explicit_values_override(self):
        inherited = tiny_model(
            cfg_dropout=0.35,
            cfg_dropout_text=None,
            cfg_dropout_speaker=None,
        )
        self.assertEqual(inherited.cfg_dropout, 0.35)
        self.assertEqual(inherited.cfg_dropout_text, 0.35)
        self.assertEqual(inherited.cfg_dropout_speaker, 0.35)

        overridden = tiny_model(
            cfg_dropout=0.35,
            cfg_dropout_text=0.0,
            cfg_dropout_speaker=1.0,
        )
        self.assertEqual(overridden.cfg_dropout_text, 0.0)
        self.assertEqual(overridden.cfg_dropout_speaker, 1.0)

    def test_direct_model_construction_rejects_invalid_rates(self):
        for kwargs, name in (
            ({"cfg_dropout": -0.1}, "cfg_dropout"),
            ({"cfg_dropout": float("inf")}, "cfg_dropout"),
            ({"cfg_dropout_text": 1.1}, "cfg_dropout_text"),
            ({"cfg_dropout_speaker": True}, "cfg_dropout_speaker"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, name):
                tiny_model(**kwargs)


class ActivationCheckpointingTest(unittest.TestCase):
    def test_enable_checkpoints_every_active_block_non_reentrantly(self):
        model = tiny_model().train()
        model.gradient_checkpointing_enable({"preserve_rng_state": True})
        real_checkpoint = dit_module.activation_checkpoint
        checkpoint_options = []

        def recording_checkpoint(function, *args, **kwargs):
            checkpoint_options.append(kwargs.copy())
            return real_checkpoint(function, *args, **kwargs)

        with patch.object(dit_module, "activation_checkpoint", side_effect=recording_checkpoint):
            output = model(
                latents=torch.randn(2, 4, 4),
                conditioning_ids=torch.tensor([[1, 2, 3], [2, 3, 4]]),
                timesteps=torch.tensor([0.25, 0.75]),
                use_cfg_dropout=False,
            )
            self.assertIsInstance(output, torch.Tensor)
            output.sum().backward()

        # One text block and two velocity-field blocks. A text-only topology does
        # not instantiate or checkpoint a speaker encoder.
        self.assertEqual(len(checkpoint_options), 3)
        self.assertTrue(all(options["use_reentrant"] is False for options in checkpoint_options))
        self.assertTrue(
            all(options["preserve_rng_state"] is True for options in checkpoint_options)
        )
        self.assertIsNotNone(model.dit.out_proj.weight.grad)

    def test_disable_restores_direct_block_execution(self):
        model = tiny_model().train()
        model.gradient_checkpointing_enable()
        model.gradient_checkpointing_disable()

        with patch.object(dit_module, "activation_checkpoint") as checkpoint_mock:
            model(
                latents=torch.randn(1, 4, 3),
                conditioning_ids=torch.tensor([[1, 2]]),
                timesteps=torch.tensor([0.5]),
                use_cfg_dropout=False,
            )

        checkpoint_mock.assert_not_called()

    def test_reentrant_checkpointing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            tiny_model().gradient_checkpointing_enable({"use_reentrant": True})


class SharedConditioningEncodingTest(unittest.TestCase):
    def test_joint_duration_and_velocity_share_unchanged_encoder_states(self):
        model = tiny_model(duration=True, speaker=True, duration_speaker=True).train()
        speaker_latent = torch.randn(2, 4, 4)

        with (
            patch.object(model.dit, "encode_text", wraps=model.dit.encode_text) as encode_text,
            patch.object(
                model.dit,
                "encode_speaker",
                wraps=model.dit.encode_speaker,
            ) as encode_speaker,
        ):
            output = model(
                latents=torch.randn(2, 4, 4),
                conditioning_ids=torch.tensor([[1, 2, 3], [2, 3, 4]]),
                timesteps=torch.tensor([0.25, 0.75]),
                speaker_latent=speaker_latent,
                use_cfg_dropout=True,
                return_duration_prediction=True,
            )

        self.assertIsInstance(output, tuple)
        velocity, duration = output
        self.assertEqual(encode_text.call_count, 1)
        self.assertEqual(encode_speaker.call_count, 1)
        (velocity.sum() + duration.sum()).backward()
        self.assertIsNotNone(model.duration_predictor.output_projection.weight.grad)

    def test_cfg_dropout_reencodes_only_changed_rows(self):
        model = tiny_model(duration=True, speaker=True, duration_speaker=True).train()
        model.cfg_dropout_text = 0.5
        model.cfg_dropout_speaker = 0.5
        text_draws = torch.tensor([0.1, 0.9, 0.9, 0.9])
        speaker_draws = torch.tensor([0.9, 0.1, 0.9, 0.9])

        with (
            patch(
                "nar_vae.models.flow_matching.torch.rand",
                side_effect=(text_draws, speaker_draws),
            ),
            patch.object(model.dit, "encode_text", wraps=model.dit.encode_text) as encode_text,
            patch.object(
                model.dit,
                "encode_speaker",
                wraps=model.dit.encode_speaker,
            ) as encode_speaker,
        ):
            model(
                latents=torch.randn(4, 4, 4),
                conditioning_ids=torch.tensor([[1, 2], [2, 3], [3, 4], [4, 5]]),
                timesteps=torch.tensor([0.1, 0.3, 0.6, 0.9]),
                speaker_latent=torch.randn(4, 4, 4),
                use_cfg_dropout=True,
                return_duration_prediction=True,
            )

        self.assertEqual([call.args[0].shape[0] for call in encode_text.call_args_list], [4, 1])
        self.assertEqual(
            [call.args[0].shape[0] for call in encode_speaker.call_args_list],
            [4, 1],
        )


class CompactNARTopologyTest(unittest.TestCase):
    def test_text_only_model_omits_ar_prefix_and_speaker_parameters(self):
        model = tiny_model()
        state_keys = set(model.state_dict())
        forbidden_fragments = (
            "latent_encoder",
            "latent_norm",
            "wk_latent",
            "wv_latent",
            "speaker_encoder",
            "speaker_norm",
            "wk_speaker",
            "wv_speaker",
        )
        self.assertFalse(
            [key for key in state_keys if any(fragment in key for fragment in forbidden_fragments)]
        )
        self.assertIsNone(model.dit.speaker_encoder)
        self.assertEqual(model.get_num_params()["speaker_encoder"], 0)

        restored = tiny_model()
        incompatible = restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])

    def test_speaker_conditioning_adds_only_the_explicit_speaker_topology(self):
        text_only = tiny_model()
        speaker_model = tiny_model(speaker=True)

        self.assertIsNotNone(speaker_model.dit.speaker_encoder)
        self.assertGreater(
            sum(parameter.numel() for parameter in speaker_model.parameters()),
            sum(parameter.numel() for parameter in text_only.parameters()),
        )
        self.assertTrue(any("speaker_encoder" in key for key in speaker_model.state_dict()))
        self.assertTrue(any("wk_speaker" in key for key in speaker_model.state_dict()))
        self.assertFalse(any("wk_latent" in key for key in speaker_model.state_dict()))


if __name__ == "__main__":
    unittest.main()
