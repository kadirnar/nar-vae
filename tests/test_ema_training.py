"""CPU-only regression tests for optimizer-step EMA semantics."""

from __future__ import annotations

import types
import unittest

import torch

from vyvotts.configuration import validate_pretraining_config
from vyvotts.finetune import EMACallback, EMAModel
from vyvotts.train import DEFAULT_TRAIN_CONFIG_PATH, _load_pretraining_yaml, pretrain, train


class PretrainingEntryPointTest(unittest.TestCase):
    def test_canonical_config_resolves_to_valid_random_pretraining(self):
        config = _load_pretraining_yaml(DEFAULT_TRAIN_CONFIG_PATH)

        validate_pretraining_config(config)
        self.assertEqual(config["training_stage"], "pretrain")
        self.assertEqual(config["model_initialization"], "random")
        self.assertIsNone(config["pretrained_checkpoint"])
        self.assertTrue(callable(pretrain))
        self.assertTrue(callable(train))


class EMATrainingTest(unittest.TestCase):
    class Wrapper(torch.nn.Module):
        def __init__(self, child, attribute):
            super().__init__()
            setattr(self, attribute, child)

    def test_callback_updates_once_after_configured_optimizer_step(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema = EMAModel(model, decay=0.5, device="cpu")
        callback = EMACallback(ema, update_every=2)
        control = object()

        with torch.no_grad():
            model.weight.fill_(3.0)
        callback.on_step_end(None, types.SimpleNamespace(global_step=1), control, model=model)
        self.assertEqual(ema.num_updates, 0)

        callback.on_step_end(None, types.SimpleNamespace(global_step=2), control, model=model)
        self.assertEqual(ema.num_updates, 1)
        torch.testing.assert_close(ema.shadow["weight"], torch.tensor([[2.0]]))

        # Repeated callback delivery for a step must not apply EMA twice.
        callback.on_step_end(None, types.SimpleNamespace(global_step=2), control, model=model)
        self.assertEqual(ema.num_updates, 1)
        torch.testing.assert_close(ema.shadow["weight"], torch.tensor([[2.0]]))

    def test_state_round_trip_preserves_resume_cadence(self):
        model = torch.nn.Linear(1, 1, bias=False)
        ema = EMAModel(model, decay=0.9, device="cpu")
        ema.update(model, step=10)

        restored = EMAModel(model, decay=0.1, device="cpu")
        restored.load_state_dict(ema.state_dict())

        self.assertEqual(restored.decay, 0.9)
        self.assertEqual(restored.num_updates, 1)
        self.assertEqual(restored.last_update_step, 10)
        self.assertFalse(restored.update(model, step=10))
        self.assertTrue(restored.update(model, step=11))

    def test_apply_and_restore_do_not_replace_parameter_storage(self):
        model = torch.nn.Linear(1, 1, bias=False)
        ema = EMAModel(model, decay=0.5, device="cpu")
        parameter_id = id(model.weight)
        original = model.weight.detach().clone()
        ema.shadow["weight"].fill_(4.0)

        ema.apply_shadow(model)
        self.assertEqual(id(model.weight), parameter_id)
        torch.testing.assert_close(model.weight, torch.tensor([[4.0]]))
        ema.restore(model)
        self.assertEqual(id(model.weight), parameter_id)
        torch.testing.assert_close(model.weight, original)

    def test_nested_ddp_and_compile_wrappers_update_canonical_ema_names(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema = EMAModel(model, decay=0.5, device="cpu")
        compiled = self.Wrapper(model, "_orig_mod")
        distributed_compiled = self.Wrapper(compiled, "module")

        with torch.no_grad():
            model.weight.fill_(3.0)
        self.assertTrue(ema.update(distributed_compiled, step=1))

        self.assertEqual(set(ema.shadow), {"weight"})
        torch.testing.assert_close(ema.shadow["weight"], torch.tensor([[2.0]]))

    def test_ema_update_fails_if_wrapper_resolves_to_a_different_topology(self):
        initialized = torch.nn.Linear(1, 1, bias=False)
        ema = EMAModel(initialized, decay=0.5, device="cpu")
        incompatible = self.Wrapper(torch.nn.Linear(1, 1, bias=True), "_orig_mod")
        del incompatible._orig_mod.weight

        with self.assertRaisesRegex(RuntimeError, "Missing.*weight"):
            ema.update(incompatible, step=1)
        self.assertEqual(ema.num_updates, 0)


if __name__ == "__main__":
    unittest.main()
