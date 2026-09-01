"""Tests for the real Cache-DiT turbo integration."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nar_vae.caching import (
    CacheDiTPoisonedError,
    CacheDiTRequestActiveError,
    CacheDiTSession,
    CacheDiTStats,
    CacheDiTUnavailableError,
    assert_cache_dit_healthy,
)
from nar_vae.caching.cache_dit import _SESSION_LOCK, _load_cache_dit, _PersistentCacheDiTRequest
from nar_vae.models.dit import DiTBlock
from nar_vae.models.flow_matching import FlowMatchingEchoDiT
from nar_vae.solvers.ode_solver import ODESolver


class IdentityAdaLN(torch.nn.Module):
    def forward(self, hidden_states, cond_embed):
        del cond_embed
        return hidden_states, torch.ones_like(hidden_states)


class RecordingAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_cache = None
        self.speaker_cache = None
        self.latent_cache = None

    def forward(
        self,
        hidden_states,
        text_mask,
        speaker_mask,
        freqs_cis,
        kv_cache_text,
        kv_cache_speaker,
        start_pos,
        kv_cache_latent,
    ):
        del text_mask, speaker_mask, freqs_cis, start_pos
        self.text_cache = kv_cache_text
        self.speaker_cache = kv_cache_speaker
        self.latent_cache = kv_cache_latent
        return torch.zeros_like(hidden_states)


class ZeroMLP(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states)


class CacheDiTTest(unittest.TestCase):
    def test_dit_block_selects_its_own_layer_caches(self):
        block = DiTBlock.__new__(DiTBlock)
        torch.nn.Module.__init__(block)
        block.layer_index = 1
        block.attention_adaln = IdentityAdaLN()
        block.mlp_adaln = IdentityAdaLN()
        block.attention = RecordingAttention()
        block.mlp = ZeroMLP()

        def marker(value):
            return torch.tensor([value]), torch.tensor([-value])

        block(
            x=torch.zeros(1, 2, 3),
            cond_embed=torch.zeros(1, 1, 9),
            text_mask=torch.ones(1, 1, dtype=torch.bool),
            speaker_mask=torch.ones(1, 1, dtype=torch.bool),
            freqs_cis=torch.zeros(1),
            kv_cache_text=[marker(1), marker(2)],
            kv_cache_speaker=[marker(3), marker(4)],
            start_pos=None,
            kv_cache_latent=None,
        )

        torch.testing.assert_close(block.attention.text_cache[0], torch.tensor([2]))
        torch.testing.assert_close(block.attention.speaker_cache[0], torch.tensor([4]))
        self.assertIsNone(block.attention.latent_cache)

    def test_missing_dependency_has_an_actionable_error(self):
        model = SimpleNamespace(dit=SimpleNamespace(blocks=torch.nn.ModuleList()))
        with patch(
            "nar_vae.caching.cache_dit.import_module",
            side_effect=ModuleNotFoundError(
                "No module named 'cache_dit'",
                name="cache_dit",
            ),
        ):
            with self.assertRaisesRegex(
                CacheDiTUnavailableError,
                r"python -m pip install -e \.",
            ):
                CacheDiTSession(model, num_steps=16).__enter__()

    def test_transitive_missing_dependency_is_not_masked(self):
        model = SimpleNamespace(dit=SimpleNamespace(blocks=torch.nn.ModuleList()))
        missing_dependency = ModuleNotFoundError(
            "No module named 'diffusers'",
            name="diffusers",
        )
        with patch(
            "nar_vae.caching.cache_dit.import_module",
            side_effect=missing_dependency,
        ):
            with self.assertRaises(ModuleNotFoundError) as caught:
                CacheDiTSession(model, num_steps=16).__enter__()

        self.assertIs(caught.exception, missing_dependency)

    def test_base_exception_during_setup_releases_the_session_lock(self):
        with patch(
            "nar_vae.caching.cache_dit._load_cache_dit",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                CacheDiTSession(SimpleNamespace(), num_steps=16).__enter__()

        self.assertTrue(_SESSION_LOCK.acquire(blocking=False))
        _SESSION_LOCK.release()

    def test_failed_setup_cleanup_poison_preserves_original_error_and_blocks_reuse(self):
        class FakeBlockAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_api = types.ModuleType("cache_dit")
        fake_api.BlockAdapter = FakeBlockAdapter
        fake_api.DBCacheConfig = FakeConfig
        fake_api.DMDCalibratorConfig = FakeConfig
        fake_api.ForwardPattern = SimpleNamespace(Pattern_3="pattern-3")
        setup_error = RuntimeError("enable failed after hook installation")

        def enable_cache(adapter, **kwargs):
            del kwargs
            adapter.kwargs["transformer"]._backend_hook_installed = True
            raise setup_error

        def disable_cache(adapter):
            del adapter
            raise RuntimeError("disable failed")

        fake_api.enable_cache = enable_cache
        fake_api.disable_cache = disable_cache
        fake_api.summary = lambda backbone, logging=False: []

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([ZeroMLP(), ZeroMLP()])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = Backbone()

        model = Model()
        with (
            patch.dict(sys.modules, {"cache_dit": fake_api}),
            self.assertWarnsRegex(RuntimeWarning, "cleanup also failed"),
            self.assertRaises(RuntimeError) as caught,
        ):
            CacheDiTSession(model, num_steps=16).__enter__()

        self.assertIs(caught.exception, setup_error)
        self.assertTrue(model.dit._backend_hook_installed)
        with self.assertRaisesRegex(CacheDiTPoisonedError, "Reconstruct"):
            assert_cache_dit_healthy(model)
        with self.assertRaises(CacheDiTPoisonedError):
            CacheDiTSession(model, num_steps=16).__enter__()
        with self.assertRaises(CacheDiTPoisonedError):
            ODESolver.sample(
                model,
                torch.tensor([[1]]),
                num_steps=1,
                latent_shape=(1, 1, 1),
            )

        self.assertTrue(_SESSION_LOCK.acquire(blocking=False))
        _SESSION_LOCK.release()

    def test_failed_normal_cleanup_poison_blocks_uncached_solver(self):
        class FakeBlockAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_api = types.ModuleType("cache_dit")
        fake_api.BlockAdapter = FakeBlockAdapter
        fake_api.DBCacheConfig = FakeConfig
        fake_api.DMDCalibratorConfig = FakeConfig
        fake_api.ForwardPattern = SimpleNamespace(Pattern_3="pattern-3")
        fake_api.enable_cache = lambda adapter, **kwargs: None
        fake_api.disable_cache = lambda adapter: (_ for _ in ()).throw(
            RuntimeError("disable failed")
        )
        fake_api.summary = lambda backbone, logging=False: []

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([ZeroMLP(), ZeroMLP()])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = Backbone()

        model = Model()
        with patch.dict(sys.modules, {"cache_dit": fake_api}):
            with self.assertRaisesRegex(RuntimeError, "disable failed"):
                with CacheDiTSession(model, num_steps=16):
                    pass

        with self.assertRaises(CacheDiTPoisonedError):
            ODESolver.sample(
                model,
                torch.tensor([[1]]),
                num_steps=1,
                latent_shape=(1, 1, 1),
            )

    def test_session_uses_dbcache_dmd_collects_stats_and_restores_model(self):
        calls = SimpleNamespace(
            enabled=False,
            disabled=False,
            adapter=None,
            cache_config=None,
            calibrator_config=None,
        )

        class FakeBlockAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_api = types.ModuleType("cache_dit")
        fake_api.__version__ = "1.5.0"
        fake_api.BlockAdapter = FakeBlockAdapter
        fake_api.DBCacheConfig = FakeConfig
        fake_api.DMDCalibratorConfig = FakeConfig
        fake_api.ForwardPattern = SimpleNamespace(Pattern_3="pattern-3")
        fake_api.steps_mask = lambda mask_policy, total_steps: [1, 0] * (total_steps // 2)

        def enable_cache(adapter, *, cache_config, calibrator_config):
            calls.enabled = True
            calls.adapter = adapter
            calls.cache_config = cache_config
            calls.calibrator_config = calibrator_config
            adapter.kwargs["transformer"]._is_cached = True

        def disable_cache(adapter):
            calls.disabled = True
            del adapter.kwargs["transformer"]._is_cached

        fake_api.enable_cache = enable_cache
        fake_api.disable_cache = disable_cache
        fake_api.summary = lambda backbone, logging=False: [
            SimpleNamespace(
                accumulated_cached_steps=6,
                accumulated_transformer_executed_steps=16,
            )
        ]

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([ZeroMLP(), ZeroMLP(), ZeroMLP()])

        model = SimpleNamespace(dit=Backbone())
        with patch.dict(sys.modules, {"cache_dit": fake_api}):
            with CacheDiTSession(model, num_steps=16) as session:
                self.assertTrue(calls.enabled)
                self.assertTrue(model.dit._is_cached)

        self.assertTrue(calls.disabled)
        self.assertFalse(hasattr(model.dit, "_is_cached"))
        self.assertEqual(session.stats.version, "1.5.0")
        self.assertEqual(session.stats.cached_steps, 6)
        self.assertEqual(session.stats.executed_steps, 16)
        self.assertEqual(session.stats.cache_ratio, 0.375)
        self.assertEqual(session.stats.block_count, 3)
        self.assertEqual(session.stats.computed_blocks_per_cached_step, 1)
        self.assertEqual(session.stats.baseline_block_calls, 48)
        self.assertEqual(session.stats.estimated_block_calls, 36)
        self.assertEqual(session.stats.block_work_reduction, 0.25)
        self.assertIs(calls.adapter.kwargs["transformer"], model.dit)
        self.assertIs(calls.adapter.kwargs["blocks"], model.dit.blocks)
        self.assertEqual(calls.adapter.kwargs["forward_pattern"], "pattern-3")
        self.assertFalse(calls.adapter.kwargs["has_separate_cfg"])
        self.assertEqual(calls.cache_config.kwargs["num_inference_steps"], 16)
        self.assertEqual(
            calls.cache_config.kwargs["steps_computation_mask"],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        )
        self.assertEqual(calls.cache_config.kwargs["steps_computation_policy"], "static")
        self.assertEqual(calls.calibrator_config.kwargs["dmd_history"], 6)
        self.assertEqual(calls.calibrator_config.kwargs["dmd_rank"], 0)

    def test_persistent_request_refreshes_context_without_removing_hooks(self):
        calls = SimpleNamespace(refreshes=[], disabled=False)

        class FakeBlockAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_api = types.ModuleType("cache_dit")
        fake_api.__version__ = "1.5.0"
        fake_api.BlockAdapter = FakeBlockAdapter
        fake_api.DBCacheConfig = FakeConfig
        fake_api.DMDCalibratorConfig = FakeConfig
        fake_api.ForwardPattern = SimpleNamespace(Pattern_3="pattern-3")

        def enable_cache(adapter, **kwargs):
            del kwargs
            adapter.kwargs["transformer"]._is_cached = True

        def refresh_context(backbone, **kwargs):
            calls.refreshes.append((backbone, kwargs))

        def disable_cache(adapter):
            calls.disabled = True
            del adapter.kwargs["transformer"]._is_cached

        fake_api.enable_cache = enable_cache
        fake_api.refresh_context = refresh_context
        fake_api.disable_cache = disable_cache
        fake_api.summary = lambda backbone, logging=False: [
            SimpleNamespace(
                accumulated_cached_steps=4,
                accumulated_transformer_executed_steps=18,
            )
        ]

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([ZeroMLP(), ZeroMLP(), ZeroMLP()])

        model = SimpleNamespace(dit=Backbone())
        with patch.dict(sys.modules, {"cache_dit": fake_api}):
            with CacheDiTSession(model, num_steps=16) as session:
                with session.request(18) as request_session:
                    self.assertIs(request_session, session)
                    self.assertTrue(model.dit._is_cached)
                self.assertTrue(model.dit._is_cached)
                self.assertFalse(calls.disabled)
                self.assertEqual(session.stats.cached_steps, 4)
                self.assertEqual(session.stats.executed_steps, 18)
                with session.request(16):
                    self.assertEqual(
                        session.stats,
                        CacheDiTStats(
                            version="1.5.0",
                            block_count=3,
                            computed_blocks_per_cached_step=1,
                        ),
                    )

                request = session.request(16)
                request.__enter__()
                try:
                    with self.assertRaisesRegex(CacheDiTRequestActiveError, "request is running"):
                        session.close()
                    self.assertTrue(session._enabled)
                    self.assertTrue(model.dit._is_cached)
                    self.assertFalse(calls.disabled)
                finally:
                    request.__exit__(None, None, None)

        self.assertTrue(calls.disabled)
        self.assertFalse(hasattr(model.dit, "_is_cached"))
        self.assertEqual(len(calls.refreshes), 3)
        refreshed_backbone, refresh_kwargs = calls.refreshes[0]
        self.assertIs(refreshed_backbone, model.dit)
        self.assertEqual(refresh_kwargs["cache_config"].kwargs["num_inference_steps"], 18)
        self.assertEqual(refresh_kwargs["calibrator_config"].kwargs["dmd_history"], 6)
        self.assertFalse(refresh_kwargs["verbose"])

    def test_failed_persistent_refresh_invalidates_hooks_and_stale_stats(self):
        calls = SimpleNamespace(disabled=False)

        class FakeBlockAdapter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_api = types.ModuleType("cache_dit")
        fake_api.__version__ = "1.5.0"
        fake_api.BlockAdapter = FakeBlockAdapter
        fake_api.DBCacheConfig = FakeConfig
        fake_api.DMDCalibratorConfig = FakeConfig
        fake_api.ForwardPattern = SimpleNamespace(Pattern_3="pattern-3")

        def enable_cache(adapter, **kwargs):
            del kwargs
            adapter.kwargs["transformer"]._is_cached = True

        def refresh_context(backbone, **kwargs):
            del backbone, kwargs
            raise RuntimeError("refresh failed")

        def disable_cache(adapter):
            calls.disabled = True
            del adapter.kwargs["transformer"]._is_cached

        fake_api.enable_cache = enable_cache
        fake_api.refresh_context = refresh_context
        fake_api.disable_cache = disable_cache
        fake_api.summary = lambda backbone, logging=False: []

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([ZeroMLP(), ZeroMLP(), ZeroMLP()])

        model = SimpleNamespace(dit=Backbone())
        with patch.dict(sys.modules, {"cache_dit": fake_api}):
            with CacheDiTSession(model, num_steps=16) as session:
                session.stats = CacheDiTStats(cached_steps=12, executed_steps=16)
                with self.assertRaisesRegex(RuntimeError, "refresh failed"):
                    with session.request(16):
                        pass

                self.assertTrue(calls.disabled)
                self.assertFalse(hasattr(model.dit, "_is_cached"))
                self.assertEqual(session.stats.cached_steps, 0)
                self.assertEqual(session.stats.executed_steps, 0)
                with self.assertRaisesRegex(RuntimeError, "not active"):
                    session.request(16)

        self.assertTrue(_SESSION_LOCK.acquire(blocking=False))
        _SESSION_LOCK.release()

    def test_failed_persistent_inference_invalidates_the_session(self):
        class FakeSession:
            def __init__(self):
                self._request_lock = Lock()
                self.closed = False
                self.concurrent_acquire_succeeded = None

            def refresh(self, num_steps):
                self.num_steps = num_steps

            def _close_locked(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                self.concurrent_acquire_succeeded = self._request_lock.acquire(blocking=False)
                if self.concurrent_acquire_succeeded:
                    self._request_lock.release()
                self.closed = True

        session = FakeSession()
        request = _PersistentCacheDiTRequest(session, 16)

        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with request as entered:
                self.assertIs(entered, session)
                raise RuntimeError("inference failed")

        self.assertTrue(session.closed)
        self.assertFalse(session.concurrent_acquire_succeeded)
        self.assertTrue(session._request_lock.acquire(blocking=False))
        session._request_lock.release()


@unittest.skipUnless(importlib.util.find_spec("cache_dit"), "Cache-DiT dependency not installed")
class CacheDiTInstalledIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _load_cache_dit()
        except CacheDiTUnavailableError as exc:
            raise unittest.SkipTest(
                f"Compatible Cache-DiT dependency not installed: {exc}"
            ) from exc

    def test_real_cache_dit_reduces_block_work_at_fixed_step_count(self):
        class CountingBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return hidden_states + 0.001

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([CountingBlock() for _ in range(24)])

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                for block in self.blocks:
                    hidden_states = block(hidden_states=hidden_states)
                return hidden_states

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit = Backbone()

        model = Model()
        total_steps = 16
        with CacheDiTSession(model, num_steps=total_steps) as session:
            for block in model.dit.blocks:
                block.calls = 0
            hidden_states = torch.zeros(1, 4, 8)
            for _ in range(total_steps):
                hidden_states = model.dit(hidden_states)

        block_calls = sum(block.calls for block in model.dit.blocks)
        self.assertEqual(session.stats.cached_steps, 3)
        self.assertLess(block_calls, total_steps * len(model.dit.blocks))
        self.assertEqual(block_calls, session.stats.estimated_block_calls)
        self.assertEqual(session.stats.executed_steps, total_steps)

    def test_real_cache_dit_runs_the_echo_dit_cfg_path(self):
        model = FlowMatchingEchoDiT(
            latent_size=4,
            model_size=8,
            num_layers=6,
            num_heads=2,
            intermediate_size=16,
            text_vocab_size=32,
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
            use_speaker_conditioning=False,
        ).eval()
        original_blocks = list(model.dit.blocks)
        block_calls = [0] * len(original_blocks)
        handles = []
        for index, block in enumerate(original_blocks):

            def count_call(module, args, output, *, index=index):
                del module, args, output
                block_calls[index] += 1

            handles.append(block.register_forward_hook(count_call))

        latents = torch.randn(1, 4, 4)
        conditioning_ids = torch.tensor([[1, 2, 3]])
        total_steps = 16
        try:
            with torch.no_grad(), CacheDiTSession(model, num_steps=total_steps) as session:
                with self.assertRaisesRegex(RuntimeError, "concurrent or nested"):
                    CacheDiTSession(model, num_steps=total_steps).__enter__()
                block_calls[:] = [0] * len(block_calls)
                for step in range(total_steps):
                    output = model.forward_with_cfg(
                        latents,
                        conditioning_ids,
                        torch.tensor([step / (total_steps - 1)]),
                        cfg_scale=2.0,
                        cfg_mode="independent",
                        cfg_scale_text=2.5,
                        cfg_scale_speaker=2.0,
                        fuse_cfg_branches=True,
                    )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(tuple(output.shape), tuple(latents.shape))
        self.assertEqual(session.stats.cached_steps, 3)
        self.assertEqual(session.stats.executed_steps, total_steps)
        self.assertLess(sum(block_calls), total_steps * len(original_blocks))
        self.assertEqual(sum(block_calls), session.stats.estimated_block_calls)
        self.assertEqual(list(model.dit.blocks), original_blocks)


if __name__ == "__main__":
    unittest.main()
