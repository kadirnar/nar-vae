"""Tests for synchronized realtime timing fields."""

import unittest
from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nar_vae.caching import (
    CacheDiTPoisonedError,
    CacheDiTRequestActiveError,
    CacheDiTStats,
)
from nar_vae.configuration import load_inference_settings
from nar_vae.inference import FlowMatchingTTSInference
from nar_vae.inference_realtime import RealtimeTTSInference, _mark_compiled_cuda_graph_step
from nar_vae.objectives import RECTIFIED_FLOW_OBJECTIVE, VP_DIFFUSION_OBJECTIVE
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


class ConstantVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        latents,
        conditioning_ids,
        timesteps,
        attention_mask=None,
        token_language_ids=None,
        alignment_mask=None,
    ):
        del conditioning_ids, timesteps, attention_mask, token_language_ids, alignment_mask
        return torch.ones_like(latents)


class RealtimeTimingTest(unittest.TestCase):
    def test_realtime_uses_the_shared_reference_duration_default(self):
        parameter = signature(RealtimeTTSInference).parameters["max_reference_seconds"]

        self.assertEqual(parameter.default, DEFAULT_MAX_REFERENCE_SECONDS)

    def test_compiled_cuda_marker_fails_clearly_on_an_unsupported_torch_build(self):
        with (
            patch("nar_vae.inference_realtime.torch.compiler", object()),
            self.assertRaisesRegex(RuntimeError, "cudagraph_mark_step_begin"),
        ):
            _mark_compiled_cuda_graph_step()

    @staticmethod
    def create_runtime() -> RealtimeTTSInference:
        tts = RealtimeTTSInference.__new__(RealtimeTTSInference)
        tts.device = torch.device("cpu")
        tts.cache_mode = None
        tts.compile_model = False
        tts.compile_mode = "reduce-overhead"
        tts._compiled_cache_session = None
        tts._uncompiled_backbone = None
        tts._uncompiled_decode = None
        tts._cache_finalizer = None
        tts.latent_size = 2
        tts.frame_rate = 4
        tts.sample_rate = 16
        tts.tokenizer = FakeTokenizer()
        tts.settings = load_inference_settings()
        tts.supports_voice_cloning = False
        tts.flow_model = ConstantVelocityModel()
        tts._decode = lambda latents: torch.zeros(1, 1, latents.shape[-1] * 4)
        return tts

    def test_context_manager_returns_and_closes_runtime(self):
        tts = self.create_runtime()

        with patch.object(tts, "close") as close:
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with tts as entered:
                    self.assertIs(entered, tts)
                    raise RuntimeError("generation failed")

        close.assert_called_once_with()

    def test_constructor_preserves_process_wide_backend_settings(self):
        before = (
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )

        def fake_base_init(runtime, **kwargs):
            del kwargs
            runtime.device = torch.device("cuda")

        try:
            with patch(
                "nar_vae.inference_realtime.FlowMatchingTTSInference.__init__",
                new=fake_base_init,
            ):
                RealtimeTTSInference("checkpoint.bin", device="cuda")

            self.assertEqual(
                (
                    torch.backends.cudnn.benchmark,
                    torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32,
                ),
                before,
            )
        finally:
            (
                torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
            ) = before

    def test_reports_first_step_and_first_audio(self):
        tts = self.create_runtime()
        tts.cache_mode = "none"

        audio, timings = tts.synthesize_fast(
            "test",
            num_steps=2,
            solver="euler",
            cfg_scale=1.0,
            cfg_mode="joint",
            duration=1.5,
            return_timing=True,
        )

        self.assertEqual(tuple(audio.shape), (24,))
        self.assertIn("ttft", timings)
        self.assertIn("ttfa", timings)
        self.assertLessEqual(timings["ttft"], timings["ttfa"])
        self.assertEqual(timings["total"], timings["ttfa"])
        self.assertAlmostEqual(
            timings["conditioning"]
            + timings["ode_sampling"]
            + timings["decoding"]
            + timings["output_transfer"],
            timings["ttfa"],
        )

    def test_rectified_flow_fast_profile_preserves_origin_solver_and_step_budget(self):
        tts = self.create_runtime()
        tts.model_manifest = SimpleNamespace(raw={"schema_version": 5})
        tts.generative_objective = RECTIFIED_FLOW_OBJECTIVE
        captured = {}

        def fake_sample(**kwargs):
            captured["num_steps"] = kwargs["num_steps"]
            captured["solver"] = kwargs["solver"]
            return torch.zeros(kwargs["latent_shape"])

        with patch("nar_vae.inference_realtime.ODESolver.sample", side_effect=fake_sample):
            tts.synthesize_fast("test", duration=1.5, cache_mode="none")

        self.assertEqual(captured, {"num_steps": 16, "solver": "euler"})

    def test_vp_profiles_keep_current_diffusion_solver_and_step_budgets(self):
        tts = self.create_runtime()
        tts.generative_objective = VP_DIFFUSION_OBJECTIVE

        self.assertEqual(
            (tts.generation_profile("quality").num_steps, tts.generation_profile("quality").solver),
            (32, "ddim"),
        )
        self.assertEqual(
            (tts.generation_profile("fast").num_steps, tts.generation_profile("fast").solver),
            (8, "ddim"),
        )

    def test_metadata_absent_protocol_runtime_uses_rectified_flow_profiles(self):
        tts = self.create_runtime()

        self.assertEqual(
            (tts.generation_profile("quality").num_steps, tts.generation_profile("quality").solver),
            (50, "heun"),
        )

    def test_poisoned_model_is_rejected_before_uncached_realtime_preprocessing(self):
        tts = self.create_runtime()
        tts.flow_model._nar_vae_cache_dit_poison_reason = "disable failed"

        with self.assertRaises(CacheDiTPoisonedError):
            tts.synthesize_fast("test", duration=1.5, cache_mode="none")

    def test_poisoned_model_is_rejected_before_uncached_base_preprocessing(self):
        tts = self.create_runtime()
        tts.flow_model._nar_vae_cache_dit_poison_reason = "disable failed"

        with self.assertRaises(CacheDiTPoisonedError):
            tts.synthesize("test", duration=1.5, cache_mode="none", show_progress=False)

    def test_conventional_single_request_apis_route_through_managed_lifecycle(self):
        tts = self.create_runtime()
        expected = torch.ones(4)
        profile = tts.generation_profile("turbo")

        with patch.object(tts, "synthesize_fast", return_value=expected) as managed:
            actual = tts.synthesize(
                "test",
                num_steps=8,
                solver="euler",
                cache_mode="cache_dit",
                duration=1.5,
            )
            self.assertIs(actual, expected)
            self.assertEqual(managed.call_args.kwargs["cache_mode"], "cache_dit")

            actual = tts.synthesize_with_config("test", profile, duration=1.5)
            self.assertIs(actual, expected)
            self.assertIs(managed.call_args.kwargs["config"], profile)

    def test_compiled_cache_runtime_rejects_unmanaged_batching(self):
        tts = self.create_runtime()
        tts.compile_model = True
        tts._compiled_cache_session = object()

        with self.assertRaisesRegex(RuntimeError, "one utterance per managed request"):
            tts.synthesize_batch(["one", "two"])

    def test_compiled_uncached_batch_marks_one_cuda_graph_request(self):
        tts = self.create_runtime()
        tts.compile_model = True
        tts.device = torch.device("cuda")
        expected = [torch.ones(2)]

        with (
            patch("nar_vae.inference_realtime._mark_compiled_cuda_graph_step") as marker,
            patch.object(
                FlowMatchingTTSInference,
                "synthesize_batch",
                return_value=expected,
            ) as base_batch,
        ):
            actual = tts.synthesize_batch(["one"])

        self.assertIs(actual, expected)
        marker.assert_called_once_with()
        base_batch.assert_called_once()

    def test_compilation_installs_cache_hooks_before_compiling_the_backbone(self):
        tts = self.create_runtime()
        original_backbone = object()
        hooked_backbone = object()
        compiled_backbone = object()
        original_decode = object()
        compiled_decode = object()
        tts.flow_model = SimpleNamespace(dit=original_backbone)
        tts._decode = original_decode
        tts.cache_mode = "cache_dit"
        events = []

        class FakeAPI:
            @staticmethod
            def set_compile_configs(**kwargs):
                events.append(("configure", kwargs))

        class FakeCacheDiTSession:
            api = FakeAPI()
            backbone = hooked_backbone

            def __init__(self, model, *, num_steps):
                self.model = model
                self.num_steps = num_steps
                self.closed = False

            def __enter__(self):
                events.append(("enable", self.model.dit))
                return self

            def close(self):
                self.closed = True

        class FakeFinalizer:
            alive = True

            def detach(self):
                self.alive = False

        def fake_compile(model, **kwargs):
            events.append(("compile", model, kwargs))
            return compiled_backbone if model is hooked_backbone else compiled_decode

        with (
            patch("nar_vae.inference_realtime.CacheDiTSession", FakeCacheDiTSession),
            patch("nar_vae.inference_realtime.torch.compile", side_effect=fake_compile),
            patch("nar_vae.inference_realtime.weakref.finalize", return_value=FakeFinalizer()),
        ):
            tts._enable_compilation("reduce-overhead")

        self.assertEqual(events[0], ("enable", original_backbone))
        self.assertEqual(
            events[1],
            ("configure", {"cuda_graphs": True, "use_fast_math": False}),
        )
        self.assertEqual(
            events[2],
            ("compile", hooked_backbone, {"mode": "reduce-overhead", "dynamic": False}),
        )
        self.assertEqual(
            events[3],
            (
                "compile",
                original_decode,
                {
                    "dynamic": False,
                    "options": {
                        "triton.cudagraphs": True,
                        "coordinate_descent_tuning": False,
                        "coordinate_descent_check_all_directions": False,
                    },
                },
            ),
        )
        self.assertIs(tts.flow_model.dit, compiled_backbone)
        self.assertIs(tts._decode, compiled_decode)
        self.assertIs(tts._uncompiled_backbone, original_backbone)
        self.assertTrue(tts.compile_model)

        session = tts._compiled_cache_session
        tts.close()
        self.assertTrue(session.closed)
        self.assertIs(tts.flow_model.dit, original_backbone)
        self.assertIs(tts._decode, original_decode)
        self.assertFalse(tts.compile_model)

    def test_compilation_keeps_the_primary_error_and_restores_eager_references(self):
        tts = self.create_runtime()
        original_backbone = object()
        original_decode = object()
        tts.flow_model = SimpleNamespace(dit=original_backbone)
        tts._decode = original_decode
        tts.cache_mode = "cache_dit"

        class FakeAPI:
            @staticmethod
            def set_compile_configs(**kwargs):
                del kwargs

        class FailingCleanupSession:
            api = FakeAPI()
            backbone = object()

            def __init__(self, model, *, num_steps):
                del model, num_steps

            def __enter__(self):
                return self

            def close(self):
                raise RuntimeError("cleanup exploded")

        with (
            patch("nar_vae.inference_realtime.CacheDiTSession", FailingCleanupSession),
            patch(
                "nar_vae.inference_realtime.torch.compile",
                side_effect=ValueError("compile exploded"),
            ),
            self.assertRaisesRegex(RuntimeError, "could not prepare") as raised,
        ):
            tts._enable_compilation("reduce-overhead")

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("compile exploded", str(raised.exception.__cause__))
        self.assertTrue(any("cleanup also failed" in note for note in raised.exception.__notes__))
        self.assertIs(tts.flow_model.dit, original_backbone)
        self.assertIs(tts._decode, original_decode)
        self.assertFalse(tts.compile_model)

    def test_close_restores_eager_references_when_cache_cleanup_fails(self):
        tts = self.create_runtime()
        eager_backbone = object()
        eager_decode = object()
        tts.flow_model = SimpleNamespace(dit=object())
        tts._decode = object()
        tts._uncompiled_backbone = eager_backbone
        tts._uncompiled_decode = eager_decode
        tts._compiled_cache_failed = True
        tts.compile_model = True

        class FailingSession:
            @staticmethod
            def close():
                raise RuntimeError("cleanup exploded")

        tts._compiled_cache_session = FailingSession()

        with self.assertRaisesRegex(RuntimeError, "cleanup exploded"):
            tts.close()

        self.assertIs(tts.flow_model.dit, eager_backbone)
        self.assertIs(tts._decode, eager_decode)
        self.assertIsNone(tts._compiled_cache_session)
        self.assertFalse(tts._compiled_cache_failed)
        self.assertFalse(tts.compile_model)

    def test_close_preserves_compiled_runtime_while_cache_request_is_active(self):
        tts = self.create_runtime()
        compiled_backbone = object()
        compiled_decode = object()
        eager_backbone = object()
        eager_decode = object()
        session = SimpleNamespace(
            close=lambda: (_ for _ in ()).throw(CacheDiTRequestActiveError("request is running"))
        )
        finalizer = SimpleNamespace(alive=True, detach=lambda: self.fail("detached finalizer"))
        tts.flow_model = SimpleNamespace(dit=compiled_backbone)
        tts._decode = compiled_decode
        tts._uncompiled_backbone = eager_backbone
        tts._uncompiled_decode = eager_decode
        tts._compiled_cache_session = session
        tts._cache_finalizer = finalizer
        tts._compiled_cache_failed = True
        tts.compile_model = True

        with self.assertRaisesRegex(CacheDiTRequestActiveError, "request is running"):
            tts.close()

        self.assertIs(tts.flow_model.dit, compiled_backbone)
        self.assertIs(tts._decode, compiled_decode)
        self.assertIs(tts._compiled_cache_session, session)
        self.assertIs(tts._uncompiled_backbone, eager_backbone)
        self.assertIs(tts._uncompiled_decode, eager_decode)
        self.assertIs(tts._cache_finalizer, finalizer)
        self.assertTrue(tts._compiled_cache_failed)
        self.assertTrue(tts.compile_model)

    def test_turbo_profile_uses_block_cache_at_the_same_step_count(self):
        tts = self.create_runtime()
        tts.generative_objective = VP_DIFFUSION_OBJECTIVE
        tts.diffusion_schedule_shift = 1.0
        captured = {}

        class FakeCacheDiTSession:
            def __init__(self, model, *, num_steps):
                captured["model"] = model
                captured["session_steps"] = num_steps
                self.stats = SimpleNamespace(
                    version="1.5.0",
                    cached_steps=5,
                    executed_steps=num_steps,
                    cache_ratio=5 / num_steps,
                    baseline_block_calls=num_steps * 24,
                    estimated_block_calls=num_steps * 24 - 5 * 16,
                    block_work_reduction=(5 * 16) / (num_steps * 24),
                )

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

        def fake_sample(**kwargs):
            captured["solver_steps"] = kwargs["num_steps"]
            captured["solver"] = kwargs["solver"]
            captured["generative_objective"] = kwargs["generative_objective"]
            captured["scm_ctx"] = kwargs["scm_ctx"]
            captured["fuse_cfg_branches"] = kwargs["fuse_cfg_branches"]
            return torch.zeros(kwargs["latent_shape"])

        with (
            patch("nar_vae.inference_realtime.CacheDiTSession", FakeCacheDiTSession),
            patch("nar_vae.inference_realtime.ODESolver.sample", side_effect=fake_sample),
            patch("nar_vae.inference_realtime.create_scm_context") as legacy_cache,
        ):
            _, timings = tts.synthesize_fast(
                "test",
                config=tts.generation_profile("turbo"),
                duration=1.5,
                return_timing=True,
            )

        self.assertEqual(captured["session_steps"], 16)
        self.assertEqual(captured["solver_steps"], 16)
        self.assertEqual(captured["solver"], "euler")
        self.assertEqual(captured["generative_objective"], VP_DIFFUSION_OBJECTIVE)
        self.assertIsNone(captured["scm_ctx"])
        self.assertTrue(captured["fuse_cfg_branches"])
        legacy_cache.assert_not_called()
        self.assertEqual(timings["cached_steps"], 5.0)
        self.assertEqual(timings["total_steps"], 16.0)
        self.assertLess(timings["estimated_block_calls"], timings["baseline_block_calls"])

    def test_cache_dit_rejects_an_explicit_partial_guidance_window(self):
        tts = self.create_runtime()

        with self.assertRaisesRegex(ValueError, "fixed CFG batch"):
            tts.synthesize_fast(
                "test",
                config=tts.generation_profile("fast"),
                cache_mode="cache_dit",
                cfg_scale=2.0,
                cfg_min_t=0.1,
                duration=1.5,
            )

    def test_explicit_none_allows_overriding_the_turbo_solver(self):
        tts = self.create_runtime()
        captured = {}

        def fake_sample(**kwargs):
            captured["solver"] = kwargs["solver"]
            captured["fuse_cfg_branches"] = kwargs["fuse_cfg_branches"]
            return torch.zeros(kwargs["latent_shape"])

        class UnexpectedCacheSession:
            def __init__(self, *args, **kwargs):
                del args, kwargs
                raise AssertionError("Cache-DiT session should not be created")

        with (
            patch("nar_vae.inference_realtime.ODESolver.sample", side_effect=fake_sample),
            patch("nar_vae.inference_realtime.CacheDiTSession", UnexpectedCacheSession),
        ):
            tts.synthesize_fast(
                "test",
                config=tts.generation_profile("turbo"),
                cache_mode="none",
                solver="heun",
                duration=1.5,
            )

        self.assertEqual(captured["solver"], "heun")
        self.assertFalse(captured["fuse_cfg_branches"])

    def test_compiled_runtime_fuses_cfg_without_cache(self):
        tts = self.create_runtime()
        tts.compile_model = True
        captured = {}

        def fake_sample(**kwargs):
            captured["fuse_cfg_branches"] = kwargs["fuse_cfg_branches"]
            return torch.zeros(kwargs["latent_shape"])

        with patch("nar_vae.inference_realtime.ODESolver.sample", side_effect=fake_sample):
            tts.synthesize_fast(
                "test",
                config=tts.generation_profile("turbo").with_overrides(cache_mode="none"),
                duration=1.5,
            )

        self.assertTrue(captured["fuse_cfg_branches"])

    def test_failed_request_does_not_expose_previous_cache_statistics(self):
        tts = self.create_runtime()
        tts.last_cache_stats = CacheDiTStats(cached_steps=7, executed_steps=8)

        with (
            patch(
                "nar_vae.inference_realtime.ODESolver.sample",
                side_effect=RuntimeError("sampling failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "sampling failed"),
        ):
            tts.synthesize_fast(
                "test",
                num_steps=2,
                solver="euler",
                cfg_scale=1.0,
                cfg_mode="joint",
                duration=1.5,
                cache_mode="none",
            )

        self.assertEqual(tts.last_cache_stats, CacheDiTStats())

    def test_compiled_cache_hooks_cannot_be_bypassed_per_request(self):
        tts = self.create_runtime()
        tts.compile_model = True
        tts._compiled_cache_session = object()

        with self.assertRaisesRegex(RuntimeError, "cannot bypass"):
            tts.synthesize_fast(
                "test",
                config=tts.generation_profile("turbo"),
                cache_mode="none",
                duration=1.5,
            )

    def test_failed_compiled_cache_request_invalidates_the_runtime(self):
        tts = self.create_runtime()
        tts.compile_model = True

        class FakeRequest:
            def __init__(self, session):
                self.session = session

            def __enter__(self):
                return self.session

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback

        class FakeCompiledCache:
            def request(self, num_steps):
                self.num_steps = num_steps
                return FakeRequest(self)

            def close(self):
                self.closed = True

        compiled_cache = FakeCompiledCache()
        tts._compiled_cache_session = compiled_cache

        with (
            patch(
                "nar_vae.inference_realtime.ODESolver.sample",
                side_effect=RuntimeError("sampling failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "sampling failed"),
        ):
            tts.synthesize_fast(
                "test",
                config=tts.generation_profile("turbo"),
                duration=1.5,
            )

        self.assertTrue(tts._compiled_cache_failed)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            tts.synthesize_fast("test", duration=1.5, cache_mode="none")

        tts.close()
        self.assertFalse(tts._compiled_cache_failed)
        self.assertTrue(compiled_cache.closed)


if __name__ == "__main__":
    unittest.main()
