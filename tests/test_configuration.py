"""Tests for packaged TOML inference profiles."""

import unittest

from vyvotts.configuration import load_inference_settings


class InferenceConfigurationTest(unittest.TestCase):
    def test_profiles_define_descending_compute_budgets(self):
        settings = load_inference_settings()

        self.assertFalse(hasattr(settings, "optimization"))

        self.assertEqual(settings.profile("quality").solver, "heun")
        self.assertEqual(settings.profile("fast").solver, "euler")
        self.assertEqual(settings.profile("fast").cache_mode, "none")
        self.assertGreater(
            settings.profile("quality").num_steps,
            settings.profile("balanced").num_steps,
        )
        self.assertGreater(
            settings.profile("balanced").num_steps,
            settings.profile("fast").num_steps,
        )
        for profile in settings.profiles.values():
            self.assertEqual(profile.cfg_scale, 1.0)
            self.assertEqual(profile.cfg_mode, "joint")
            self.assertEqual(profile.cfg_scale_text, 0.0)
            self.assertEqual(profile.cfg_scale_speaker, 0.0)

    def test_turbo_uses_cache_without_reducing_solver_steps(self):
        settings = load_inference_settings()
        fast = settings.profile("fast")
        turbo = settings.profile("turbo")

        self.assertEqual(turbo.num_steps, fast.num_steps)
        self.assertEqual(turbo.solver, fast.solver)
        self.assertEqual(turbo.cache_mode, "cache_dit")
        self.assertEqual((turbo.cfg_min_t, turbo.cfg_max_t), (0.0, 1.0))

        with self.assertRaisesRegex(ValueError, "Euler"):
            turbo.with_overrides(solver="heun")
        with self.assertRaisesRegex(ValueError, "fixed CFG batch"):
            turbo.with_overrides(cfg_scale=2.0, cfg_min_t=0.1)
        with self.assertRaisesRegex(ValueError, "alternating CFG"):
            turbo.with_overrides(cfg_scale=2.0, cfg_mode="alternating")
        with self.assertRaisesRegex(ValueError, "fixed CFG batch"):
            turbo.with_overrides(
                cfg_mode="independent",
                cfg_scale=1.0,
                cfg_scale_text=2.0,
                cfg_min_t=0.1,
            )
        with self.assertRaisesRegex(ValueError, "alternating CFG"):
            turbo.with_overrides(
                cfg_mode="alternating",
                cfg_scale=1.0,
                cfg_scale_text=2.0,
                cfg_scale_speaker=1.0,
            )
        with self.assertRaisesRegex(ValueError, "at least 8 steps"):
            turbo.with_overrides(num_steps=7)

        uncached = turbo.with_overrides(cache_mode="none", solver="heun")
        self.assertEqual(uncached.cache_mode, "none")
        self.assertEqual(uncached.solver, "heun")

    def test_duration_estimate_fails_instead_of_truncating_at_the_ceiling(self):
        duration = load_inference_settings().duration

        self.assertEqual(duration.estimate("Hi"), 1.5)
        self.assertGreater(duration.estimate("A complete sentence for speech."), 1.5)
        with self.assertRaisesRegex(ValueError, "exceeds the configured maximum"):
            duration.estimate("x" * 1000)
        with self.assertRaisesRegex(ValueError, "at least"):
            duration.estimate("Hi", 1.0)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown inference profile"):
            load_inference_settings().profile("one-step-noise")

    def test_unknown_cache_mode_is_rejected(self):
        fast = load_inference_settings().profile("fast")

        with self.assertRaisesRegex(ValueError, "Unknown cache mode"):
            fast.with_overrides(cache_mode="mystery")

    def test_profiles_reject_nonfinite_and_ill_typed_numerical_options(self):
        fast = load_inference_settings().profile("fast")

        for field, value in (
            ("num_steps", True),
            ("num_steps", 1.5),
            ("cfg_scale", float("nan")),
            ("cfg_scale_text", float("inf")),
            ("cfg_scale_speaker", -1.0),
            ("cfg_min_t", float("nan")),
            ("initial_noise_scale", float("inf")),
            ("temporal_rescale_k", 0.0),
            ("temporal_rescale_sigma", float("nan")),
            ("target_latent_std", float("nan")),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                fast.with_overrides(**{field: value})


if __name__ == "__main__":
    unittest.main()
