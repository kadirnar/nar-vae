"""The top-level package must remain lightweight."""

import subprocess
import sys
import unittest


class LazyImportTest(unittest.TestCase):
    def test_canonical_import_is_lazy_and_versioned(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, nar_vae; "
                    "assert nar_vae.__version__; "
                    "assert 'vyvotts.inference' not in sys.modules; "
                    "assert 'vyvotts.dataset.prepare' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_does_not_eagerly_load_inference_or_dataset(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, vyvotts; "
                    "assert 'vyvotts.inference' not in sys.modules; "
                    "assert 'vyvotts.dataset.prepare' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_canonical_submodules_alias_the_compatibility_module_graph(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib, importlib.resources; "
                    "canonical = importlib.import_module('nar_vae.configuration'); "
                    "compatibility = importlib.import_module('vyvotts.configuration'); "
                    "assert canonical is compatibility; "
                    "assert canonical.GenerationConfig is compatibility.GenerationConfig; "
                    "assert canonical.__name__ == 'vyvotts.configuration'; "
                    "assert canonical.__spec__.name == 'vyvotts.configuration'; "
                    "assert importlib.reload(compatibility) is canonical; "
                    "canonical_cache = importlib.import_module('nar_vae.caching.cache_dit'); "
                    "compatibility_cache = importlib.import_module('vyvotts.caching.cache_dit'); "
                    "assert canonical_cache is compatibility_cache; "
                    "assert canonical_cache._SESSION_LOCK is compatibility_cache._SESSION_LOCK; "
                    "assert canonical_cache.CacheDiTUnavailableError is "
                    "compatibility_cache.CacheDiTUnavailableError; "
                    "canonical_configs = importlib.import_module('nar_vae.configs'); "
                    "compatibility_configs = importlib.import_module('vyvotts.configs'); "
                    "assert canonical_configs is compatibility_configs; "
                    "assert canonical_configs.__spec__.name == 'vyvotts.configs'; "
                    "assert importlib.resources.files('nar_vae.configs')"
                    ".joinpath('inference.toml').is_file(); "
                    "assert importlib.resources.files('vyvotts.configs')"
                    ".joinpath('inference.toml').is_file()"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
