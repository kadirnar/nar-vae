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
                    "assert 'nar_vae.inference' not in sys.modules; "
                    "assert 'nar_vae.dataset.prepare' not in sys.modules"
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
                    "import sys, nar_vae; "
                    "assert 'nar_vae.inference' not in sys.modules; "
                    "assert 'nar_vae.dataset.prepare' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_submodules_reload_and_expose_packaged_resources(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib, importlib.resources; "
                    "configuration = importlib.import_module('nar_vae.configuration'); "
                    "assert configuration.__name__ == 'nar_vae.configuration'; "
                    "assert configuration.__spec__.name == 'nar_vae.configuration'; "
                    "assert importlib.reload(configuration) is configuration; "
                    "cache = importlib.import_module('nar_vae.caching.cache_dit'); "
                    "assert cache.__name__ == 'nar_vae.caching.cache_dit'; "
                    "assert cache.__spec__.name == 'nar_vae.caching.cache_dit'; "
                    "assert importlib.reload(cache) is cache; "
                    "configs = importlib.import_module('nar_vae.configs'); "
                    "assert configs.__spec__.name == 'nar_vae.configs'; "
                    "assert importlib.resources.files('nar_vae.configs')"
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
