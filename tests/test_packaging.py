"""Tests for declarative project packaging metadata."""

import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).parents[1]


class PackagingMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_project_metadata_is_declarative(self):
        project = self.metadata["project"]

        self.assertEqual(project["name"], "nar-vae")
        self.assertEqual(project["requires-python"], ">=3.10")
        self.assertEqual(project["dynamic"], ["version"])
        self.assertEqual(
            self.metadata["tool"]["setuptools"]["dynamic"]["version"],
            {"attr": "vyvotts._version.__version__"},
        )
        self.assertFalse((ROOT / "setup.py").exists())
        self.assertEqual(
            project["urls"],
            {
                "Homepage": "https://github.com/kadirnar/nar-vae",
                "Repository": "https://github.com/kadirnar/nar-vae",
                "Issues": "https://github.com/kadirnar/nar-vae/issues",
            },
        )
        self.assertIn("torch>=2.2", project["dependencies"])
        self.assertIn("torchaudio>=2.2", project["dependencies"])

    def test_requirements_remain_compatible_with_project_dependencies(self):
        requirements = {
            line.strip()
            for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        dependencies = set(self.metadata["project"]["dependencies"])
        optional_dependencies = {
            dependency
            for group in self.metadata["project"]["optional-dependencies"].values()
            for dependency in group
        }

        self.assertTrue(requirements.issubset(dependencies | optional_dependencies))
        self.assertIn("tomli>=2.0; python_version < '3.11'", dependencies)

    def test_package_does_not_install_commands(self):
        self.assertNotIn("scripts", self.metadata["project"])

    def test_cache_dit_is_an_optional_turbo_dependency(self):
        project = self.metadata["project"]

        self.assertFalse(
            any(dependency.startswith("cache-dit") for dependency in project["dependencies"])
        )
        self.assertEqual(project["optional-dependencies"]["turbo"], ["cache-dit>=1.5,<2"])

    def test_wandb_is_optional_from_the_training_stack(self):
        project = self.metadata["project"]

        self.assertNotIn("wandb", project["optional-dependencies"]["train"])
        self.assertEqual(project["optional-dependencies"]["wandb"], ["wandb"])

    def test_wheel_metadata_has_no_direct_url_dependencies(self):
        project = self.metadata["project"]
        dependencies = list(project["dependencies"])
        for group in project["optional-dependencies"].values():
            dependencies.extend(group)
        self.assertFalse(any(" @ git+" in dependency for dependency in dependencies))

    def test_yaml_and_toml_configs_are_package_data(self):
        package_data = self.metadata["tool"]["setuptools"]["package-data"]["vyvotts"]

        self.assertIn("configs/*.yaml", package_data)
        self.assertIn("configs/*.toml", package_data)

    def test_downloader_is_a_library_api(self):
        self.assertFalse((ROOT / "hf_down.py").exists())
        self.assertTrue((ROOT / "vyvotts" / "hub.py").is_file())
        self.assertFalse((ROOT / "vyvotts" / "cli").exists())

    def test_canonical_and_compatibility_packages_are_built(self):
        include = self.metadata["tool"]["setuptools"]["packages"]["find"]["include"]

        self.assertIn("nar_vae*", include)
        self.assertIn("vyvotts*", include)
        self.assertTrue((ROOT / "nar_vae" / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
