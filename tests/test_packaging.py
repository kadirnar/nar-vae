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
            {"attr": "nar_vae._version.__version__"},
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
        self.assertIn("torch>=2.9", project["dependencies"])
        self.assertIn("torchaudio>=2.9", project["dependencies"])

    def test_pyproject_is_the_single_dependency_manifest(self):
        dependencies = set(self.metadata["project"]["dependencies"])

        self.assertFalse((ROOT / "requirements.txt").exists())
        self.assertNotIn("optional-dependencies", self.metadata["project"])
        self.assertIn("tomli>=2.0; python_version < '3.11'", dependencies)
        for required in (
            "accelerate>=1.3,<2",
            "build",
            "cache-dit>=1.5,<2",
            "datasets==3.4.1",
            "kernels>=0.16,<0.17",
            "PyYAML",
            "ruff",
            "torchcodec",
            "transformers>=4.49,<5",
            "wandb",
        ):
            with self.subTest(required=required):
                self.assertIn(required, dependencies)

    def test_package_does_not_install_commands(self):
        self.assertNotIn("scripts", self.metadata["project"])

    def test_cache_dit_is_installed_by_the_single_setup(self):
        project = self.metadata["project"]

        self.assertIn("cache-dit>=1.5,<2", project["dependencies"])

    def test_wandb_is_mandatory_in_the_single_setup(self):
        project = self.metadata["project"]

        self.assertIn("wandb", project["dependencies"])

    def test_wheel_metadata_has_no_direct_url_dependencies(self):
        project = self.metadata["project"]
        dependencies = list(project["dependencies"])
        self.assertFalse(any(" @ git+" in dependency for dependency in dependencies))

    def test_yaml_and_toml_configs_are_package_data(self):
        package_data = self.metadata["tool"]["setuptools"]["package-data"]["nar_vae"]

        self.assertIn("configs/*.yaml", package_data)
        self.assertIn("configs/*.toml", package_data)

    def test_downloader_is_a_library_api(self):
        self.assertFalse((ROOT / "hf_down.py").exists())
        self.assertTrue((ROOT / "nar_vae" / "hub.py").is_file())
        self.assertFalse((ROOT / "nar_vae" / "cli").exists())

    def test_only_nar_vae_package_is_built(self):
        include = self.metadata["tool"]["setuptools"]["packages"]["find"]["include"]

        self.assertEqual(include, ["nar_vae*"])
        self.assertTrue((ROOT / "nar_vae" / "__init__.py").is_file())

    def test_repository_has_one_namespace_and_no_agent_instruction_files(self):
        former_word = "vy" + "vo"
        former_package = former_word + "tts"

        self.assertFalse((ROOT / former_package).exists())
        self.assertFalse((ROOT / "AGENTS.md").exists())
        self.assertFalse((ROOT / "agent.md").exists())
        for path in ROOT.rglob("*"):
            if not path.is_file() or {".git", ".ruff_cache", "__pycache__"}.intersection(
                path.parts
            ):
                continue
            if path.suffix.lower() not in {".md", ".py", ".toml", ".txt", ".yaml", ".yml"}:
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(former_word, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
