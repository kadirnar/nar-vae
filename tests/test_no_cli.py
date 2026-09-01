"""Guard the library-only public surface."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
PACKAGE = ROOT / "vyvotts"


class LibraryOnlySurfaceTest(unittest.TestCase):
    def test_package_contains_no_cli_entry_points(self):
        for path in PACKAGE.rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                imported_modules = {
                    alias.name
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import)
                    for alias in node.names
                }
                imported_modules.update(
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module is not None
                )
                function_names = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                main_guards = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test)
                ]

                self.assertNotIn("argparse", imported_modules)
                self.assertNotIn("build_parser", function_names)
                self.assertNotIn("main", function_names)
                self.assertFalse(main_guards)

    def test_workflows_have_python_api_replacements(self):
        expected_functions = {
            "benchmark.py": "run_benchmark",
            "benchmark_solvers.py": "compare_solvers",
            "train.py": "train",
            "finetune.py": "finetune",
            "dataset/finetune_prepare.py": "prepare_finetune_dataset",
        }

        for relative_path, function_name in expected_functions.items():
            with self.subTest(path=relative_path, function=function_name):
                path = PACKAGE / relative_path
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                function_names = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                self.assertIn(function_name, function_names)

        train_source = (PACKAGE / "train.py").read_text(encoding="utf-8")
        train_functions = {
            node.name
            for node in ast.parse(train_source).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn("pretrain", train_functions)


if __name__ == "__main__":
    unittest.main()
