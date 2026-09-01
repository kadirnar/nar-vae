# Repository Guidelines

## Project Structure & Module Organization

The implementation package lives in `vyvotts/`, with `nar_vae` as the canonical public import
surface and `vyvotts` retained as a compatibility namespace. Top-level modules cover inference, training,
checkpoints, benchmarking, and configuration. Keep specialized code in these subpackages:
`models/`, `solvers/`, `serving/`, `caching/`, `dacvae/`, `dataset/`, and
`losses/`. Runtime YAML and TOML files belong in `vyvotts/configs/`; they are included
in built packages. Tests live in `tests/` and mirror behavior rather than package paths.
Documentation artwork belongs in `docs/assets/`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"` installs the package editable with Ruff and build tools.
- `python -m pip install -e ".[train]"` adds dataset and training dependencies. Training
  requires an NVIDIA CUDA environment.
- `python -m unittest discover -s tests -v` runs the complete test suite.
- `ruff check .` checks imports and configured `E`, `F`, and `I` rules.
- `ruff format --check .` verifies formatting; use `ruff format .` to apply it.
- `python -m build` creates source and wheel distributions.

NAR-VAE has no CLI entry point. Exercise inference and training through the Python API shown
in `README.md`.

## Agent Startup and Scope

Before changing files, read `AGENTS.md` and `agent.md` completely, inspect Git status, and preserve
unrelated work. For latency or streaming work also read `docs/inference_optimization.md`. Follow the
user-authorized scope, keep research decisions in the relevant document under `docs/`, and report
verification and remaining hardware gates. Commit or push only when the user explicitly authorizes
publication.

## Coding Style & Naming Conventions

Target Python 3.10 or newer. Use four-space indentation, double quotes, and a maximum
line length of 100 characters; Ruff configuration in `pyproject.toml` is authoritative.
Use `snake_case` for modules, functions, variables, and configuration keys; use
`PascalCase` for classes and `UPPER_SNAKE_CASE` for constants. Keep optional integrations
lazy so importing `nar_vae` or `vyvotts` does not require training or acceleration extras. Add public
exports deliberately in the relevant `__init__.py`.

## Testing Guidelines

Tests use `unittest`, including mocks for external services and optional dependencies.
Name files `test_<feature>.py`, test cases `Test<Feature>`, and methods `test_<behavior>`.
Add regression coverage for behavior changes. Avoid requiring network access,
Hugging Face credentials, large checkpoints, or a GPU in the default suite. No numeric
coverage threshold is configured; focus on meaningful paths and failure cases.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `Add real Cache-DiT turbo
optimization` and occasional scoped forms such as `docs: simplify README`. Follow that
style and keep commits focused. Pull requests should explain motivation and user-visible
impact, list commands run, and link relevant issues. For performance or audio
quality changes, include the checkpoint, hardware, settings, and before/after measurements.

## Security & Configuration Tips

Pass private Hub credentials through `HF_TOKEN` or `hf auth login`. Never commit tokens,
model checkpoints, datasets, generated audio, or machine-specific paths.
