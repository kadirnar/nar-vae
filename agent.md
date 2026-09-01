# NAR-VAE Development Agent

## Mission

Build a readable, modular diffusion text-to-speech system that can be pretrained from scratch on
one or many GPUs, preserves zero-shot speaker identity across languages, and can ultimately serve
at least 50 concurrent streaming users on one named GPU with measured p95 time-to-first-audio
below 100 ms.

This is an evidence target, not a capability claim. Existing weights remain limited to the
capabilities established by their versioned metadata and evaluation.

## Startup Contract

Before every change:

1. Read `AGENTS.md` and this file completely.
2. Inspect Git status and preserve all unrelated tracked and untracked work.
3. Execute the user-authorized scope and document material research decisions under `docs/`.
4. Read the relevant source, tests, configuration, packaging, and documentation before editing.
5. For inference, TTFA, concurrency, or streaming work, read
   `docs/inference_optimization.md` completely.

## Engineering Contract

1. Preserve backward compatibility unless the user explicitly authorizes a versioned migration.
2. Keep optional integrations lazy. Importing `nar_vae` or `vyvotts` must not require CUDA, downloads,
   credentials, training extras, or serving extras.
3. Prefer small modules, typed public boundaries, explicit configuration, and deterministic state.
4. Add network-free, GPU-free regression tests for every behavior change. Hardware tests supplement
   rather than replace these contracts.
5. Keep target-text language and reference-audio language independent across API, datasets,
   collation, training, conditioning, inference, benchmarks, and evaluation.
6. Fail closed on checkpoint capabilities. Never infer multilingual, speaker, cross-lingual,
   duration, codec, quantization, or streaming support from configuration alone.
7. Never commit credentials, datasets, checkpoints, generated audio, or machine-specific paths.

## Inference Truth Contract

- The current global ODE completes before full DACVAE decode and is non-streaming.
- An internal first ODE step is TTFT telemetry, not playable-audio TTFA.
- Post-hoc waveform slicing is transport chunking, not model streaming.
- Current-checkpoint compilation, CUDA graphs, fused CFG, Cache-DiT, precision changes, and dynamic
  batching may improve throughput but cannot create a final latent prefix.
- True streaming requires separately trained blockwise/local generation and a causal codec with
  versioned checkpoint metadata.
- The 50-user, p95 TTFA-below-100-ms target requires synchronized-burst and steady-stream tests on
  a named GPU. Report queueing separately from compute and preserve acoustic quality gates.

## Research Policy

- Research only a concrete gap exposed by the selected goal, test, profile, or evaluation.
- Prefer primary sources: papers, official code, dataset cards, standards, and vendor docs.
- Record useful findings in the relevant architecture, training, or inference document with the
  source, evidence, decision, risk, and testable adoption condition.
- Check license compatibility before using external code. Do not copy unclear or incompatible
  implementations.
- Treat a research idea as unproven until focused tests or measured evaluation support it.

## Priority Order

Unless the user changes the order:

1. Make random-initialized pretraining reproducible, resumable, observable, and correct on one or
   many server GPUs.
2. Train and evaluate compact latent, hierarchical flow, multilingual text, and monotonic-duration
   candidates for small-model intelligibility and acoustic quality.
3. Continue only project-produced pretraining checkpoints through SFT, few-step distillation, and
   optional flow-native GRPO.
4. Measure conditioning reuse, batching, compilation, Cache-DiT, solver, and precision choices on
   the target server GPU without changing quality gates.
5. Train and evaluate blockwise generation plus a causal lower-rate codec before making streaming
   claims, then demonstrate the complete multilingual, speaker, latency, and concurrency gates.

## Definition of Done

A cycle is complete only when its acceptance criteria pass, focused and full verification results
are recorded, user-facing behavior is documented, and remaining risks or hardware gates are
explicit. Commit or push only when the user authorizes publication.
