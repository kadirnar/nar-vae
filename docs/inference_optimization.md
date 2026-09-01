# Diffusion inference optimization

This document defines the supported route to faster inference for the immutable-DACVAE
architecture. It is an implementation and benchmark contract, not a latency or quality claim.
No optimization in this document changes, fine-tunes, replaces, or bypasses DACVAE.

## Supported generation path

New checkpoints use cosine variance-preserving diffusion with v-prediction. In generation
direction, `t=0` is Gaussian noise and `t=1` is clean data:

```text
alpha(t) = sin(pi*t/2)
sigma(t) = cos(pi*t/2)
x_t      = alpha(t) * x_clean + sigma(t) * epsilon
v_t      = sigma(t) * x_clean - alpha(t) * epsilon
```

The deterministic DDIM step reconstructs the model's clean and noise estimates and advances them
to the next schedule point:

```text
x_clean_hat = alpha(t) * x_t + sigma(t) * v_t
epsilon_hat = sigma(t) * x_t - alpha(t) * v_t
x_next      = alpha(t_next) * x_clean_hat + sigma(t_next) * epsilon_hat
```

This analytic update is the canonical fast sampler. The ODE solvers remain useful as numerical
references and for legacy rectified-flow checkpoints. Objective, schedule, and solver are bound by
checkpoint metadata; the runtime rejects attempts to decode a flow checkpoint as VP diffusion or
to use DDIM with a flow checkpoint.

For current schema-5 VP checkpoints, the shipped profile names describe fixed compute budgets
only:

| Profile | Sampler | Neural evaluations | Intended use |
| --- | --- | ---: | --- |
| `quality` | deterministic DDIM | 32 | Initial comparison baseline |
| `balanced` | deterministic DDIM | 16 | Candidate after quality gating |
| `fast` | deterministic DDIM | 8 | Aggressive candidate after quality gating |

Rectified-flow checkpoints—including authenticated schema-2/EchoDiT-v3 compatibility weights and
current schema-5 flow exports—use objective-compatible ODE profiles instead of borrowing VP/DDIM
settings. The budgets below reproduce the historical schema-2 profiles; for later flow schemas
they are a conservative library policy, not an authenticated training-time calibration:

| Profile | Sampler | Neural evaluations | Cache mode |
| --- | --- | ---: | --- |
| `quality` | Heun | 50 | none |
| `balanced` | Euler | 32 | none |
| `fast` | Euler | 16 | none |
| `turbo` | Euler | 16 | Cache-DiT |

These labels do not establish perceptual quality, real-time factor, or hardware latency. A trained
checkpoint must be evaluated at every advertised budget. For schema-2 weights these profiles are
compatibility behavior, not recommendations for a newly trained VP model.

## Work removed from the diffusion loop

The architecture keeps expensive, step-invariant work outside the repeated DiT evaluations:

- The pinned XPhoneBERT backbone is frozen. Training uses cached token-aligned states; inference
  evaluates it once per request and reuses the result for every DDIM step.
- A small learned adapter projects the 768-dimensional provider states to the acoustic hidden
  size. The external backbone is neither registered in nor saved with the acoustic model.
- Reference audio is encoded once with the unchanged DACVAE and speaker encoder. A learned-query
  resampler compresses the variable-length reference to eight fixed tokens before diffusion.
- The duration/MAS path is evaluated before sampling. Its expanded text state is reused through
  the trajectory.
- Conditional and unconditional classifier-free-guidance branches can share one batched DiT
  evaluation. The neutral default remains `cfg_scale=1`; other values require checkpoint-specific
  evaluation.

Reference and text caches contain sensitive or contract-bound data. A serving cache must be
bounded and scoped to an authorized request/session. Its key must include exact input bytes,
normalization/frontend version, language metadata, provider and checkpoint hashes, preprocessing
parameters, and dtype. A cache must never allow one user's voice reference to reach another user.

## Training-time sequence reduction

Target-latent packing happens outside DACVAE and is part of the acoustic checkpoint topology.
The canonical `P2` setting halves the DiT target time axis while leaving the codec unchanged.
`P1` is the quality control. `P4` is a separately trained aggressive ablation; an inference caller
cannot change the patch factor of existing weights.

The same rule applies to the eight-token reference resampler: its token count is authenticated
checkpoint topology, not an inference knob.

## Runtime optimizations

Apply optimizations in this order and compare each change against the same checkpoint, requests,
sampler budget, and random seeds:

1. Use BF16 on hardware with reliable BF16 tensor-core support. Keep FP32 as the numerical
   reference and record the actual attention backend.
2. Bucket text, reference, and target lengths. Batch only compatible requests with the same
   checkpoint, profile, precision, guidance layout, and padded shapes.
3. Benchmark `torch.compile` after the shape buckets are stable. Record cold compile time,
   recompilations, graph-cache memory, warmed latency, and fallback operations.
4. Fuse conditional/unconditional guidance branches where guidance is enabled. Compare against
   neutral conditioning because guidance increases effective batch and compute.
5. Evaluate reduced precision or deployment engines only as separately versioned artifacts.
   Require language-stratified intelligibility and speaker-similarity parity before promotion.

The runtime does not silently enable TF32, global cuDNN benchmarking, quantization, or process-wide
backend flags. The serving application owns those choices and must record them.

Select BF16 explicitly when constructing the runtime:

```python
from nar_vae.inference import FlowMatchingTTSInference

tts = FlowMatchingTTSInference.from_preset(
    "small",
    flow_model_path="checkpoints/nar_vae_pretrain/pytorch_model.bin",
    dacvae_model="facebook/dacvae-watermarked",
    device="cuda",
    acoustic_dtype="bf16",
)
```

The checkpoint is authenticated and loaded in FP32 on CPU first. Only learned floating acoustic
parameters are then converted before the accelerator transfer. Complex RoPE caches, FP64 objective
metadata, and every other model buffer retain their exact dtype and value. DACVAE is neither cast
nor changed; BF16 generated latents are converted at its FP32 decode boundary. `float32` is the
default and numerical reference. FP16 is deliberately not exposed by this runtime contract.

Cache-DiT/SCM reuse is experimental and is not combined with the canonical analytic DDIM path.
The sampler rejects that combination because cached intermediate states change the solver being
measured. It may be studied separately with compatible legacy Euler/ODE execution, with the exact
cache policy and acoustic error reported.

## Benchmark contract

Every result must name:

- code revision, checkpoint and manifest hashes, immutable codec identity, and frozen-provider
  identity;
- GPU, driver, CUDA and PyTorch versions, precision, attention backend, and compile policy;
- target/reference language pair, text-length distribution, reference duration, and generated
  duration distribution;
- target patch size, reference-summary count, sampler, schedule shift, neural evaluations, and
  guidance values;
- cold and warmed latency, conditioning time, per-step DiT time, DACVAE decode time, transfer time,
  peak memory, throughput, and real-time factor.

Measure synchronized requests and steady arrivals separately. Report p50/p95/p99 rather than one
best run. Do not call complete-waveform latency "time to first audio."

For every advertised target/reference language pair, gate a faster profile on:

- WER or CER with a pinned external recognizer;
- speaker similarity with a pinned external verifier;
- duration and text-coverage errors;
- multilingual/code-switch slices relevant to the advertised capability;
- blinded human naturalness, pronunciation, speaker identity, and artifact judgments.

No profile is promoted solely because its loss, neural-evaluation count, or real-time factor is
lower.

## Distillation and lower-step research

A later diffusion-native distillation stage is reasonable only after the 32-evaluation teacher is
competent. DMOSpeech provides relevant 2025 evidence for diffusion distillation with differentiable
intelligibility and speaker objectives. Teacher and student checkpoints, timestep mapping,
objective, and evaluator versions must be explicit. This repository does not currently claim a
distilled one/few-step model.

Sway Sampling and flow-specific solvers from F5-TTS or Irodori cannot be copied into VP diffusion
by renaming their time variable. Any alternative schedule needs a derivation for the trained VP
parameterization and an equal-evaluation comparison.

## Streaming boundary

The current DiT has global target dependencies and the unchanged DACVAE decoder produces the full
waveform. It therefore returns complete audio, not independently playable causal chunks. True
streaming would require a separately trained local/blockwise acoustic topology and a compatible
causal decoding contract. Because this project requires DACVAE to remain unchanged, that redesign
is out of scope and must not be implied by buffering slices of a completed waveform.

## Primary sources

- [DiTTo-TTS, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/80e77d9ed2f74dcaf1a42cb1a2593559-Paper-Conference.pdf)
- [DMOSpeech, ICML 2025](https://proceedings.mlr.press/v267/li25ay.html)
- [F5-TTS, ACL 2025](https://aclanthology.org/2025.acl-long.313/)
- [StyleTTS-ZS, NAACL 2025](https://aclanthology.org/2025.naacl-long.242/)
- [Echo-TTS](https://jordandarefsky.com/blog/2025/echo/)
- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
