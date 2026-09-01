# NAR-VAE Architecture and Training Plan

NAR-VAE is a non-autoregressive diffusion and flow-matching TTS library. Its model-training path
starts from random initialization. It must not import a third-party TTS checkpoint, silently adapt
the legacy Echo checkpoint, or claim multilingual, speaker, duration, or streaming capability
before the corresponding weights and evaluation evidence exist.

This document separates implemented software contracts from research that requires server-side
training. No architecture choice by itself guarantees natural speech or low word-error rate.

## Training stages

1. **Codec pretraining** trains the first-party waveform VAE from raw audio when a new codec is
   required. Until that pipeline is implemented, prepared latent datasets must record the exact
   codec identity and are not evidence of end-to-end from-scratch training.
2. **Acoustic pretraining** initializes every NAR-VAE text, duration, speaker, language, and flow
   parameter randomly and optimizes paired text/audio latents. This is the required first model
   stage and never accepts a pretrained TTS checkpoint.
3. **SFT** continues only a NAR-VAE checkpoint produced by the pretraining stage. Full-parameter
   SFT is the reference path; parameter-efficient adapters are optional experiments.
4. **Future few-step distillation** may use only the project's own converged teacher. Quality,
   balanced, and turbo students would be different trained checkpoints, not runtime switches that
   invent quality; this repository does not yet provide a distillation trainer.
5. **Preference/RL post-training** is optional and the implemented GRPO entry point follows only a
   converged, hash-bound SFT export descended from this project's scratch pretraining stage.
   Flow-native GRPO needs stochastic trajectories, a frozen reference, KL/flow-loss anchoring,
   supervised replay, and independent evaluation. It is not a conventional supervised loss.

Resuming an interrupted pretraining or SFT run is fail closed. Rank zero atomically creates an
immutable `run_manifest.json`, then synchronizes its run UUID, stage, library, and normalized
configuration identity across ranks. Dataset preparation writes
`nar_vae_dataset_manifest.json`, an exact SHA-256 inventory of the materialized prepared dataset;
it deliberately does not crawl the raw-audio source tree. A local training run verifies that
inventory, row metadata, and the persisted Dataset state fingerprint before run creation. A remote
run requires a full Hub commit SHA, rejects executable dataset builders, and verifies that same
byte manifest in the commit-contained prepared snapshot. That
resolved data identity is part of both the normalized configuration hash and run manifest. Only
`resume_from_checkpoint` is excluded from the configuration hash. Each completed `checkpoint-N`
seals the full Trainer/flow artifact set in `checkpoint_manifest.json`; `true` scans newest first
and selects the latest completely validated checkpoint, skipping incomplete or invalid candidates
from that immutable run. An explicitly named `checkpoint-N` remains fail closed. Resume
re-resolves the data identity first, so modified local artifacts or a different Hub identity cannot
reach Trainer checkpoint loading. SFT additionally requires its original pretraining lineage and,
when configured, both resumable and export EMA artifacts. Resume is distinct from initializing
pretraining with another model.

## Target architecture

### Compact latent space

The current bundled DACVAE produces 128-dimensional latents at about 86 frames per second. That
makes global attention and every ODE evaluation expensive. Server experiments should compare
continuous waveform-VAE latents at 32 and 64 dimensions and approximately 12 and 24 frames per
second. Codec reconstruction alone is not the selection metric: every candidate must be evaluated
through downstream TTS WER/CER, speaker similarity, naturalness, and artifact rates.

[SimpleSpeech 2](https://arxiv.org/abs/2408.13893) and
[LongCat-AudioDiT](https://arxiv.org/abs/2603.29339) motivate compact-latent ablations, but their
chosen dimensions are hypotheses rather than universal constants.

### Small and base flow models

The intended small zero-shot model is a hierarchical one-dimensional OT-CFM transformer rather
than a flat, full-resolution DiT. The training experiments should combine:

- multiscale temporal stages or temporal patching;
- local/depthwise convolution before global attention;
- text cross-attention plus temporal reference-prompt tokens;
- rotary position embeddings and QK normalization;
- adaLN-Zero conditioning on timestep, target language, and style;
- a 40–80M small target and an approximately 120M base target.

[ZipVoice](https://arxiv.org/abs/2506.13053) provides direct evidence for a hierarchical,
multilingual, few-step speech model, while [P-Flow](https://research.nvidia.com/labs/adlr/projects/pflow/)
supports data-efficient prompt-conditioned flow matching. Results from those systems are not
quality claims for NAR-VAE until reproduced on NAR-VAE data and checkpoints.

For constrained single-speaker deployments, a separate 10–25M convolution/transformer U-Net
preset should be evaluated. [Matcha-TTS](https://arxiv.org/abs/2309.03199) reports an 18.2M
acoustic model with strong low-NFE results, and [LightGrad](https://arxiv.org/abs/2308.16569)
reports a smaller diffusion model. Those experiments do not establish multilingual zero-shot
quality.

### Text and language conditioning

The frontend is versioned with the checkpoint:

- single-language models may use a compact grapheme or phoneme vocabulary;
- multilingual models use normalized multilingual subwords with an optional IPA/phoneme channel;
- target language is encoded both per token and globally;
- reference-audio language remains separate and cannot drive target pronunciation;
- batches are balanced by language, then speaker/domain and approximate frame count.

Evaluation is split by language, speaker, text length, reference language, and seen/unseen pair.
A configuration list cannot create multilingual capability without trained tensors and metadata.

### Artifact and representation binding

Every new acoustic export carries `nar_vae_manifest.json`. It binds all weight files actually used
at inference (including the full base under a sparse EMA overlay), active architecture topology,
trained capability declarations, frontend version, codec source/backend, pinned Hub commit and
filename where applicable, codec byte SHA-256, sample rate, hop length, and latent width. Prepared
rows carry the same representation contract. Fresh SFT preserves it exactly; the only permitted
speaker-topology change is an explicit text-only to speaker-conditioned initialization. Inference
hashes the selected acoustic artifact and any required EMA base before checkpoint deserialization,
then hashes codec bytes before codec deserialization. A same-width replacement codec is therefore
not considered compatible.

### Duration and alignment

There is no universally best duration method. Canonical new scratch pretraining enables an optional,
independently versioned MAS capability for the small and low-resource baseline. A compact text prior
scores a fixed projection of clean target latents; a detached monotonic search supplies hard paths,
while selected likelihoods and positive per-token `log1p` duration predictions remain
differentiable. Stable largest-remainder allocation gives every valid token at least one frame and
exactly matches a requested utterance total. Checkpoint tensors and metadata gate the capability;
SFT must match its parent and configuration alone cannot add it.

This implementation treats every unmasked conditioning token, including boundary tokens, as an
alignable token. Dataset preflight therefore requires at least as many valid codec frames as tokens.
The flow DiT keeps global cross-attention to unexpanded token states and adds a learned projection of
the MAS-expanded text state to each latent frame. Training computes one hard path from clean-latent
likelihoods and reuses it for both the duration objective and velocity conditioning. Inference
allocates predicted token contributions to the exact requested frame total, expands once, and reuses
that prepared frame state across the ODE trajectory and every CFG branch.

Online MAS is correct but sequential over rows and frames, so it is not a fast or fully parallel
training path. The
[official Matcha-TTS extraction guidance](https://github.com/shivammehta25/Matcha-TTS/wiki/Extracting-phoneme-alignments-and-improving-GPU-utilisation)
recommends fixing extracted alignments once they stabilize to improve utilization. Persisted
duration ingestion is not implemented yet and remains a measured follow-up rather than a current
speed claim. Duration evaluation includes insertions, deletions, truncation, long-form stability,
and duration error.

An alignment-free DiT may retain a learned utterance-length distribution for high-data
experiments. Prompt text/audio ratios are only a fallback: [F5-TTS](https://arxiv.org/abs/2410.06885)
documents that its inference duration estimate is heuristic, whereas
[Matcha-TTS](https://arxiv.org/abs/2309.03199) uses monotonic alignment and a learned duration
predictor. Neither the compatibility heuristic nor a learned predictor is allowed to truncate at
the inference ceiling: an over-limit estimate fails so the caller can split or reject the request.
Batch `max_duration` is likewise a validation limit, never a clipping operation.

## Efficient DiT pretraining

The portable training contract supports CPU construction and server-side CUDA execution. The
first implementation priorities are correctness and reproducibility:

- DDP with one process per GPU when the model fits;
- real non-reentrant activation checkpointing around text, speaker, and DiT blocks;
- BF16, FP16 with scaling, and FP32 modes selected explicitly;
- deterministic frame-budget buckets and distributed sampler epochs;
- global valid-frame loss normalization across ranks;
- AdamW as the supported baseline, with fused AdamW only after runtime capability checks;
- optional compilation on stable model regions and length buckets;
- PyTorch SDPA by default, with a pinned, optional
  [Hugging Face Flash Attention 3 kernel](https://huggingface.co/kernels-community/flash-attn3)
  only on compatible CUDA servers and only after numerical parity checks;
- EMA updated after successful optimizer steps, not per microbatch;
- rank-zero-only, optional/offline W&B logging;
- FSDP as a later option after DDP parity, with block-level wrapping and distributed checkpoints.

The current Trainer entry points optimize training batches only. They do not construct a held-out
dataset or decode generated audio during training, so `do_validation: true` is rejected instead of
loading a codec and silently reporting no validation metrics. Server runs must evaluate each
exported checkpoint with explicit, versioned ASR, acoustic, speaker, and listening evaluators.

PyTorch documents the relevant behavior for
[DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html),
[AMP](https://docs.pytorch.org/docs/stable/amp.html), and
[FSDP](https://docs.pytorch.org/docs/stable/fsdp.html). W&B recommends rank-zero-only logging for
the common DDP setup in its [distributed training guide](https://docs.wandb.ai/models/track/log/distributed-training).

Architecture-training ablations include OT pairing, uniform versus logit-normal timesteps,
temporal patch size, local/global attention ratios, and project-trained or self-distilled auxiliary
representation alignment. External pretrained representation encoders are excluded by the
from-scratch contract. The current scratch initializer uses shape-compatible adaLN-Zero gates and
condition projections; strict checkpoint loading overwrites that initialization. Ablations are never
enabled solely because they improved image generation.

## Inference profiles

The packaged profile names are compatibility labels for exact numerical contracts, not evidence of
quality or speed. They do not select or imply a checkpoint. The current uncalibrated starting points
are:

- **quality:** 50 Heun steps (100 neural-function evaluations), uncached;
- **balanced:** 32 Euler steps, uncached;
- **fast:** 16 Euler steps, uncached;
- **turbo:** the same 16 Euler steps with experimental Cache-DiT reuse.

Every packaged profile uses neutral conditional inference (`cfg_scale=1`, joint mode) and no
independent text- or speaker-guidance contribution. Non-neutral guidance, fewer steps, altered
noise, and temporal or latent rescaling require checkpoint-specific acoustic evaluation before
deployment. In particular, a profile label does not turn a teacher into a distilled student.

Solver, schedule, precision, initial-noise scale, guidance behavior, and checkpoint identity are
part of the evaluation key. Distillation uses only a NAR-VAE teacher created by the project's own
pretraining stage. [FlashSpeech](https://arxiv.org/abs/2404.14700) and
[CoMoSpeech](https://arxiv.org/abs/2305.06908) motivate few-step experiments, but do not justify
changing a teacher checkpoint's step count without retraining.

Prepared text and reference conditioning are computed once per request and reused at every ODE
evaluation. Compatible requests may share a tensor batch; incompatible shape, checkpoint,
language, precision, CFG, or solver state must not be mixed.

Cache-DiT remains an optional, fail-closed accelerator for calibrated multi-step models. Cache
state is request-local, conditional and unconditional branches remain separate, and state resets
on every utterance, shape, language, speaker, CFG, precision, solver, or checkpoint change. It is
unlikely to help an already distilled one- to four-step student and it does not create true
streaming. See the [official Cache-DiT repository](https://github.com/vipshop/cache-dit).

True streaming requires a separately trained blockwise/local generator and causal codec. The
current global ODE plus full-utterance decode reports complete-waveform latency only.

## SFT and GRPO

SFT is supervised continuation of a compatible NAR-VAE pretraining checkpoint. It must validate
the architecture, codec, text frontend, language registry, speaker layout, and training stage
before loading weights.

GRPO is experimental because a deterministic flow ODE does not expose a stochastic policy
likelihood. [F5R-TTS](https://arxiv.org/abs/2504.02407) defines a probabilistic flow policy during
training; [FlowTTS-GRPO](https://arxiv.org/abs/2606.23190) instead constructs stochastic
trajectories for an already pretrained deterministic flow. A NAR-VAE implementation must include:

- groups with identical text, reference, target language, and duration constraints;
- stochastic rollouts and trajectory log-probabilities;
- clipped ratios, a frozen reference, KL limits, and flow-loss replay;
- language-appropriate ASR WER/CER, speaker similarity, and perceptual-quality rewards;
- penalties for silence, clipping, truncation, repetition, nonfinite audio, and pathological
  duration;
- independent held-out evaluators and human listening tests to detect reward hacking.

Published flow-TTS GRPO is compute-intensive. It is a post-training option, not a prerequisite for
a low-cost baseline.

## Release gates

No checkpoint is described as high quality, low WER, multilingual, zero-shot, fast, or streaming
until a server run records:

- dataset and license/provenance manifests;
- checkpoint, codec, frontend, and evaluator hashes;
- parameter count, GPU type, precision, batch/frame budget, and training compute;
- held-out WER or CER by language, including insertion/deletion/substitution rates;
- speaker similarity, intelligibility, perceptual metrics, and human A/B results;
- quality-versus-NFE and quality-versus-cache curves;
- peak allocated/reserved memory, throughput, latency percentiles, and failure rates.

The software can make these experiments reproducible; only trained artifacts and measurements can
establish their results.
