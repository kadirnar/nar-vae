# Architecture

The current NAR-VAE architecture is a non-autoregressive conditional latent-diffusion TTS model.
New checkpoints train a cosine variance-preserving process with v-prediction. A separately supplied DACVAE maps audio to
and from continuous latents and remains frozen and unchanged during preparation, pretraining, SFT,
GRPO compatibility checks, and inference.

No architecture guarantees high quality by construction. All quality and speed claims require a
trained export plus held-out evaluation.

## End-to-end data flow

```text
                                       +-- duration / MAS -------------------+
                                       |                                     |
phones -> frozen XPhoneBERT -> adapter +-- text K/V ------------------------+|
                                                                             vv
noise / noisy DACVAE target ----------------------------------------------> EchoDiT -> latent
                                                                             ^
reference audio -> unchanged DACVAE -> bidirectional reference encoder       |
                                      -> learned-query resampler ------------+

generated latent -> unchanged DACVAE decoder -> waveform
```

The heavyweight text backbone and DACVAE are dependencies, not acoustic checkpoint parameters.
Only the feature adapter, reference path, alignment/duration modules, language state, and DiT are
optimized in canonical pretraining.

## Strict VP diffusion

The library uses generation-direction time: `t=0` is noise and `t=1` is clean data. For clean
latent `x`, noise `epsilon`, and `phi=pi*t/2`:

```text
alpha = sin(phi)
sigma = cos(phi)
x_t = alpha * x + sigma * epsilon
v = sigma * x - alpha * epsilon
```

This is a variance-preserving path because `alpha^2 + sigma^2 = 1`. The optional positive
schedule shift modifies log-SNR and renormalizes both coefficients. Generic Euler, midpoint,
Heun, and RK4 sampling convert predicted v to the probability-flow derivative with the exact
shift-dependent chain factor. Deterministic DDIM instead reconstructs clean/noise and advances
analytically between schedule points.

DDIM rejects options that would silently change the trained prior or terminal distribution:
non-unit initial noise scale, post-hoc target standard deviation, temporal score rescaling, and
unvalidated step-cache extrapolation. The 32/16/8 profiles are evaluation candidates, not quality
guarantees.

The checkpoint stores the objective code and schedule shift atomically. Absence of objective
metadata has one legacy meaning: the old straight rectified-flow path. A VP checkpoint missing its
shift, or a flow checkpoint carrying diffusion metadata, is rejected.

## Frozen multilingual text states

Canonical training does not learn a text Transformer from token IDs. A pinned XPhoneBERT provider
runs outside the acoustic model and returns, on one shared token axis:

- provider token IDs;
- contextual hidden states;
- attention and MAS-alignment masks;
- per-token language IDs;
- cache version and full provider-contract digest.

The acoustic checkpoint contains only a `768 -> text_model_size` adapter plus optional learned
language embeddings. Input features are detached at the adapter boundary. The provider is absent
from the module tree, optimizer, DDP state, and acoustic checkpoint.

Canonical `phonemes` mode follows XPhoneBERT's native word-segmented phoneme input. Every language
needs reviewed phones or a separately pinned and evaluated normalization/G2P pipeline. Provider
unknown tokens, invented style/control phones, changed artifact bytes, and cache-contract mismatch
fail closed. Code-switched spans may carry independent language IDs only when tokenizer offsets
preserve their boundaries.

The legacy compact token frontend and scratch text Transformer remain available only for legacy
checkpoint compatibility. They are not the canonical new-training topology.

## Reference-audio voice cloning

Prepared storage keeps one target DACVAE latent per utterance. During training, the dynamic
reference wrapper chooses another recording with the same namespaced speaker ID and a different
utterance ID/audio hash. It crops a deterministic patch-aligned reference, normally 3–12 seconds,
and never passes the speaker ID into the model.

The reference encoder:

1. patches codec latents in time;
2. applies bidirectional self-attention;
3. includes a learned global timbre state;
4. uses eight learned cross-attention queries to resample arbitrary reference length to a fixed
   speaker context.

Eight is checkpointed architecture, not an inference knob. Null speaker CFG uses the same fixed
token shape and an explicit mask, so conditional/unconditional branches cannot drift in topology.
Set the count to zero only to construct a legacy checkpoint with its original uncompressed state.

The bottleneck bounds DiT cross-attention cost and discourages direct transcript copying, but it
does not prove content leakage is gone. Release evaluation must include same/different transcript,
short/long, same/cross-language, shuffled-reference, and multi-reference probes.

## Duration and monotonic alignment

Canonical pretraining jointly learns duration with exact monotonic alignment search (MAS):

1. a text prior predicts diagonal-Gaussian statistics for alignable text states;
2. a fixed orthonormal projection maps detached clean DACVAE frames to the alignment space;
3. batched dynamic programming finds a maximum-likelihood monotonic path;
4. the hard path supplies per-token duration targets and exact frame-level text expansion;
5. a speaker-aware predictor learns token contributions and total frame count.

Only pronunciation-bearing provider tokens own acoustic frames. BOS/EOS, padding, boundaries,
punctuation, and non-alignable marks remain visible to contextual attention but receive zero
duration. Training and inference use the cached frozen state for duration, MAS, DiT text K/V, and
CFG; the provider is not rerun inside the diffusion trajectory.

## EchoDiT

EchoDiT operates on packed continuous DACVAE targets and combines:

- latent self-attention and joint text/reference K/V attention;
- RMSNorm, RoPE, QK normalization, gated attention/MLP outputs, and low-rank AdaLN-Zero;
- global target-language conditioning in the timestep/AdaLN path;
- per-token language state in the text adapter;
- MAS-expanded frame text added to the latent stream;
- learned null text and fixed-shape null speaker states for CFG;
- exact target packing/unpacking stored as `target_patch_size`.

`P2` is the canonical efficiency candidate and shortens the target attention sequence by two.
`P1` is the quality control; `P4` must remain an explicit ablation until WER, speaker similarity,
and listening tests justify it. Packing happens after codec encoding and never modifies DACVAE.

## Parameter and compute accounting

For the canonical frozen-text, multilingual, voice-conditioned, duration/MAS topology with eight
speaker summaries, trainable counts range from 3.60M (`nano`) to 556.22M (`xlarge`); `small` is
44.85M. Counts exclude:

- the frozen 88M-class XPhoneBERT, used offline during preparation and once per inference request;
- the immutable DACVAE codec;
- optimizer state and activation memory.

These categories must be reported separately. Saying "44.85M" does not mean the entire inference
process has only 44.85M resident parameters.

Cost-reduction paths that preserve the declared objective include cached text states, target P2
packing, fixed speaker summaries, BF16, non-reentrant activation checkpointing, deterministic cost
buckets, exact global valid-element normalization, persistent workers, DDP, and one invariant
conditioning encode per trajectory.

Optional experiments include fused AdamW, Muon, compilation, TF32, alternate attention kernels,
larger target patches, distillation, and calibrated branch/temporal skipping. Each needs numerical
and quality parity tests on the target hardware/checkpoint.

## Manifests and compatibility

An exported model manifest binds:

- all tensor-shaping architecture, including target patch and speaker-summary count;
- VP/flow objective and diffusion schedule;
- frozen text provider revisions, filenames, hashes, feature layer/dtype, alignment and cache
  versions;
- language and target/reference-pair capability;
- duration and MAS topology;
- DACVAE source, revision, filename, byte hash, backend, sample rate, hop length, latent width, and
  the content-seeded sampled-posterior encoding policy;
- stage lineage and exact weight hashes.

Checkpoint inspectors validate topology before constructing a model. Inference authenticates the
manifest and weights before deserialization when using the built-in loader. Missing or conflicting
new metadata is not inferred from tensor shapes when doing so would be ambiguous.

Current exports use model-manifest schema 5 and prepared-representation contract 3. Authenticated
schema-2 manifests load only through the exact origin compatibility lane: EchoDiT architecture v3,
the original `cl100k_base` token/control-ID envelope, and the historical global-RNG DACVAE
posterior call for reference audio. Schema-3 and schema-4 manifests also remain readable. None of
these compatibility manifests is mutated or supplemented in its raw representation mapping, so
its canonical hash stays unchanged.

Schemas 2–4 are inference inputs only. Current training and export refuse to relabel their unbound
posterior sampling as the schema-5 content-seeded policy. This exception preserves old checkpoint
behavior; new schema-5 preparation and inference continue to use call-local, content-seeded
posterior sampling.

## Training stages

1. **Pretraining** randomly initializes the acoustic adapter, speaker/resampler, language,
   duration/MAS, and DiT parameters. It consumes frozen cached text states and accepts no external
   TTS checkpoint.
2. **SFT** continues one manifest-bound NAR-VAE pretraining export with identical objective,
   provider, codec, and tensor topology unless an explicitly supported versioned expansion path is
   selected.
3. **GRPO** currently implements rectified-flow SDE ratios only and therefore rejects VP
   checkpoints. Its accepted parent is a current schema-5 rectified-flow NAR-VAE SFT export;
   schemas 2–4 are inference-only and cannot enter training or export. A diffusion-native policy
   objective must be derived and tested before canonical VP models can use that stage.

## Evaluation gates

Every trained preset and language pair should report WER/CER, insertion/deletion/substitution
rates, seen/unseen speaker similarity, cross-language similarity, duration error, repetition,
truncation, silence and clipping rates, perceptual metrics, and blinded listening comparisons.
Also report data hours, speaker/language balance, GPU and precision, trainable/resident parameter
counts, GPU-hours, peak memory, throughput, real-time factor, and latency percentiles.

See [research_2025_2026.md](research_2025_2026.md) for the evidence and ablation plan.
