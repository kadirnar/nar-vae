# NAR-VAE Post-Pretraining for Intelligibility and Zero-Shot Voice Cloning

This plan improves the shared NAR-VAE model after from-scratch broad pretraining. It does **not** teach a
requested voice through speaker-specific fine-tuning. Training, validation, preference data, and
checkpoint selection operate on a population of speakers. Every release evaluation speaker is
unseen during every training stage and supplies only reference audio at inference time.

The plan is an experiment protocol, not a quality claim. A checkpoint becomes multilingual,
speaker-conditioned, or cross-lingual only after it has learned the corresponding data and saved
the strict capability metadata described in the README.

## Non-negotiable zero-shot boundary

- No release-evaluation speaker may appear in pretraining, post-training, replay, synthetic prompt,
  preference, or calibration data.
- Do not create a speaker-specific adapter, embedding table entry, LoRA, prompt inversion, gradient
  update, or cached latent optimized for an evaluation or user voice.
- A target utterance may never be its own reference. Select a different utterance from the same
  speaker. When the data permits, vary microphones and acoustic environments to test channel
  robustness without making either one a required grouping field.
- Speaker IDs are data-management keys used for split validation and same-speaker reference
  pairing. They are not model inputs or inference-time identities.
- Target text language and reference-audio language remain independent. Cross-lingual examples use
  a target utterance in one language and a different reference utterance from the same speaker in
  another language.

These rules target zero-shot generalization. [VALL-E X](https://arxiv.org/abs/2303.03926) explicitly
separates a source-language acoustic prompt from target-language text, while
[YourTTS](https://arxiv.org/abs/2112.02418) and [XTTS](https://arxiv.org/abs/2406.04904) show that
multilingual, multi-speaker training can support zero-shot transfer. They do not imply that NAR-VAE
weights already have that capability.

## Dataset contract

Keep a raw, auditable manifest before DACVAE encoding. At minimum, each row should contain:

| Field | Requirement |
| --- | --- |
| `utterance_id` | Stable and globally unique; use it for duplicate and contamination checks. |
| `audio` | Mono-decodable audio; retain the original file hash and source URI outside model input. |
| `text` | Human transcript or a reviewed transcript with recorded provenance and confidence. |
| `language` | Explicit canonical language code; never infer it silently from text. |
| `speaker_id` | Stable within the licensed source; required for split and reference safety. |
| `license` / `consent` | Machine-checkable usage, redistribution, and voice-model permission. |
| `duration` / quality fields | Duration, clipping, speech ratio, SNR estimate, and review state. |

Accent/region, speaking style, age band, or gender may be useful for audit and balanced sampling,
but only when collection and use are lawful, consented, and documented. They must not be guessed
and treated as ground truth.

### Split and pairing rules

1. Split by speaker first, then build utterance pairs. Train, validation, and test speaker sets must
   be disjoint. Deduplicate audio and near-duplicate transcripts across all splits before encoding.
2. Give every included speaker at least two usable utterances and always select a reference
   utterance different from the target. Evaluate channel variation separately when the source
   metadata supports it.
3. Build same-speaker/different-utterance pairs. Vary reference duration and choose one or several
   references within the configured duration bound; never concatenate the target audio itself.
4. Preserve both roles: `language` is the target transcript/audio language and `speaker_language`
   is the selected reference-audio language. Do not copy one field over the other.
5. Use genuinely multilingual speakers for supervised cross-lingual pairs. Monolingual speakers
   still improve same-language cloning, but cannot provide ground-truth recordings of their voice
   speaking an unrecorded language.
6. Keep an unseen-speaker test matrix covering same-language, cross-lingual, code-switched, short
   reference, long reference, clean reference, and realistic noisy-reference conditions.

Validate the split before expensive DACVAE preparation:

```python
from datasets import load_dataset

from nar_vae.dataset import validate_zero_shot_splits

DATASET_REVISION = "0123456789abcdef0123456789abcdef01234567"  # Replace with the real commit.
splits = {
    name: load_dataset(
        "owner/multilingual-speech",
        revision=DATASET_REVISION,
        split=name,
    )
    for name in ("train", "validation", "test")
}
summary = validate_zero_shot_splits(
    splits,
    speaker_id_column="speaker_id",
    utterance_id_column="utterance_id",
    language_column="language",
)
print(summary)
```

Preparation applies the same-speaker/different-utterance reference rule:

```python
from nar_vae.dataset import prepare_finetune_dataset

DATASET_REVISION = "0123456789abcdef0123456789abcdef01234567"  # Replace with the real commit.

prepare_finetune_dataset(
    "owner/multilingual-speech",
    dataset_revision=DATASET_REVISION,
    output_dir="posttrain_data",
    split="train",
    dacvae_model="facebook/dacvae-watermarked",
    dacvae_revision="8680102d141858a21bd533543966a2eb2e569f92",
    dacvae_sha256="573cf4770ea4a25507f26965d05ae720bcd34295a9f60c06ef3c3805826b68e4",
    dacvae_backend="bundled",
    speaker_id_column="speaker_id",
    utterance_id_column="utterance_id",  # Omit to derive a content-bound ID.
    language_column="language",
    max_reference_seconds=12.0,
    reference_seed=1234,
)
```

Dataset download, license review, quality filtering, and split validation happen outside the
default test suite. The validator itself is deterministic and network-free.

### Quality filtering and sampling

Reject empty, clipped, unintelligible, severely truncated, mislabeled, duplicate, unlicensed, or
transcript-mismatched material. Keep useful but harder clean speech—names, numbers, abbreviations,
punctuation, code switching, and long-form sentences—in labeled buckets instead of allowing easy
short sentences to dominate.

If speaker/language balancing is needed, perform it while building the external dataset or sampler;
the packaged trainer only frame-buckets prepared rows. Track hours, speakers, utterances,
transcript-review rate, and target/reference pair counts for every language pair. Choose sampling
temperatures and stage mixtures only from measured ablation results.

## Training stages after broad pretraining

Each stage starts from a versioned NAR-VAE checkpoint produced by the project's own pretraining
stage, uses a fixed data snapshot and seed, and is evaluated against the same unseen-speaker matrix.
No third-party TTS checkpoint initializes these stages. The existing SFT machinery may execute
these runs, but the operation is population-level model continuation—not per-voice fine-tuning.

### Stage 0 — Freeze the baseline

Record the checkpoint hash, model preset, DACVAE identity and encoding mode, tokenizer/language
registry versions, data snapshot, random seed, optimizer state, and evaluator revisions. Generate
the fixed test matrix before changing training. This establishes per-language-pair WER/CER,
speaker similarity, duration/truncation, and human-listening baselines.

### Stage 1 — High-quality supervised continuation

Continue flow-matching and learned-duration training on the reviewed multilingual, multi-speaker
subset. Retain representative broad-pretraining replay to detect and limit catastrophic forgetting.
Balance language and speaker sampling, keep text/language/speaker CFG dropout active, and update the
shared model rather than installing identities for individual speakers.

The expected benefit is cleaner pronunciation and acoustics from better targets. It must be
verified: codec reconstruction quality or lower training loss alone is not a release criterion.

### Stage 2 — Reference-robust zero-shot cloning

Train on same-speaker/different-utterance references. Randomize reference length within deployment
bounds and include realistic channel variation on the reference branch while keeping target
recordings clean enough to remain trustworthy acoustic targets. Mix same-language and true
cross-lingual pairs from multilingual speakers. Continue learned target-language conditioning and
track exact target/reference language-pair metadata throughout the batch; reference language is a
validated capability label, not a separate model embedding.

This stage targets speaker preservation without memorizing an evaluation voice. Report speaker
similarity separately for same-language and cross-lingual pairs and stratify it by reference length,
reference channel, noise condition, and target language.

### Stage 3 — Intelligibility and hard-text continuation

Continue on reviewed hard-text buckets: proper names, numbers, dates, currencies, acronyms,
punctuation, long sentences, rare grapheme sequences, and code switching. Maintain ordinary-text
replay so the model does not specialize only to challenges. Use the normal supervised flow and
duration objectives first.

Use a fixed external ASR model to select checkpoints, not as unquestioned linguistic ground truth.
ASR-guided error rates have been shown to improve TTS checkpoint selection over training loss in a
controlled study ([Baby et al.](https://arxiv.org/abs/2006.01463)), but evaluator errors and language
bias require transcript review and human listening. A differentiable ASR or speaker-consistency
loss is a separate experiment: joint TTS/ASR work reports that optimizing recognizability alone can
hurt speaker characteristics, motivating explicit speaker checks
([Makishima et al.](https://arxiv.org/abs/2207.04659)).

### Stage 4 — Optional prompt-transcript and self-correction experiments

Only after Stages 1–3 have stable baselines, evaluate transcript-free prompt conditioning or
self-correction in an isolated checkpoint version. [X-Voice](https://arxiv.org/abs/2605.05611)
reports a second stage using speaker-consistent synthesized prompt pairs with prompt text masked.
That is evidence for an experiment, not permission to adopt its exact data scale, generate
unconsented voices, or relabel Echo as transcript-free. Synthetic prompts must preserve split
isolation and pass license, identity, and artifact audits.

### Stage 5 — Flow-checkpoint-only GRPO experiment

The implemented GRPO likelihood and rollout equations are rectified-flow-specific. They reject
the canonical `vp_diffusion_v` checkpoints described in this repository; changing a config label
does not make the policy objective diffusion-native. Do not schedule this stage for the new VP
model until its transition density, timestep weighting, replay contract, and tests have been
derived for VP diffusion.

For a separately maintained legacy rectified-flow experiment, start only from a converged,
hash-bound NAR-VAE SFT export whose manifest retains its NAR-VAE
scratch-pretraining parent. Third-party, legacy, pretraining-only, and already-GRPO
weights are not accepted as a fresh reference. Copy the packaged
`nar_vae.post_training.DEFAULT_GRPO_CONFIG_PATH`, point `parent_checkpoint` at the SFT
`final/flow_model/pytorch_model.bin`, and call `grpo_post_train(config_path, reward=...)` from a
guarded Python module. Launch that same module directly for one GPU or with `torchrun` for DDP; the
library intentionally installs no console command.

Bind the callback with `bind_reward_evaluator_manifest`. Its component names must exactly match
`reward_weights`, while every configured evaluator records an implementation name, immutable
revision, and artifact SHA-256. Use language-appropriate WER/CER, speaker-verification, perceptual,
silence, clipping, truncation, repetition, and duration signals. The library does not download
these expensive evaluators. W&B logging is mandatory and rank zero reports globally reduced
metrics; isolated servers may use W&B offline mode.

Each row is a uniquely identified prompt group and stays on one rank. The stage creates an
independent frozen SFT reference, disables CFG for RL rollouts, reuses one detached rollout for at
least two policy epochs, and holds old/reference likelihoods fixed so PPO ratios and clipping are
meaningful. MAS checkpoints use a fixed SFT-reference token allocation throughout rollout,
reference scoring, repeated policy updates, and supervised replay. Variable true lengths may share
a padded batch: padded tail durations are masked from likelihood/KL, decoding happens separately at
each exact latent length, and evaluators receive only the exact waveform prefix.

Each sealed `checkpoint-N` atomically records the policy, frozen reference, optimizer, scheduler,
all rank-local model/rollout/loader RNG streams, trainer cursor, exact SFT weight selection,
reward/config identity, and hash-bound dataset/reference manifests. Use
`resume_from_checkpoint: true` for the latest valid checkpoint from that immutable run, or select
one explicit sibling `checkpoint-N`. Automatic recovery preserves any corrupt newer seal under a
hidden `.rejected-checkpoint-N.*` name before publishing its replayed replacement; explicit
selection remains fail closed. A GRPO final export has a manifest accepted by the normal
manifest-required inference path and retains the selected SFT filename/SHA (plus the full base
weight for a sparse EMA reference).

Collect pairwise ratings for intelligibility, naturalness, speaker identity, pronunciation, and
artifacts, with language-matched listeners where possible. Do not promote a model on its combined
training reward alone. Agree on metric thresholds, rater protocol, and allowed speaker-similarity
versus intelligibility trade-offs before the run.

## Evaluation and promotion gates

For each target/reference language pair, report:

- WER for supported spaced-word scripts and CER for the current CJK scripts, with the fixed ASR
  model and text normalization version;
- cosine similarity from a fixed external speaker-verification model, never the model's own
  conditioning latents;
- duration error, truncation rate, silence/clipping/artifact rates, and failure count;
- naturalness, intelligibility, and speaker-match listening results with the rater protocol;
- named numerical generation profile, ODE solver/steps, CFG settings, seed, GPU, precision, and
  checkpoint hash.

Use `nar_vae.quality.cross_lingual_quality_report` for the objective record. Promote only a
Pareto-safe candidate: it must not regress the agreed intelligibility metric or speaker similarity
against the frozen baseline, and it must pass the language-pair and human-listening gates selected
before the run. If one dimension improves while the other regresses, stop for human direction; do
not hide the trade-off inside an average score.

Record the evaluator implementation version and exact weight hash where available. A WER/CER value
without a threshold selected before evaluation is a measurement, not a pass. Compare multiple
numerical compute budgets; profile names such as `quality` or `fast` are compatibility labels and
do not establish either property. Training quality does not establish the 50-user/sub-100 ms
serving goal; that remains a separate named-GPU streaming and load benchmark.

## Experiment record

Every run should record:

- parent and candidate checkpoint hashes and exact capability metadata;
- immutable manifest/data snapshot and rejected-row counts by reason;
- speaker overlap, duplicate, and target/reference-utterance separation results;
- hours, speakers, and utterances by target language and target/reference language pair;
- optimizer, learning-rate schedule, batch size, effective global batch, GPUs, precision, and seed;
- all objective and listening metrics by language pair and challenge bucket;
- the promotion decision, evidence, regressions, risks, and unresolved questions.

Until a trained checkpoint passes this protocol, the repository provides post-training and
evaluation infrastructure only—not demonstrated multilingual or zero-shot cloning quality.
