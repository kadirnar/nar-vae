# Training

Canonical NAR-VAE training learns a small acoustic diffusion model from random initialization. It
does not learn a text Transformer from token IDs. A pinned XPhoneBERT provider is frozen, evaluated
during preparation, and reduced to cached token-aligned states. DACVAE is also frozen and remains
unchanged.

No trained checkpoint is bundled. The configuration is a reproducible starting point; quality,
speaker similarity, language coverage, memory, and training time must be measured on the selected
data and hardware.

## 1. Install

```bash
python -m pip install -e .
wandb login
```

The training entry points currently require W&B. Use `WANDB_MODE=offline` on an isolated server.
Training requires a suitable NVIDIA GPU; dataset preparation may be distributed as well.

## 2. Build the text contract before preparing audio

The canonical frontend consumes the exact phoneme sequence expected by XPhoneBERT. It never
silently runs an unversioned G2P system and never substitutes an unknown phone. Each raw row needs
audio, transcript, language, stable speaker and utterance identities, and reviewed provider-native
phones:

```python
{
    "audio": {"array": waveform_float32, "sampling_rate": 44100},
    "text": "The exact transcript.",
    "phonemes": reviewed_provider_phones,
    "language": "en",
    "speaker_id": "corpus-a/speaker-001",
    "utterance_id": "corpus-a/recording-001",
}
```

XPhoneBERT uses whitespace-separated phone units and the U+2581 `▁` word-boundary unit. Create
the phones with a pinned normalization, segmentation, and G2P pipeline for each language, inspect
its unknown-phone rate, and version that pipeline with the dataset. Do not copy a plausible-looking
IPA example from this documentation and assume it belongs to the provider vocabulary.

The acoustic config pins the provider repository, full commit, artifact hashes, vocabulary/PAD,
hidden layer, output dtype, alignment policy, and cache version. Load that same config for dataset
preparation:

```python
from pathlib import Path

import yaml

model_config = yaml.safe_load(
    Path("nar_vae/configs/echodit_config.yaml").read_text(encoding="utf-8")
)
```

Passing ``frozen_text_config=model_config`` below constructs the provider only after distributed
initialization has selected each process's local CUDA device. The first construction downloads and
hash-verifies the pinned artifacts. The external model is in evaluation mode, has gradients
disabled, and is not inserted into the acoustic model. For a single-process pipeline that reuses an
already loaded provider, ``FrozenTextProvider.from_config(...)`` and ``frozen_text_provider=...``
remain available explicitly.

## 3. Prepare a multilingual cloning dataset

```python
from nar_vae.dataset import prepare_from_hf_dataset

prepare_from_hf_dataset(
    dataset_name="owner/speech-corpus",
    dataset_revision="FULL_40_CHARACTER_COMMIT_SHA",
    split="train",
    output_dir="data/prepared",
    audio_column="audio",
    text_column="text",
    phonemes_column="phonemes",
    language_column="language",
    speaker_id_column="speaker_id",
    utterance_id_column="utterance_id",
    dacvae_model="facebook/dacvae-watermarked",
    dacvae_revision="8680102d141858a21bd533543966a2eb2e569f92",
    dacvae_sha256="573cf4770ea4a25507f26965d05ae720bcd34295a9f60c06ef3c3805826b68e4",
    dacvae_backend="bundled",
    frozen_text_config=model_config,
    device="cuda",
)
```

Every prepared row binds the provider contract and stores IDs, frozen states, attention/alignment
masks, token-language IDs, cache dtype/version, and contract SHA. Training preflight rejects mixed
providers, IDs outside the provider vocabulary, changed cache versions, wrong feature widths or
dtypes, invalid masks, non-finite values, and codec/frontend mismatches.

Representation contract v3 also binds `codec_encoding_policy=posterior_sample_seeded_v1`.
DACVAE's learned posterior remains unchanged and sampled; NAR-VAE derives a call-local seed from
the exact mono float32 waveform after resampling and any reference truncation, together with the
authenticated codec SHA-256. Re-preparing identical content therefore reproduces its latent on the
same device/backend without consuming process-global RNG state. Posterior-mean encoding is not a
compatible shortcut. Contract-v2 datasets must be prepared again rather than mixed into a v3 run.

For code-switching, supply `language_spans_column`; every span must contain provider-native phones
and its own language. A provider token may not cross a language boundary.

For zero-shot voice cloning, use speakers with at least two genuinely different recordings. Keep
training, validation, and test speakers disjoint. Preparation stores each utterance latent once.
`DynamicReferenceDataset` selects another same-speaker utterance during training, verifies distinct
utterance/audio identity, and takes a bounded deterministic crop. `speaker_id` is used only for
pairing and split checks; it is never a model input.

## 4. Configure pretraining

```bash
cp nar_vae/configs/echodit_config.yaml echodit_config.yaml
cp nar_vae/configs/pretrain_config.yaml train.yaml
```

Keep `train.yaml` beside its copied base or update its `extends` path. Edit the dataset, output, and
the exact supported language pairs:

```yaml
TTS_dataset_local: ./data/prepared
save_folder: ./checkpoints/pretrain
model_preset: small

text_conditioning_mode: frozen_features
conditioning_feature_size: 768
freeze_text_encoder: false  # trains the small adapter; the provider stays external/frozen

generative_objective: vp_diffusion_v
diffusion_schedule_shift: 1.0
target_patch_size: 2
speaker_num_summary_tokens: 8

use_speaker_conditioning: true
use_language_conditioning: true
supported_languages: [en, tr]
supported_language_pairs:
  - [en, en]
  - [tr, tr]
  - [tr, en]

use_duration_predictor: true
use_mas_duration: true

report_to: wandb
wandb_project: nar-vae-pretraining
wandb_run_name: small-multilingual-vp
```

Only declare languages and target/reference pairs present in the prepared data. The preflight
coverage check rejects unsupported declarations. `diffusion_schedule_shift: 1.0` is the unshifted
cosine-VP baseline and is stored in every checkpoint; changing it requires new training.

The canonical `small` topology has about 44.85M trainable acoustic parameters. XPhoneBERT and the
immutable codec are reported separately. `P2` target packing and eight reference summaries are
checkpoint architecture, not settings that may be changed later at inference.

## 5. Start training

Save a guarded entry point as `train_job.py`:

```python
from nar_vae.train import pretrain

if __name__ == "__main__":
    pretrain("train.yaml")
```

One GPU:

```bash
python train_job.py
```

DDP, one process per GPU:

```bash
torchrun --standalone --nproc-per-node=8 train_job.py
```

Exact interruption recovery may use a checkpoint from the same run. Pretraining rejects external
TTS weights. The frozen text states are data dependencies, not pretrained acoustic weights.

## Cost controls and required comparisons

| Setting | Canonical choice | Gate |
| --- | --- | --- |
| Model preset | `small` | Compare `nano`, `tiny`, and `small` at equal data/steps. |
| Target packing | `P2` | Keep `P1` as quality control; treat `P4` as aggressive. |
| Reference summaries | 8 | Compare 4/8/16 and the legacy uncompressed path. |
| Precision | BF16 | Keep FP32 numerical checks; use BF16 only on suitable hardware. |
| Batching | Padded-attention cost budget | Tune the budget from measured memory and throughput. |
| Activation checkpointing | Enabled | Trades compute for memory. |
| Optimizer | AdamW | Muon/AdamW and fused AdamW are measured experiments. |
| MAS | Exact batched GPU path | Monitor failed/degenerate alignments by language. |
| Compilation | Off initially | Enable after shapes/buckets are stable and parity-tested. |

At minimum, report GPU-hours, peak memory, examples and audio seconds per second, convergence,
WER/CER per language, speaker similarity per target/reference pair, duration error, repetition and
truncation, and blinded listening results. The `quality`, `balanced`, and `fast` sampler names are
not substitutes for those measurements.

Run release evaluation outside the training loop on a speaker-disjoint, immutable split. The
library's `cross_lingual_quality_report` combines a target-language ASR transcript and embeddings
from a pinned external speaker verifier, but it deliberately has no universal thresholds:

```python
from nar_vae.quality import cross_lingual_quality_report

report = cross_lingual_quality_report(
    reference_text=target_transcript,
    hypothesis_text=pinned_asr_transcript,
    reference_speaker_embedding=reference_embedding,
    synthesized_speaker_embedding=generated_embedding,
    target_language="tr",
    reference_language="en",
    asr_model=pinned_asr_identity,
    speaker_verification_model=pinned_verifier_identity,
    maximum_error_rate=predeclared_pair_wer_gate,
    minimum_speaker_similarity=predeclared_pair_similarity_gate,
)
```

Choose gates before evaluating a checkpoint and report each declared target/reference-language
pair separately. The bundled waveform evaluator is English-only; multilingual release testing
requires language-appropriate, revision-pinned ASR models.

## SFT and post-training

Population-level SFT uses `nar_vae/configs/finetune_config.yaml` and must match the parent objective,
schedule, provider, codec, target packing, reference summaries, language registry, duration, and
MAS topology exactly. It is not single-speaker adaptation and must preserve speaker-disjoint
evaluation.

The implemented GRPO stage is rectified-flow-native and rejects canonical VP checkpoints. A
diffusion-native policy likelihood and replay contract must be derived and tested before GRPO can
be used on this model. Teacher distillation is also deferred until a strong 32-evaluation VP
checkpoint exists.
