# Low-cost diffusion TTS and Turkish voice cloning (2025–2026 review)

## Implemented decision

The first experiment is a Turkish-only, fixed-DACVAE rectified-flow model:

1. revision-pinned `jhu-clsp/mmBERT-small` runs frozen outside the acoustic checkpoint;
2. cached FP16 contextual states feed a small trainable residual acoustic projector;
3. reference audio is encoded by the unchanged DACVAE and a trainable temporal speaker encoder;
4. the `small` EchoDiT uses lossless internal target-latent packing with `P=2`; and
5. an utterance-level, speaker-aware duration head is trained, while hard MAS is disabled for the
   first run.

Use [`turkish_frozen_config.yaml`](../nar_vae/configs/turkish_frozen_config.yaml). The run declares
`supported_languages: [tr]` but does not learn a constant language embedding. Language remains
strict dataset, checkpoint, and request metadata. A later genuinely multilingual experiment uses
[`multilingual_frozen_config.yaml`](../nar_vae/configs/multilingual_frozen_config.yaml), learned
target-language conditioning, and exact observed target/reference pair declarations.

`nar_vae/dacvae/**` was not changed. Dataset latents, masks, duration targets, ODE states, and codec
calls remain native `[batch, 128, frames]` values at 44.1 kHz with hop 512. Packing is undone before
the raw-frame flow loss and decoder boundary.

## What the 2025–2026 systems actually do

- [Echo (2025)](https://jordandarefsky.com/blog/2025/echo/) documents rectified flow, temporal
  transcript-free reference conditioning, joint text/reference K/V, and independent condition
  dropout. Its own text path is not a frozen pretrained provider: UTF-8 bytes pass through an
  in-model learned 256-entry embedding and a 14-layer bidirectional Transformer in the pinned
  [model](https://github.com/jordandare/echo-tts/blob/2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e/model.py)
  and [inference configuration](https://github.com/jordandare/echo-tts/blob/2ed95fce62d33bf7b56f835fd9ec0f0b6fb9155e/inference.py).
  The public repository has no training code, so its optimizer groups cannot be independently
  audited. Echo is architecture evidence for RF/reference conditioning, not frozen multilingual
  text conditioning; its 2.4B scale is also not this low-cost recipe.
- [Irodori-TTS v4 Small (2026)](https://huggingface.co/Aratako/Irodori-TTS-v4-Small) uses a
  fine-tuned Japanese ModernBERT text/caption backbone. The official
  [training configuration](https://github.com/Aratako/Irodori-TTS/blob/main/configs/train_v4_small.yaml)
  assigns that backbone a `1e-5` learning rate versus the main `1e-4` rate. The later
  [v4.1 update](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) freezes the already-trained
  backbone while replacing and retraining the duration predictor. Irodori therefore supports
  low-rate joint adaptation, not permanently frozen-backbone quality parity.
- [DiTTo-TTS (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/80e77d9ed2f74dcaf1a42cb1a2593559-Paper-Conference.pdf)
  is direct 2025 evidence that a frozen pretrained ByT5 encoder can condition diffusion TTS. It is
  evidence for the method class, not proof that frozen mmBERT is quality-equivalent for Turkish or
  that its result transfers unchanged to a fixed DACVAE.
- [mmBERT-small (2025)](https://huggingface.co/jhu-clsp/mmBERT-small) is a 140M, 384-wide
  encoder-only masked-language model trained on text from over 1,800 languages. Its published
  evidence concerns textual classification and retrieval, not Turkish pronunciation, TTS, voice
  cloning, or permanently frozen acoustic conditioning. This repository pins and freezes it as an
  experimental raw-Turkish feature baseline whose CER and pronunciation must be measured.
- [ZipVoice (2025)](https://arxiv.org/abs/2506.13053) reports a distilled four-step system and
  motivates temporal hierarchy and convolutional bypass experiments. Distillation requires a
  trained teacher and another stage, so it is not claimed by the first run.
- [F5-TTS (ACL 2025)](https://aclanthology.org/2025.acl-long.313/) supports the CFM/DiT family, but
  its prompt-infilling interface uses prompt text and is not the transcript-free reference path
  implemented here.
- [X-Voice (2026)](https://arxiv.org/abs/2605.05611) and
  [Cross-Lingual F5-TTS (2025)](https://arxiv.org/abs/2509.14579) reinforce that cross-lingual
  cloning needs explicit training data and evaluation. A language label alone creates no
  cross-lingual ability.
- [SmoothCache for TTS (2025)](https://arxiv.org/abs/2509.08696) motivates conservative
  retraining-free cache experiments. Solver profiles or caching are not trained few-step students.

The packaged default deliberately excludes a phoneme frontend until the Turkish normalizer, G2P
revision, locale, output inventory, and lexicon hashes are represented in the immutable frontend
contract. Pre-2025 model comparisons are outside this design decision.

## Why Turkish single-language mode has no learned language ID

A single constant `tr` embedding contains no information that distinguishes training rows. It can
be absorbed as an arbitrary global bias and can entangle style or speaker information. The Turkish
model therefore validates every row and request as `tr` while omitting language-embedding and
global-language weights. When two or more languages are trained together, the multilingual recipe
enables those weights and requires an explicit ID for every request.

Same-language Turkish voice cloning is implicit in the monolingual topology. Cross-language pairs
are legal only in the learned multilingual topology and only when the dataset preflight observes
every declared pair.

## Frozen frontend and cache contract

`FrozenTextFrontend` is intentionally not an `nn.Module` child of the acoustic model. It loads with
`trust_remote_code=False`, stays in evaluation/inference mode, has no trainable parameters, and
cannot enter the acoustic optimizer or checkpoint. The trainable acoustic projector remains part of
EchoDiT.

The representation contract binds:

- model and tokenizer IDs plus immutable commit revisions;
- hidden layer, width, maximum length, dtype, input mode, and declared license;
- NFC-only preprocessing, which preserves Turkish casing, apostrophes, punctuation, and numbers;
- a canonical fingerprint used by prepared rows and the model manifest; and
- the exact DACVAE source, revision, filename, SHA-256, sample rate, hop, and latent width.

Prepared features stay FP16 on disk and through collation, then are cast to the acoustic projector's
weight dtype at the model boundary. BF16 cache storage is rejected because the current Arrow path
cannot preserve it without a separate bit-storage contract. Online inference rebuilds the same
pinned provider from the manifest. The provider forward currently runs in FP32 and casts only its
cached output, so preparation and online-inference memory/latency must be profiled separately.
Freezing removes backward activations and optimizer state; it does not remove the 140,493,696
external parameters from deployment.

## Voice conditioning and CFG

Each training target must use a different utterance from the same speaker as its reference. Up to
three references can be concatenated under one duration cap, and inference accepts one or multiple
reference recordings. The reference transcript is never required.

Text and speaker CFG dropout are independent. Conditional, null-text, and null-speaker branches are
constructed once and shared by direct, prepared, sequential, and fused inference paths. A null
speaker has exactly one valid zero patch; padded reference length cannot leak into the unconditional
branch through its mask.

## Why `small/P2` and total duration are the first defaults

With 128-channel DACVAE latents, `P=2` maps 256 acoustic values into the 384-wide `small` backbone.
`P=4` would map 512 values through width 384 and impose a rank-deficient output bottleneck. Both
configuration validation and the model constructor now reject this condition. P2 approximately
halves the DiT token sequence; actual speed and memory still require measurement.

The Turkish recipe contains 44,153,793 trainable acoustic parameters, including a 174,720-parameter
frozen-feature projector path and the trainable speaker/duration modules. Report the external
mmBERT and fixed DACVAE separately. The multilingual P2 topology contains 44,163,521 trainable
acoustic parameters.

The total-duration head learns exact target DACVAE frame counts cheaply. Hard MAS is not a first-run
default because contextual-token boundary tokens need a validated zero-duration policy and the
current backtracking path has CPU/Python work. MAS remains available for a later controlled
experiment after profiling and alignment-unit design.

## Prepare a Turkish voice-cloning dataset

The source dataset needs `audio`, transcript, and speaker-ID columns, with at least two utterances
for each usable speaker. Pin the source dataset to a full commit. The preparer excludes the target
utterance when selecting references.

```python
from pathlib import Path

import yaml

from nar_vae.dataset import prepare_from_hf_dataset
from nar_vae.text_frontend import FrozenTextFrontendSpec

entry = Path("nar_vae/configs/turkish_frozen_config.yaml")
overrides = yaml.safe_load(entry.read_text(encoding="utf-8"))
base = yaml.safe_load((entry.parent / overrides.pop("extends")).read_text(encoding="utf-8"))
base.update(overrides)

prepare_from_hf_dataset(
    dataset_name="OWNER/TURKISH_DATASET",
    dataset_revision="FULL_40_CHARACTER_COMMIT_SHA",
    split="train",
    output_dir=base["TTS_dataset_local"],
    audio_column="audio",
    text_column="text",
    speaker_id_column="speaker_id",
    language="tr",
    max_reference_utterances=3,
    max_reference_seconds=12.0,
    dacvae_model=base["dacvae_model"],
    dacvae_revision=base["dacvae_revision"],
    dacvae_filename=base["dacvae_filename"],
    dacvae_sha256=base["dacvae_sha256"],
    dacvae_backend=base["dacvae_backend"],
    text_frontend_spec=FrozenTextFrontendSpec.from_config(base),
    device="cuda",
)
```

Create a fully resolved editable config without modifying the packaged file:

```bash
python - <<'PY'
from pathlib import Path

import yaml

entry = Path("nar_vae/configs/turkish_frozen_config.yaml")
overrides = yaml.safe_load(entry.read_text(encoding="utf-8"))
base = yaml.safe_load((entry.parent / overrides.pop("extends")).read_text(encoding="utf-8"))
base.update(overrides)
Path("turkish_train.yaml").write_text(
    yaml.safe_dump(base, sort_keys=False),
    encoding="utf-8",
)
PY
```

Adjust data/output paths and hardware batch settings in `turkish_train.yaml`, then:

```python
from nar_vae.train import pretrain

if __name__ == "__main__":
    pretrain("turkish_train.yaml")
```

## Required evaluation before quality claims

At minimum, compare equal-data/equal-compute Turkish runs over mmBERT final versus selected
intermediate layers, `P1` versus `P2`, and several reference durations. A larger frozen ByT5 run can
serve as the method-evidence baseline if budget permits; provider LoRA or low-learning-rate
fine-tuning is a later comparison, not part of the frozen first run.

Report Turkish WER/CER, speaker similarity on unseen speakers, pronunciation slices for
`I/İ/ı/i`, `ğ`, `ö`, `ü`, `ş`, `ç`, numbers, dates, currency, abbreviations and proper-name
apostrophes, MOS/CMOS, GPU-hours, samples/s, peak VRAM, p50/p95 RTF, cache storage, and separate
trainable-acoustic/external-text/fixed-codec parameter counts. Also test 1/3/10/12-second,
multi-clip, noisy, and reverberant references.

The repository ships an architecture and validated contracts, not a trained Turkish checkpoint.
“High quality,” “frozen non-inferiority,” “production-ready cloning,” and “few-step quality” remain
experimental targets. Reference voices also require consent, provenance controls, and an abuse or
watermark policy before production use.
