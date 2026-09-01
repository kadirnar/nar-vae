# NAR-VAE

NAR-VAE is a low-cost, non-autoregressive latent-diffusion TTS library for multilingual speech
and zero-shot reference-audio voice cloning. The new checkpoint topology uses strict cosine
variance-preserving (VP) diffusion with v-prediction, a frozen multilingual phoneme backbone, a
compact learned voice resampler, and an unchanged DACVAE codec.

No trained NAR-VAE checkpoint is published yet. The implementation and compatibility contracts
are ready for training, but naturalness, WER/CER, speaker similarity, and speed must be measured
after training. Configuration profile names are not quality claims.

## Architecture

```text
reviewed multilingual phones
        |
        v
pinned frozen XPhoneBERT --cached states--> small trainable adapter --+
                                                                    |
reference audio -> unchanged DACVAE encoder -> speaker encoder       +--> small EchoDiT
                  (sampled posterior, content-seeded externally)     |     VP v-prediction
                                             -> 8 learned summaries  |
                                                                    |
Gaussian noise -----------------------------------------------------+--> DDIM/ODE
                                                                         |
                                                                         v
                                                      unchanged DACVAE decoder -> audio
```

For clean DACVAE latent `x`, Gaussian noise `epsilon`, and generation-direction time
`t in [0, 1]`:

```text
alpha(t) = sin(pi*t/2)
sigma(t) = cos(pi*t/2)
x_t      = alpha(t) * x + sigma(t) * epsilon
v_target = sigma(t) * x - alpha(t) * epsilon
```

This is diffusion, not the straight rectified-flow interpolation used by Echo-TTS and Irodori.
Legacy flow checkpoints remain loadable and are identified separately; objective and schedule
metadata cannot be mixed.

The design review, source links, accepted ideas, rejected alternatives, and required ablations are
in [docs/research_2025_2026.md](docs/research_2025_2026.md).

## Why training is relatively cheap

- XPhoneBERT is frozen, runs during data preparation, and its token-aligned states are cached.
  It is not saved in the acoustic checkpoint or evaluated at every diffusion step.
- The canonical `small` acoustic model has about 44.85M trainable parameters with multilingual,
  duration/MAS, and voice-cloning modules. The external text backbone and frozen codec are reported
  separately.
- Arbitrary-length reference state is compressed to eight learned summary tokens before DiT
  cross-attention.
- Target-latent `P2` packing halves the DiT time axis outside DACVAE; `P1` remains the quality
  control and `P4` an aggressive ablation.
- Frozen text states, target DACVAE latents, and source utterance latents are prepared once.
- BF16, activation checkpointing, padded-attention cost batching, exact batched GPU MAS, DDP, and
  optional Muon/AdamW are supported.

Canonical trainable parameter counts for the frozen-text, voice-conditioned topology are:

| Preset | Trainable acoustic parameters |
| --- | ---: |
| nano | 3.60M |
| tiny | 15.32M |
| small | 44.85M |
| medium | 109.66M |
| large | 284.41M |
| xlarge | 556.22M |

These counts exclude the immutable DACVAE and frozen 88M-class XPhoneBERT provider.

## Install

Python 3.10+ is supported. Training requires a suitable NVIDIA CUDA server.

```bash
git clone https://github.com/kadirnar/nar-vae.git
cd nar-vae
python -m pip install -e .
wandb login
```

`pyproject.toml` is the dependency manifest. W&B is required by the current training entry points;
use `WANDB_MODE=offline` on an isolated server.

## Prepare multilingual cloning data

Each raw row should contain audio, a transcript, language, stable speaker/utterance identity, and
reviewed, provider-native phonemes compatible with the pinned XPhoneBERT vocabulary. The value
below is deliberately a placeholder: do not copy an IPA string until your language frontend has
been tested against the pinned tokenizer.

```python
{
    "audio": {"array": waveform_float32, "sampling_rate": 44100},
    "text": "Merhaba dünya.",
    "phonemes": reviewed_provider_phones,
    "language": "tr",
    "speaker_id": "corpus-a/speaker-001",
    "utterance_id": "corpus-a/recording-001",
}
```

Canonical frozen-feature preparation requires supplied phones; it does not guess G2P. Pin and
test normalization/G2P separately for every language you intend to advertise. Unknown phones,
unsupported control tags, mixed provider contracts, and changed artifact hashes fail closed.

Voice-cloning speakers need at least two genuinely different utterances. Preparation stores one
target latent per utterance. During training, `DynamicReferenceDataset` selects and crops another
same-speaker recording, so speaker IDs are never model inputs and reference arrays are not
duplicated on disk.

See [docs/train.md](docs/train.md) for provider construction and dataset preparation.

## Train

Copy the base and stage configurations, then edit the dataset, output, language, and exact
target/reference-language pairs:

```bash
cp nar_vae/configs/echodit_config.yaml echodit_config.yaml
cp nar_vae/configs/pretrain_config.yaml train.yaml
```

```python
from nar_vae.train import pretrain

if __name__ == "__main__":
    pretrain("train.yaml")
```

One GPU:

```bash
python train_job.py
```

DDP:

```bash
torchrun --standalone --nproc-per-node=8 train_job.py
```

The default is `small`, strict VP diffusion, `P2` target packing, eight speaker summaries, frozen
768-D text states, learned duration, and exact MAS. The frozen provider is a data/inference
dependency; only its small adapter is trained.

## Voice-cloning inference

```python
from nar_vae import FlowMatchingTTSInference

tts = FlowMatchingTTSInference.from_preset(
    "small",
    flow_model_path="checkpoints/pretrain/final/flow_model/pytorch_model.bin",
    dacvae_model="facebook/dacvae-watermarked",
    device="cuda",
)

audio = tts.synthesize_with_config(
    "Bu bir ses klonlama örneğidir.",
    tts.generation_profile("quality"),
    phonemes=reviewed_provider_phones,
    reference_audio="speaker.wav",
    language="tr",
    reference_language="en",
)
tts.save_audio(audio, "clone.wav")
```

Inference authenticates the model manifest, objective, schedule, architecture, frozen-provider
identity, codec hash, speaker-summary topology, language capability, and requested language pair
before model use. The provider is evaluated once per request; DDIM then reuses its cached state.

The packaged VP profiles use 32, 16, and 8 deterministic DDIM evaluations. Start with `quality`
and compare all faster profiles against held-out listening, WER/CER, and speaker-similarity gates.

More detail: [architecture](docs/architecture.md), [training](docs/train.md),
[post-training](docs/post_pretraining.md), and
[inference optimization](docs/inference_optimization.md).

MIT licensed.
