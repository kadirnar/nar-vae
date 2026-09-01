# NAR-VAE

NAR-VAE is a non-autoregressive text-to-speech research library built around conditional flow
matching, EchoDiT, and continuous DACVAE latents. It supports acoustic-model scratch pretraining,
SFT, flow-native GRPO, single- and multi-GPU training, multilingual conditioning, and Cache-DiT
inference. The separately supplied DACVAE codec remains fixed in every implemented training stage.

> NAR-VAE does not publish a trained acoustic checkpoint yet. Quality, WER, multilingual,
> zero-shot, latency, and streaming claims require a trained model and held-out evaluation.

## Setup

Python 3.10 or newer is required. Training is intended for an NVIDIA CUDA server.

```bash
git clone https://github.com/kadirnar/nar-vae.git
cd nar-vae
python -m pip install -e .
```

`pyproject.toml` is the only dependency manifest. This single installation includes training, W&B,
testing, Cache-DiT, and the supported attention integration.

## Dataset format

Dataset preparation accepts Hugging Face-style paired audio and text rows:

```python
raw_row = {
    "audio": {"array": waveform_float32, "sampling_rate": 48000},
    "text": "The transcript matching this waveform.",
    "language": "en",  # Optional multilingual label.
    "speaker_id": "speaker-1",  # Optional same-speaker reference pairing.
}
```

Speaker-conditioned preparation selects another utterance from the same speaker; the target
utterance is never used as its own voice reference.

The prepared dataset contains:

```text
latents:                 float32[latent_width, frames]
latent_num_frames:       int
conditioning_ids:        list[int]
language:                str
representation_contract: tokenizer and exact codec identity
speaker_latents:         float32[latent_width, reference_frames]  # optional
speaker_language:        str                                      # optional
utterance_id:            str                                      # SFT/GRPO
```

Preparation saves the dataset with `Dataset.save_to_disk` and writes
`nar_vae_dataset_manifest.json`. Training verifies the dataset inventory, tokenizer contract, codec
revision, and codec SHA-256 before loading or resuming.

## Scratch-pretraining architecture

Every acoustic-model component starts from random weights. The DACVAE codec is a fixed
representation dependency, not a pretrained TTS teacher.

```text
text IDs -> trainable text encoder --------------------+
language embedding (optional) -------------------------+
speaker latents -> speaker encoder (optional) ---------+-> EchoDiT -> velocity
noise + clean codec latents -> flow interpolation -----+
clean latents + text states -> MAS/duration path -------+
```

For noise `x0`, clean codec latents `x1`, and timestep `t`:

```text
xt = (1 - t) * x0 + t * x1
target velocity = x1 - x0
```

EchoDiT predicts the velocity. MAS produces hard monotonic token/frame alignments for the learned
duration head and duration-expanded frame conditioning. No external language model or third-party
TTS checkpoint initializes the acoustic model.

## Pretraining

Create an editable configuration from the packaged scratch-pretraining recipe:

```bash
python - <<'PY'
from pathlib import Path

import yaml

from nar_vae.train import DEFAULT_TRAIN_CONFIG_PATH

entry = Path(DEFAULT_TRAIN_CONFIG_PATH)
overrides = yaml.safe_load(entry.read_text(encoding="utf-8"))
base = yaml.safe_load((entry.parent / overrides.pop("extends")).read_text(encoding="utf-8"))
base.update(overrides)
Path("pretrain.yaml").write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
PY
```

Set at least these values:

```yaml
training_stage: pretrain
model_initialization: random
pretrained_checkpoint: null
model_preset: small

TTS_dataset_local: ./data/prepared
dacvae_model: ./codecs/dacvae/weights.pth
dacvae_backend: bundled
dacvae_sha256: <64-character-lowercase-sha256>
save_folder: ./checkpoints/nar_vae_small_pretrain

report_to: wandb
wandb_project: nar-vae-pretraining
wandb_run_name: small-baseline
```

W&B is mandatory for pretraining, SFT, and GRPO. Authenticate normally for online logging or set
`WANDB_MODE=offline` on an isolated server. Only rank zero creates a W&B run.

Create `pretrain_job.py`:

```python
from nar_vae.train import pretrain

if __name__ == "__main__":
    pretrain("pretrain.yaml")
```

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_job.py
```

Multi-GPU DDP:

```bash
torchrun --standalone --nproc-per-node=8 pretrain_job.py
```

### Training optimizations

| Packaged default | Optional or tunable |
| --- | --- |
| BF16 mixed precision | FP32/FP16 comparison runs |
| non-reentrant activation checkpointing | disable if memory is plentiful |
| frame-budget batching and bucketing | tune frame budget and workers per server |
| stratified logit-normal timestep sampling | uniform or logit-normal sampling |
| pinned-memory loading and DDP-safe `drop_last` | DDP bucket-size tuning |
| AdamW and gradient clipping | fused AdamW on supported CUDA builds |
| PyTorch SDPA attention | FA3 with `NAR_VAE_USE_FA3=1` after parity tests |
| eager execution | `torch_compile: true` after shape-bucket tests |
| strict FP32 matmul behavior | `tf32: true` after numerical checks |

The `tiny` and `small` presets reduce training cost. Larger presets, compilation, fused AdamW,
TF32, and FA3 are not assumed to improve every server; measure throughput, memory, convergence,
WER/CER, and listening quality before keeping them. FSDP is not implemented; use DDP.

## Inference

Inference requires a locally trained NAR-VAE export and the exact codec used to prepare its data:

```python
from nar_vae import FlowMatchingTTSInference

tts = FlowMatchingTTSInference.from_preset(
    "small",
    flow_model_path="checkpoints/nar_vae_small_sft/final/flow_model/pytorch_model.bin",
    dacvae_model="codecs/dacvae/weights.pth",
    device="cuda",
)

audio = tts.synthesize_with_config(
    "This sentence uses a locally trained checkpoint.",
    tts.generation_profile("quality"),
)
tts.save_audio(audio, "output.wav")
```

Inference verifies the model manifest, weight hashes, architecture, capabilities, tokenizer, and
codec SHA-256. Profile names describe numerical settings, not demonstrated quality. Current
inference returns a complete waveform after ODE integration and codec decoding; it is not streaming.

## More documentation

- [Architecture and training](docs/architecture.md)
- [SFT, GRPO, and evaluation](docs/post_pretraining.md)
- [Inference optimization](docs/inference_optimization.md)

MIT licensed.
