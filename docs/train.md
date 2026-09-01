# Training

This example trains the NAR-VAE acoustic model. The pinned DACVAE codec is downloaded from the
Hugging Face Hub and kept frozen. The default `small` model has **77,210,945 total parameters**,
all trainable. This count excludes the codec, optimizer state, gradients, and activations.

Training is intended for an NVIDIA CUDA server. No training is run during installation.

## 1. Install

```bash
python -m pip install -e .
wandb login
```

W&B is required. On an offline server, use `WANDB_MODE=offline` instead of logging in.

## 2. Prepare the mini dataset

The example uses
[`SynDataLab-EN-Refs/echo-ref-speakers-4k-en`](https://huggingface.co/datasets/SynDataLab-EN-Refs/echo-ref-speakers-4k-en/tree/28bbd4e65dd6d5ec4da30e21e8999c33b20a902f)
at revision `28bbd4e65dd6d5ec4da30e21e8999c33b20a902f`. It contains 4,000 synthetic
English rows with `audio`, `text`, and `speaker_id` columns.

The dataset card declares no license. Confirm that you have permission to use it before training.
The card says its audio was generated with `jordand/echo-tts-base`; NAR-VAE does not load that
model, but it is part of the data provenance. The dataset also contains one utterance for each
speaker ID, so this small example does not enable speaker conditioning or train voice cloning.

Create `prepare_data.py`:

```python
from nar_vae.dataset import prepare_from_hf_dataset

prepare_from_hf_dataset(
    dataset_name="SynDataLab-EN-Refs/echo-ref-speakers-4k-en",
    dataset_revision="28bbd4e65dd6d5ec4da30e21e8999c33b20a902f",
    split="train",
    output_dir="data/echo-ref-4k",
    dacvae_model="facebook/dacvae-watermarked",
    dacvae_revision="8680102d141858a21bd533543966a2eb2e569f92",
    dacvae_filename="weights.pth",
    dacvae_sha256="573cf4770ea4a25507f26965d05ae720bcd34295a9f60c06ef3c3805826b68e4",
    dacvae_backend="bundled",
    language="en",
    device="cuda",
)
```

Run it on the server:

```bash
python prepare_data.py
```

## 3. Create the configuration

```bash
cp nar_vae/configs/echodit_config.yaml train.yaml
```

Edit these values in `train.yaml`:

```yaml
dacvae_model: facebook/dacvae-watermarked
dacvae_revision: 8680102d141858a21bd533543966a2eb2e569f92
dacvae_filename: weights.pth
dacvae_sha256: 573cf4770ea4a25507f26965d05ae720bcd34295a9f60c06ef3c3805826b68e4
dacvae_sample_rate: 44100
dacvae_hop_length: 512

model_preset: small
use_speaker_conditioning: false
use_language_conditioning: false
supported_languages: [en]

use_mas_duration: true
mas_duration_loss_weight: 0.1
mas_alignment_loss_weight: 0.01

TTS_dataset_local: ./data/echo-ref-4k
save_folder: ./checkpoints/nar_vae_small_pretrain

report_to: wandb
wandb_project: nar-vae-pretraining
wandb_run_name: echo-ref-4k-small
```

The Hub revision and SHA-256 pin the exact codec artifact. The sample rate, hop length, and latent
width must also match the prepared dataset.

## 4. Train

Create `train_job.py`:

```python
from nar_vae.train import pretrain

if __name__ == "__main__":
    pretrain("train.yaml")
```

One GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train_job.py
```

Multiple GPUs:

```bash
torchrun --standalone --nproc-per-node=8 train_job.py
```

Checkpoints are written under `save_folder`. This mini dataset is useful for checking that the
pipeline works; it is not enough to demonstrate production speech quality or low WER.
