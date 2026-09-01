# NAR-VAE

NAR-VAE is a research library for low-cost, non-autoregressive text-to-speech with
diffusion transformers and conditional flow matching. The Python package and all new examples use
`nar_vae`.

> [!IMPORTANT]
> NAR-VAE does not currently publish a trained NAR-VAE checkpoint. The repository provides model,
> training, post-training, and inference infrastructure; it does not yet demonstrate high-quality,
> low-WER, multilingual, zero-shot, real-time, or streaming speech. Those labels require a trained
> artifact and held-out evaluation on named data and hardware.

The current implementation includes:

- random-initialized acoustic pretraining and full-parameter supervised fine-tuning (SFT);
- tiny through xlarge model presets (with `small` as the scratch default), explicit mixed
  precision, activation checkpointing, optional compilation, and one-process-per-GPU distributed
  training;
- optional, rank-zero-safe Weights & Biases reporting;
- monolingual or explicitly versioned multilingual conditioning;
- a learned utterance-length predictor plus monotonic alignment search (MAS) and exact duration
  allocation utilities;
- exact-latent-length inference batching and optional, fail-closed Cache-DiT acceleration; and
- an experimental executable flow-native Group Relative Policy Optimization (Flow-GRPO) stage
  plus lower-level research primitives.

## Install

Python 3.10 or newer is required. Install from a source checkout:

```bash
python -m pip install -e .
```

For a CPU-only development environment, install matching official PyTorch and TorchAudio CPU
wheels first, then install NAR-VAE. Server training should instead use the PyTorch build matched to
the server's CUDA driver. The optional Cache-DiT extra currently brings its own newer PyTorch floor.

Training is intended for a CUDA server and has a separate dependency set:

```bash
python -m pip install -e ".[train]"
```

Add integrations only when needed:

```bash
# Training with W&B
python -m pip install -e ".[train,wandb]"

# Inference with Cache-DiT
python -m pip install -e ".[turbo]"

# Optional pinned Flash Attention 3 dispatch on a compatible CUDA server
python -m pip install -e ".[flash-attention]"

# Optional faster codec backend, pinned to the reviewed source revision
python -m pip install 'fast-dacvae @ git+https://github.com/kadirnar/fast-dacvae.git@406f2e5c803927ef18cc9bbe38d715e5417459b9'
```

The fast codec is accepted only when its installed PEP 610 provenance names that exact repository
and resolved Git commit. PyPI, local-directory, missing-provenance, and other-revision installs fail
closed; `dacvae_backend: bundled` remains independent of this optional package.

## Acoustic pretraining from scratch

`nar_vae.train.pretrain` is the canonical first acoustic-model stage. It constructs every active
text, flow, language, speaker, and duration component from random weights. It rejects
`pretrained_checkpoint` and other external TTS initialization fields. Interruption recovery is
allowed only through `resume_from_checkpoint: true` or a `save_folder/checkpoint-N` directory from
the same run. There is no OPT language-model backbone or OPT checkpoint integration; text
conditioning uses the project's trainable encoder.

Materialize the packaged pretraining configuration as one editable YAML file:

```bash
python - <<'PY'
from pathlib import Path

import yaml

from nar_vae.train import DEFAULT_TRAIN_CONFIG_PATH

entry = Path(DEFAULT_TRAIN_CONFIG_PATH)
overrides = yaml.safe_load(entry.read_text(encoding="utf-8"))
base = yaml.safe_load((entry.parent / overrides.pop("extends")).read_text(encoding="utf-8"))
base.update(overrides)
Path("pretrain.yaml").write_text(
    yaml.safe_dump(base, sort_keys=False),
    encoding="utf-8",
)
PY
```

Before launching, set the prepared dataset, output directory, model preset, precision, batch size,
and worker count for the target server. Keep these scratch-stage invariants:

```yaml
training_stage: pretrain
model_initialization: random
pretrained_checkpoint: null
model_preset: small
mixed_precision: bf16
TTS_dataset_local: ./data/prepared
dacvae_model: ./codecs/dacvae/weights.pth
dacvae_backend: bundled
dacvae_sha256: <64-character lowercase SHA-256>
max_frames_per_batch: 4096
max_examples_per_batch: null  # Uses batch_size.
frame_bucket_size: 128
allow_legacy_frame_length_inference: false
```

### Dataset row format

The preparation API accepts a Hugging Face-style row containing paired audio and text. Language,
speaker, and recording-session fields are optional and are selected through the corresponding
`*_column` arguments:

```python
raw_row = {
    "audio": {"array": waveform_float32, "sampling_rate": 48000},
    "text": "A transcript matching the waveform.",
    "language": "en",  # Optional multilingual label.
    "speaker_id": "speaker-1",  # Optional voice-reference grouping.
    "session_id": "session-1",  # Optional leakage-safe reference selection.
}
```

Preparation converts it to the training schema below and saves the result with
`Dataset.save_to_disk` plus `nar_vae_dataset_manifest.json`:

```text
latents:                 float32[latent_channels, latent_frames]
latent_num_frames:       int
conditioning_ids:        list[int]
language:                str
representation_contract: versioned text/codec identity
speaker_latents:         float32[latent_channels, reference_frames]  # optional
speaker_language:        str                                         # optional
utterance_id:            str                                         # SFT/GRPO
```

### Pretraining architecture

The acoustic model is initialized entirely from random weights. A trainable text encoder, optional
language embedding, and optional speaker encoder condition an adaLN-Zero EchoDiT velocity model.
At training time, straight conditional flow matching interpolates Gaussian noise and clean codec
latents; the DiT predicts the velocity. MAS supplies hard monotonic token/frame paths for the
learned duration head and duration-expanded frame conditioning:

```text
text IDs -> text encoder ---------> cross-attention + frame text state ---+
speaker reference -> speaker encoder (optional) --------------------------+-> EchoDiT -> velocity
noise + clean codec latents -> flow interpolation ------------------------+
clean codec latents + text state -> MAS -> duration targets --------------+
```

Prepared datasets persist `latent_num_frames`, so the Trainer can build deterministic sortish
batches without loading every latent array. The frame budget caps the sum of unpadded latent frames
per device, while `max_examples_per_batch` retains an example-count ceiling. Set
`max_frames_per_batch: 0` to restore the normal fixed-size Trainer loader. In DDP, NAR-VAE builds
one global batch plan and lets Accelerate shard it once; `dataloader_drop_last: true` is required so
all ranks execute the same number of steps.

Every newly prepared row also carries one `representation_contract` with the text-frontend version,
codec source/backend, pinned Hub revision and filename when applicable, codec artifact SHA-256,
sample rate, hop length, and latent width. Pretraining and SFT require that contract by default,
require it to be identical across rows, and compare it with the training YAML. Set
`allow_legacy_representation: true` only after separately auditing and migrating older data.

Preparation also writes `nar_vae_dataset_manifest.json` beside each local
`Dataset.save_to_disk` result. It hashes the exact prepared Arrow/metadata artifact inventory and
records its row count, columns, and persisted Dataset state fingerprint; it does not recursively
hash the raw-audio source tree. Pretraining and SFT reject a missing manifest, altered artifact, or
unlisted file before creating or resuming a run. Re-prepare older local datasets once with this
version rather than fabricating the manifest.

Codec strings are local-only. Point `dacvae_model` at the exact local weight file and set
`dacvae_sha256` to the output of `sha256sum`; remote preparation must construct
`HubDACVAESource(repo_id, revision=<full commit>, filename=...)`. This opt-in source stores the same
revision, filename, and downloaded artifact hash in every prepared row.

The configured local path is authoritative and a missing path fails instead of silently switching
to the Hub. A remote training repository must contain the same manifested
`Dataset.save_to_disk` artifacts at its root; executable dataset builders that can fetch mutable
external URLs are rejected. To choose that prepared remote snapshot explicitly, set the local path
to `null`, replace the placeholder repo id, and provide its full 40-character commit SHA:

```yaml
TTS_dataset_local: null
TTS_dataset: organization/prepared-speech
TTS_dataset_revision: 0123456789abcdef0123456789abcdef01234567
dataset_download_workers: 8  # Valid range: 1..32.
```

Remote runs bind that commit together with the exact prepared Arrow/metadata byte inventory and
persisted Dataset fingerprint. The resolved local or remote content identity is stored in the
immutable training run; same-run resume validates it before any Trainer weights or optimizer state
are loaded.

Create a guarded training script so it works with both a direct Python launch and `torchrun`:

```python
# pretrain_job.py
from nar_vae.train import pretrain

if __name__ == "__main__":
    pretrain("pretrain.yaml")
```

Run on one server GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_job.py
```

Run one process per GPU with DistributedDataParallel (DDP):

```bash
torchrun --standalone --nproc-per-node=8 pretrain_job.py
```

With fixed-size loading, `batch_size` is per device and the effective global batch is
`batch_size * gradient_accumulation_steps * world_size`. With frame-budget loading, the number of
examples varies by step and `batch_size` is the default per-device maximum. The training entry
point supports FP32, FP16, or BF16, AdamW (including opt-in fused AdamW on compatible CUDA builds),
non-reentrant gradient checkpointing, gradient clipping, warmup and scheduler controls, data-loader
controls, DDP bucket settings, checkpoint retention, and optional `torch.compile`. FSDP is not
implemented by this entry point and fails closed instead of being silently ignored.

To enable W&B, install the extra and set:

```yaml
report_to: wandb
wandb_project: nar-vae-pretraining
wandb_run_name: small-baseline
wandb_log_model: false
```

The Trainer integration logs from the world process only. `report_to: none` is the default; normal
W&B environment variables, including offline mode, remain available.

### Codec boundary

Acoustic pretraining currently consumes prepared continuous codec latents. The codec is a separate,
versioned representation dependency, not a TTS teacher, and its identity, latent width, frame rate,
and preprocessing must remain fixed across dataset preparation, training, SFT, and inference. The
bundled codec loader and optional faster backend do not make this an end-to-end first-party scratch
pipeline. Truly end-to-end training from raw waveform still requires a first-party codec-pretraining
and evaluation stage.

## Monolingual and multilingual runs

Language capability is learned and stored in checkpoint tensors; listing a language in YAML does
not create support for it. A monolingual run keeps conditioning disabled:

```yaml
use_language_conditioning: false
supported_languages: [en]
```

A multilingual run enables it before pretraining and lists only language codes represented in the
paired dataset and supported by the versioned registry:

```yaml
use_language_conditioning: true
supported_languages: [en, tr, de]
```

Balance and evaluate data per language instead of allowing the largest corpus to dominate. Speaker
conditioning is an independent capability and must be trained from valid reference latents.
Cross-lingual voice transfer additionally requires separate, trained target-language and
reference-language coverage; it is never inferred from a multilingual label.

## Duration and alignment

There is no duration method that is universally best for every diffusion TTS model. For compact and
low-resource experiments, the recommended research path is MAS-derived token alignments followed by
a lightweight learned duration predictor, with integer durations reconciled to the exact requested
frame total.

The compatibility configuration keeps only the learned utterance-frame-count head. Canonical new
scratch pretraining additionally enables `use_mas_duration`: a compact text-conditioned Gaussian
prior scores a fixed projection of clean codec frames, no-gradient MAS paths provide hard targets,
and the model learns positive per-token `log1p` durations plus alignment likelihood. Both objectives
use padding masks and globally normalized valid counts under DDP. The checkpoint tensors and MAS
version metadata, rather than YAML alone, gate this capability; SFT must retain the parent
architecture and cannot add MAS to a legacy checkpoint.

`predict_token_duration_frames(..., total_frames=...)` exposes stable positive integer allocation
whose sum exactly matches the requested total. Every unmasked conditioning token—including boundary
tokens—receives at least one frame, so MAS dataset preflight rejects rows with fewer latent frames
than tokens. The velocity DiT retains global cross-attention to the token sequence and also adds a
learned projection of duration-expanded text state at every latent frame. Training uses the hard MAS
path for that regulator; inference predicts one exact integer allocation and reuses the expanded
state across every ODE evaluation and CFG branch.

Online MAS is a sequential dynamic program over each row and frame, so it can reduce accelerator
utilization and must not be described as the fastest duration path. The
[Matcha-TTS alignment guide](https://github.com/shivammehta25/Matcha-TTS/wiki/Extracting-phoneme-alignments-and-improving-GPU-utilisation)
recommends extracting and fixing alignments after they stabilize. Persisted-duration ingestion is a
future optimization here; the current implementation intentionally recomputes online paths. See the
[architecture and training plan](https://github.com/kadirnar/nar-vae/blob/main/docs/architecture.md)
for evaluation gates.

## Supervised fine-tuning

SFT is population-level supervised continuation, not per-user voice adaptation. A fresh SFT run may
load only a local NAR-VAE pretraining export. Pretraining writes `lineage.json` and
`nar_vae_manifest.json` beside `pytorch_model.bin`. The model manifest binds every loaded weight
artifact by SHA-256, the exact active architecture and capabilities, and the dataset's frontend and
codec contract. SFT rejects a missing manifest, a non-pretraining parent, a renamed artifact, a
hash mismatch, or a codec/frontend change. External and legacy TTS checkpoints are not accepted by
the manifest-required inference path.

Copy the packaged SFT template without depending on a repository-internal path:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path

source = files("nar_vae.configs").joinpath("finetune_config.yaml")
Path("sft.yaml").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
PY
```

Set `pretrained_checkpoint` to the pretraining export's `pytorch_model.bin`, keep both adjacent
JSON manifests, and configure the prepared SFT dataset. Then use the same single-GPU or `torchrun`
launch pattern:

```python
# sft_job.py
from nar_vae.finetune import finetune

if __name__ == "__main__":
    finetune("sft.yaml")
```

```bash
CUDA_VISIBLE_DEVICES=0 python sft_job.py
torchrun --standalone --nproc-per-node=8 sft_job.py
```

Same-run resume restores Trainer state and, when enabled, EMA state. SFT exports a new hash-bound
manifest that records its pretraining parent. Dataset design, unseen-speaker boundaries, reward
selection, and promotion criteria are covered in the
[post-pretraining guide](https://github.com/kadirnar/nar-vae/blob/main/docs/post_pretraining.md).

## Flow-GRPO post-training

`nar_vae.post_training.grpo_post_train` is the executable server-side GRPO stage. It accepts only a
local, hash-bound NAR-VAE SFT export descended from this repository's scratch-pretraining stage;
the packaged template points to
`checkpoints/nar_vae_small_sft/final/flow_model/pytorch_model.bin`. It will not import a third-party
pretrained TTS model. Copy `DEFAULT_GRPO_CONFIG_PATH`, then replace the dataset/checkpoint paths and
every evaluator identity before launch. Prepared fine-tuning rows include a unique `utterance_id`;
set `utterance_id_column` during preparation to preserve a trusted source ID, or preparation derives
a deterministic content-bound ID.

```python
# grpo_job.py
from pathlib import Path

import yaml

from nar_vae.post_training import (
    DEFAULT_GRPO_CONFIG_PATH,
    bind_reward_evaluator_manifest,
    grpo_post_train,
)

config_path = Path("grpo.yaml")
if not config_path.exists():
    config_path.write_text(
        Path(DEFAULT_GRPO_CONFIG_PATH).read_text(encoding="utf-8"), encoding="utf-8"
    )
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))


def speech_reward(audio, batch):
    # Return one [prompt_batch, group_size] tensor for each configured component.
    # Run pinned, language-appropriate ASR/CER, speaker, and perceptual evaluators here.
    raise NotImplementedError


speech_reward = bind_reward_evaluator_manifest(speech_reward, config["reward_evaluators"])

if __name__ == "__main__":
    grpo_post_train(config_path, reward=speech_reward)
```

```bash
CUDA_VISIBLE_DEVICES=0 python grpo_job.py
torchrun --standalone --nproc-per-node=8 grpo_job.py
```

One dataset row is one prompt group; candidates never cross ranks. The stage reuses a detached
rollout for `policy_update_epochs` (at least two), keeps old/reference log probabilities fixed,
normalizes reward components across ranks, decodes and evaluates exact unpadded lengths, and uses
the SFT reference's fixed MAS token allocation throughout each update. It writes atomic,
hash-inventoried checkpoints containing policy/reference weights, optimizer, scheduler, all
rank-local RNG state, and reference/dataset manifests. Set `resume_from_checkpoint: true` to select
the latest fully validated checkpoint from the same immutable run, or name that run's
`checkpoint-N` for fail-closed explicit recovery. When automatic recovery replays past a corrupt
newer seal, those rejected bytes are preserved under a hidden `.rejected-checkpoint-N.*` name
before the replacement is published.

W&B is rank-zero-only and receives rank-averaged metrics. Install `.[train,wandb]`, then set
`report_to: wandb`, `wandb_project`, and `wandb_run_name`. External ASR, speaker, and perceptual
evaluators are deliberately not bundled or downloaded: their implementation, immutable revision,
and SHA-256 must match `reward_evaluators`, and the callable must be bound as above. Use held-out
metrics and human listening for promotion because optimizing a combined reward can hide reward
hacking. The lower-level `FlowGRPOTrainer` remains available for research integrations.

## Inference with your checkpoint

Because there is no released trained NAR-VAE checkpoint, inference examples require a compatible
local export and the exact codec used to prepare its training latents. The architecture preset must
also match the checkpoint:

```python
from nar_vae import FlowMatchingTTSInference

tts = FlowMatchingTTSInference.from_preset(
    "small",
    flow_model_path="checkpoints/nar_vae_small_sft/final/flow_model/pytorch_model.bin",
    dacvae_model="./codecs/dacvae/weights.pth",
    device="cuda",
)

profile = tts.generation_profile("quality")
audio = tts.synthesize_with_config(
    "This sentence is generated by a locally trained checkpoint.",
    profile,
)
tts.save_audio(audio, "output.wav")
```

Inference loads `nar_vae_manifest.json` beside the acoustic weights and verifies the selected
artifact plus any required EMA base before deserializing the checkpoint. It then checks the
architecture/frontend contract and verifies the local codec SHA-256 before deserializing the codec.
A remote codec is accepted only as an explicit `HubDACVAESource` whose full commit and filename
match that manifest.

`quality`, `balanced`, `fast`, and `turbo` are numerical generation profiles, not verified quality
tiers. Select solver steps, guidance, and caching only from measurements on the exact
checkpoint. Report WER or CER by language, speaker similarity where applicable, perceptual and
listening results, latency, memory, and the checkpoint/codec hashes. All packaged profiles start
from neutral conditional inference (`cfg_scale=1`) with no text- or speaker-guidance branches;
non-neutral CFG is an explicit checkpoint-specific experiment.

### Exact-length batching

```python
audios = tts.synthesize_batch(
    ["A short sentence.", "A second sentence with a different length."],
    num_steps=32,
    solver="euler",
)
```

The batch path resolves every utterance length first, groups requests with the exact same latent
frame count, masks text padding, and restores input order. It does not place differently sized
target-noise tensors in one padded ODE batch. This protects valid audio from padded latent state,
though heterogeneous requests may form small groups and reduce throughput. `max_duration` is an
optional validation ceiling; it never clips a request. Heuristic or learned duration predictions
above the configured ceiling fail with an error so callers can split the text deliberately.

### Optional Cache-DiT

Cache-DiT is experimental and currently applies to single-utterance multi-step inference:

```python
audio = tts.synthesize(
    "Cache reuse must be calibrated against the uncached result.",
    num_steps=16,
    solver="euler",
    cache_mode="cache_dit",
    cfg_scale=1.0,
    cfg_mode="joint",
    cfg_min_t=0.0,
    cfg_max_t=1.0,
)
```

The runtime requires at least eight Euler steps. With guidance enabled, alternating CFG is rejected
and the CFG window must remain fixed over `[0, 1]`. Cache state is request-local and resets for each
utterance; incompatible shape, language, speaker, guidance, precision, solver, or checkpoint state
must never be shared. Cache-DiT is unlikely to help a one- to four-step distilled student, does not
guarantee a speedup, and must be gated against uncached WER/CER and listening results.

Current inference integrates and decodes the complete utterance before returning audio. It is not
streaming, and reported latency is complete-waveform latency. See
[inference optimization](https://github.com/kadirnar/nar-vae/blob/main/docs/inference_optimization.md)
for the measurement contract, Cache-DiT
constraints, and the separate architecture work required for true streaming.

## Documentation

- [Architecture and training plan](https://github.com/kadirnar/nar-vae/blob/main/docs/architecture.md)
- [Post-pretraining, SFT, Flow-GRPO, and evaluation](https://github.com/kadirnar/nar-vae/blob/main/docs/post_pretraining.md)
- [Inference optimization and streaming research](https://github.com/kadirnar/nar-vae/blob/main/docs/inference_optimization.md)

## Development

```bash
python -m pip install -e ".[train,dev]"
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
python -m build
```

The default test suite is network-free and GPU-free. Server training and quality claims require
separate, recorded experiments.

## License

MIT. See [LICENSE](https://github.com/kadirnar/nar-vae/blob/main/LICENSE).
