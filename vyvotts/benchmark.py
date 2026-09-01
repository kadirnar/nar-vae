"""Reproducible complete-waveform latency benchmark for NAR-VAE."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from vyvotts.benchmarking import (
    environment,
    file_metadata,
    package_source_hashes,
    summarize,
)
from vyvotts.checkpoint import HubCheckpointSource
from vyvotts.configuration import (
    GenerationConfig,
    load_inference_settings,
    validate_cache_dit_options,
)
from vyvotts.dacvae import (
    DACVAE_BACKENDS,
    HubDACVAESource,
    describe_dacvae_source,
)
from vyvotts.inference_realtime import RealtimeTTSInference
from vyvotts.languages import DEFAULT_LANGUAGE, LanguagePair
from vyvotts.serving import StageTiming, non_claim_evidence, summarize_stage_timings


def _run_once(
    tts: RealtimeTTSInference,
    *,
    text: str,
    duration: float | None,
    config: GenerationConfig,
    seed: int,
    language_pair: LanguagePair,
    reference_audio: str | Path | torch.Tensor | None,
    reference_sample_rate: int | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()

    audio, timings = tts.synthesize_fast(
        text,
        config=config,
        duration=duration,
        return_timing=True,
        reference_audio=reference_audio,
        reference_sample_rate=reference_sample_rate,
        language=language_pair.target,
        reference_language=language_pair.reference,
    )
    if not torch.isfinite(audio).all():
        raise RuntimeError("Inference returned non-finite audio samples.")
    return audio, timings


def run_benchmark(
    *,
    checkpoint: str | Path | HubCheckpointSource,
    dacvae_model: str | Path | HubDACVAESource | None = None,
    dacvae_backend: str = "bundled",
    device: str = "cuda",
    text: str = "The quick brown fox jumps over the lazy dog.",
    language: str = DEFAULT_LANGUAGE,
    reference_audio: str | Path | torch.Tensor | None = None,
    reference_sample_rate: int | None = None,
    reference_language: str | None = None,
    profile: str = "fast",
    duration: float | None = None,
    num_steps: int | None = None,
    solver: str | None = None,
    cfg_scale: float | None = None,
    cfg_mode: str | None = None,
    warmup_runs: int = 2,
    runs: int = 30,
    seed: int = 1234,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    base_weights: bool = False,
    cache_mode: str | None = None,
    output: str | Path = "benchmark_results.json",
    save_audio: str | Path | None = None,
) -> dict[str, object]:
    """Run the TTFT/TTFA benchmark and return its serializable result.

    Pass an explicit output path to place the JSON elsewhere; ``save_audio``
    additionally saves the final measured waveform.
    """
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive.")
    if num_steps is not None and num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if warmup_runs < 0 or runs <= 0:
        raise ValueError("warmup_runs must be non-negative and runs must be positive.")
    if dacvae_backend not in DACVAE_BACKENDS:
        raise ValueError(
            f"Unknown DACVAE backend {dacvae_backend!r}; expected one of {DACVAE_BACKENDS}."
        )
    if cache_mode not in (None, "none", "cache_dit", "fast", "ultra"):
        raise ValueError(
            "cache_mode must be one of: None, 'none', 'cache_dit', 'fast', or 'ultra'."
        )
    if dacvae_model is None:
        raise ValueError(
            "dacvae_model is required; pass an exact local artifact or a commit-pinned "
            "HubDACVAESource."
        )

    requested_checkpoint = (
        checkpoint.repo_id if isinstance(checkpoint, HubCheckpointSource) else str(checkpoint)
    )
    language_pair = LanguagePair.resolve(
        language,
        reference_language,
        has_reference=reference_audio is not None,
    )
    dacvae_description = describe_dacvae_source(dacvae_model)
    requested_dacvae_model = dacvae_description.identifier
    output_path = Path(output)
    save_audio_path = Path(save_audio) if save_audio is not None else None

    profile_config = load_inference_settings().profile(profile)
    effective_cache_mode = cache_mode if cache_mode is not None else profile_config.cache_mode
    config_cache_mode = (
        effective_cache_mode if effective_cache_mode in ("none", "cache_dit") else "none"
    )
    config = profile_config.with_overrides(
        num_steps=num_steps,
        solver=solver,
        cfg_scale=cfg_scale,
        cfg_mode=cfg_mode,
        cache_mode=config_cache_mode,
    )
    if effective_cache_mode == "cache_dit":
        validate_cache_dit_options(
            num_steps=config.num_steps,
            solver=config.solver,
            cfg_scale=config.cfg_scale,
            cfg_mode=config.cfg_mode,
            cfg_scale_text=config.cfg_scale_text,
            cfg_scale_speaker=config.cfg_scale_speaker,
            cfg_min_t=config.cfg_min_t,
            cfg_max_t=config.cfg_max_t,
        )

    tts = RealtimeTTSInference(
        flow_model_path=checkpoint,
        dacvae_model=dacvae_model,
        dacvae_backend=dacvae_backend,
        device=device,
        compile_model=compile_model,
        compile_mode=compile_mode,
        cache_mode=effective_cache_mode,
        prefer_ema=not base_weights,
    )
    checkpoint_provenance = tts.checkpoint_provenance
    checkpoint_path = checkpoint_provenance.path
    effective_cfg = tts._effective_cfg(
        cfg_scale=config.cfg_scale,
        cfg_mode=config.cfg_mode,
        cfg_scale_text=config.cfg_scale_text,
        cfg_scale_speaker=config.cfg_scale_speaker,
        speaker_latent=torch.empty(1) if reference_audio is not None else None,
    )

    if tts.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(tts.device)

    warmups = []
    audio = None
    for index in range(warmup_runs):
        audio, timings = _run_once(
            tts,
            text=text,
            duration=duration,
            config=config,
            seed=seed + index,
            language_pair=language_pair,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
        )
        warmups.append(
            {
                "run": index + 1,
                **timings,
                "stages": StageTiming.from_complete_waveform_timings(timings).to_dict(),
            }
        )

    measurements = []
    for index in range(runs):
        audio, timings = _run_once(
            tts,
            text=text,
            duration=duration,
            config=config,
            seed=seed + warmup_runs + index,
            language_pair=language_pair,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
        )
        measurements.append(
            {
                "run": index + 1,
                **timings,
                "stages": StageTiming.from_complete_waveform_timings(timings).to_dict(),
            }
        )

    assert audio is not None
    timing_names = (
        "ttft",
        "ttfa",
        "conditioning",
        "ode_sampling",
        "decoding",
        "output_transfer",
        "total",
    )
    summary = {name: summarize([float(run[name]) for run in measurements]) for name in timing_names}
    summary["stages"] = summarize_stage_timings(
        StageTiming.from_complete_waveform_timings(run) for run in measurements
    )
    for name in (
        "cached_steps",
        "total_steps",
        "cache_ratio",
        "baseline_block_calls",
        "estimated_block_calls",
        "block_work_reduction",
    ):
        if measurements and all(name in run for run in measurements):
            summary[name] = summarize([float(run[name]) for run in measurements])

    source_hashes = package_source_hashes(Path(__file__).parent)

    loaded_codec = getattr(tts, "dacvae", None)
    loaded_codec_path = getattr(loaded_codec, "nar_vae_codec_path", None)
    dacvae_artifact = Path(loaded_codec_path) if loaded_codec_path is not None else None
    if dacvae_artifact is None and not isinstance(dacvae_model, HubDACVAESource):
        candidate = Path(requested_dacvae_model).expanduser()
        if candidate.is_dir():
            candidate = candidate / "weights.pth"
        if candidate.is_file():
            dacvae_artifact = candidate

    model_artifacts = {
        "selected": file_metadata(
            checkpoint_path,
            label=checkpoint_provenance.selected_filename,
        ),
    }
    base_checkpoint = checkpoint_provenance.base_path
    if (
        base_checkpoint is not None
        and base_checkpoint != checkpoint_path
        and base_checkpoint.is_file()
    ):
        model_artifacts["base"] = file_metadata(
            base_checkpoint,
            label=checkpoint_provenance.base_filename,
        )

    if save_audio_path is not None:
        save_audio_path.parent.mkdir(parents=True, exist_ok=True)
        tts.save_audio(audio, str(save_audio_path))

    result = {
        "schema_version": 5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": non_claim_evidence(
            result_kind="complete_waveform_benchmark",
            synthetic=False,
            hardware_measured=True,
        ),
        "definitions": {
            "ttft": (
                "Elapsed time from synthesize_fast entry through the first completed "
                "ODE latent integration step. This is not time-to-first-text-token."
            ),
            "ttfa": (
                "Elapsed time from synthesize_fast entry until the complete, "
                "non-streaming waveform is available on CPU."
            ),
            "streaming": False,
            "stage_timing": (
                "Queue is zero for this direct local call; generation maps to ode_sampling, "
                "decode to decoding, transfer to output_transfer, and packetization is zero "
                "because no independently playable packet is produced. Legacy timing fields "
                "remain unchanged."
            ),
        },
        "model": {
            "source_kind": checkpoint_provenance.kind,
            "hf_id": (
                checkpoint_provenance.source
                if checkpoint_provenance.kind == "huggingface_hub"
                else None
            ),
            "requested_source": requested_checkpoint,
            "requested_revision": checkpoint_provenance.requested_revision,
            "revision": checkpoint_provenance.resolved_revision,
            "selected_filename": checkpoint_provenance.selected_filename,
            "artifacts": model_artifacts,
            "dtype": str(next(tts.flow_model.parameters()).dtype),
            "capabilities": {
                "speaker_conditioning": tts.supports_voice_cloning,
                "multilingual": tts.supports_multilingual,
                "cross_lingual": tts.supports_cross_lingual,
                "learned_duration": getattr(tts, "uses_learned_duration", False),
                "monotonic_alignment": getattr(tts, "uses_mas_duration", False),
                "target_languages": list(tts.supported_languages),
                "reference_languages": list(tts.supported_reference_languages),
            },
        },
        "dacvae": {
            "source_kind": (
                "huggingface_hub" if isinstance(dacvae_model, HubDACVAESource) else "local"
            ),
            "hf_id": (
                dacvae_description.identifier if isinstance(dacvae_model, HubDACVAESource) else None
            ),
            "requested_source": requested_dacvae_model,
            "revision": dacvae_description.revision,
            "filename": dacvae_description.filename,
            "sha256": getattr(loaded_codec, "nar_vae_codec_sha256", None),
            "backend": dacvae_backend,
            "sample_rate": int(tts.sample_rate),
            "hop_length": int(tts.hop_length),
            "artifact": (
                file_metadata(dacvae_artifact, label="weights.pth")
                if dacvae_artifact is not None
                else None
            ),
        },
        "configuration": {
            "text": text,
            "language": language_pair.target,
            "language_pair": {
                "target": language_pair.target,
                "reference": language_pair.reference,
                "cross_lingual": language_pair.is_cross_lingual,
            },
            "reference_audio": (
                file_metadata(Path(reference_audio), label=str(reference_audio))
                if isinstance(reference_audio, (str, Path))
                else {
                    "kind": "tensor",
                    "shape": list(reference_audio.shape),
                    "sample_rate": reference_sample_rate,
                }
                if isinstance(reference_audio, torch.Tensor)
                else None
            ),
            "profile": profile,
            "requested_duration_s": duration,
            "actual_duration_s": audio.numel() / tts.sample_rate,
            "num_steps": config.num_steps,
            "solver": config.solver,
            "cfg_scale": config.cfg_scale,
            "cfg_mode": config.cfg_mode,
            "cfg_scale_text": config.cfg_scale_text,
            "cfg_scale_speaker": config.cfg_scale_speaker,
            "effective_cfg_scale": effective_cfg[0],
            "effective_cfg_mode": effective_cfg[1],
            "effective_cfg_scale_text": effective_cfg[2],
            "effective_cfg_scale_speaker": effective_cfg[3],
            "cfg_min_t": config.cfg_min_t,
            "cfg_max_t": config.cfg_max_t,
            "initial_noise_scale": config.initial_noise_scale,
            "temporal_rescale_k": config.temporal_rescale_k,
            "temporal_rescale_sigma": config.temporal_rescale_sigma,
            "target_latent_std": config.target_latent_std,
            "prefer_ema": not base_weights,
            "compile_model": compile_model,
            "compile_mode": compile_mode if compile_model else None,
            "cache_mode": effective_cache_mode,
            "warmup_runs": warmup_runs,
            "measured_runs": runs,
            "seed": seed,
        },
        "optimization": {
            "backend": effective_cache_mode,
            "cache_dit_version": tts.last_cache_stats.version,
            "cached_steps": tts.last_cache_stats.cached_steps,
            "executed_steps": tts.last_cache_stats.executed_steps,
            "cache_ratio": tts.last_cache_stats.cache_ratio,
            "baseline_block_calls": tts.last_cache_stats.baseline_block_calls,
            "estimated_block_calls": tts.last_cache_stats.estimated_block_calls,
            "block_work_reduction": tts.last_cache_stats.block_work_reduction,
        },
        "environment": environment(),
        "source_sha256": source_hashes,
        "warmup": warmups,
        "measurements": measurements,
        "summary": summary,
        "audio_validation": {
            "samples": int(audio.numel()),
            "duration_s": audio.numel() / tts.sample_rate,
            "finite": bool(torch.isfinite(audio).all()),
            "peak": float(audio.abs().max()),
            "rms": float(audio.square().mean().sqrt()),
            "artifact": (
                file_metadata(save_audio_path, label=str(save_audio_path))
                if save_audio_path is not None
                else None
            ),
        },
    }
    if tts.device.type == "cuda":
        result["environment"]["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(
            tts.device
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Results saved to {output_path}")
    print(f"TTFT median: {summary['ttft']['median_s']:.6f}s")
    print(f"TTFA median: {summary['ttfa']['median_s']:.6f}s")
    tts.close()
    return result
