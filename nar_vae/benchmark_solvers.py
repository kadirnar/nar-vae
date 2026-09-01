"""Compare meaningful-output durations for every supported ODE solver."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from nar_vae.benchmarking import (
    environment,
    file_metadata,
    package_source_hashes,
    summarize,
)
from nar_vae.checkpoint import HubCheckpointSource
from nar_vae.configuration import SOLVER_NFE_PER_STEP, SOLVERS, load_inference_settings
from nar_vae.dacvae import (
    DACVAE_BACKENDS,
    HubDACVAESource,
    describe_dacvae_source,
)
from nar_vae.inference_realtime import RealtimeTTSInference
from nar_vae.languages import DEFAULT_LANGUAGE, normalize_language
from nar_vae.quality import audio_metrics, evaluate_audio_file
from nar_vae.serving import StageTiming, non_claim_evidence, summarize_stage_timings


def _markdown_table(result: dict) -> str:
    lines = [
        "| ODE solver | Steps | NFE | Median TTFT | Median TTFA | p95 TTFA | Median RTF | WER | WER gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for solver in SOLVERS:
        entry = result["solvers"][solver]
        quality = entry.get("quality")
        wer = f"{quality['word_error_rate']:.1%}" if quality else "not run"
        if quality is None:
            passed = "not run"
        elif quality.get("passed") is None:
            passed = "threshold not set"
        else:
            passed = "pass" if quality["passed"] else "fail"
        lines.append(
            f"| {solver} | {entry['num_steps']} | {entry['nfe']} "
            f"| {entry['summary']['ttft']['median_s'] * 1000:.2f} ms "
            f"| {entry['summary']['ttfa']['median_s']:.3f} s "
            f"| {entry['summary']['ttfa']['p95_s']:.3f} s "
            f"| {entry['median_rtf']:.3f} "
            f"| {wer} | {passed} |"
        )
    if result.get("configuration", {}).get("measured_runs", 0) < 20:
        lines.extend(
            [
                "",
                "> p95 values are preliminary because fewer than 20 measured runs were used.",
            ]
        )
    return "\n".join(lines) + "\n"


def compare_solvers(
    *,
    checkpoint: str | Path | HubCheckpointSource,
    dacvae_model: str | Path | HubDACVAESource | None = None,
    dacvae_backend: str = "bundled",
    device: str = "cuda",
    text: str = "The quick brown fox jumps over the lazy dog.",
    language: str = DEFAULT_LANGUAGE,
    profile: str = "fast",
    num_steps: int | None = None,
    nfe_budget: int | None = None,
    duration: float | None = None,
    warmup_runs: int = 1,
    runs: int = 3,
    seed: int = 20260730,
    base_weights: bool = False,
    compile_model: bool = False,
    compile_mode: str = "reduce-overhead",
    evaluate_asr: bool = False,
    maximum_wer: float | None = None,
    output: str | Path = "optimizer_durations.json",
    markdown: str | Path = "optimizer_durations.md",
    audio_dir: str | Path = "solver-outputs",
) -> dict[str, object]:
    """Benchmark each supported ODE solver and return the comparison data.

    ``checkpoint`` must be a local artifact or an explicitly revision-pinned
    :class:`HubCheckpointSource`. JSON, Markdown, and solver audio files are
    written to the supplied paths. ASR transcription records WER, but sets a
    pass/fail gate only when ``maximum_wer`` is supplied explicitly.
    """
    if num_steps is not None and num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if nfe_budget is not None and nfe_budget <= 0:
        raise ValueError("nfe_budget must be positive.")
    if num_steps is not None and nfe_budget is not None:
        raise ValueError("num_steps and nfe_budget are mutually exclusive.")
    if nfe_budget is not None:
        incompatible = [
            solver
            for solver, evaluations in SOLVER_NFE_PER_STEP.items()
            if nfe_budget % evaluations
        ]
        if incompatible:
            names = ", ".join(incompatible)
            raise ValueError(f"nfe_budget must divide evenly across every solver: {names}.")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive.")
    if warmup_runs < 0 or runs <= 0:
        raise ValueError("warmup_runs must be non-negative and runs must be positive.")
    if maximum_wer is not None and not evaluate_asr:
        raise ValueError("maximum_wer requires evaluate_asr=True.")
    if maximum_wer is not None and not 0 <= maximum_wer <= 1:
        raise ValueError("maximum_wer must be between 0 and 1.")
    if dacvae_backend not in DACVAE_BACKENDS:
        raise ValueError(
            f"Unknown DACVAE backend {dacvae_backend!r}; expected one of {DACVAE_BACKENDS}."
        )
    if dacvae_model is None:
        raise ValueError(
            "dacvae_model is required; pass a local artifact or the Hugging Face ID in the "
            "checkpoint manifest."
        )

    requested_checkpoint = (
        checkpoint.repo_id if isinstance(checkpoint, HubCheckpointSource) else str(checkpoint)
    )
    language = normalize_language(language)
    dacvae_description = describe_dacvae_source(dacvae_model)
    requested_dacvae_model = dacvae_description.identifier
    output_path = Path(output)
    markdown_path = Path(markdown)
    audio_directory = Path(audio_dir)

    base_config = (
        load_inference_settings()
        .profile(profile)
        .with_overrides(num_steps=num_steps, cache_mode="none")
    )
    tts = RealtimeTTSInference(
        flow_model_path=checkpoint,
        dacvae_model=dacvae_model,
        dacvae_backend=dacvae_backend,
        device=device,
        compile_model=compile_model,
        compile_mode=compile_mode,
        cache_mode="none",
        prefer_ema=not base_weights,
    )
    checkpoint_provenance = tts.checkpoint_provenance
    checkpoint_path = checkpoint_provenance.path
    effective_cfg = tts._effective_cfg(
        cfg_scale=base_config.cfg_scale,
        cfg_mode=base_config.cfg_mode,
        cfg_scale_text=base_config.cfg_scale_text,
        cfg_scale_speaker=base_config.cfg_scale_speaker,
        speaker_latent=None,
    )
    audio_directory.mkdir(parents=True, exist_ok=True)
    solver_results = {}

    for solver in SOLVERS:
        evaluations_per_step = SOLVER_NFE_PER_STEP[solver]
        solver_steps = (
            nfe_budget // evaluations_per_step if nfe_budget is not None else base_config.num_steps
        )
        config = base_config.with_overrides(solver=solver, num_steps=solver_steps)
        warmups = []
        measurements = []
        audio = None
        for index in range(warmup_runs + runs):
            run_seed = seed + index
            torch.manual_seed(run_seed)
            if tts.device.type == "cuda":
                torch.cuda.manual_seed_all(run_seed)
                torch.cuda.synchronize(tts.device)
            audio, timings = tts.synthesize_fast(
                text,
                config=config,
                duration=duration,
                return_timing=True,
                language=language,
            )
            row = {"run": index + 1, **timings}
            row["stages"] = StageTiming.from_complete_waveform_timings(timings).to_dict()
            if index < warmup_runs:
                warmups.append(row)
            else:
                row["run"] = index - warmup_runs + 1
                measurements.append(row)

        assert audio is not None
        audio_path = audio_directory / f"{solver}.wav"
        tts.save_audio(audio, str(audio_path))
        timing_names = (
            "ttft",
            "ttfa",
            "conditioning",
            "ode_sampling",
            "decoding",
            "output_transfer",
            "total",
        )
        summary = {
            name: summarize([float(row[name]) for row in measurements]) for name in timing_names
        }
        summary["stages"] = summarize_stage_timings(
            StageTiming.from_complete_waveform_timings(row) for row in measurements
        )
        median_rtf = summary["ttfa"]["median_s"] / (audio.numel() / tts.sample_rate)
        quality = (
            evaluate_audio_file(
                audio_path,
                text,
                device=device,
                maximum_wer=maximum_wer,
                language=language,
            )
            if evaluate_asr
            else None
        )
        solver_results[solver] = {
            "num_steps": solver_steps,
            "nfe": solver_steps * evaluations_per_step,
            "warmup": warmups,
            "measurements": measurements,
            "summary": summary,
            "median_rtf": median_rtf,
            "audio_path": str(audio_path),
            "audio_artifact": file_metadata(audio_path, label=str(audio_path)),
            "audio": audio_metrics(audio, tts.sample_rate),
            "quality": quality,
        }
        print(f"{solver}: median TTFA={summary['ttfa']['median_s']:.3f}s, RTF={median_rtf:.3f}")

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

    loaded_codec = getattr(tts, "dacvae", None)
    resolved_dacvae_source = getattr(tts, "dacvae_source", dacvae_model)
    resolved_dacvae_description = describe_dacvae_source(resolved_dacvae_source)
    loaded_codec_path = getattr(loaded_codec, "nar_vae_codec_path", None)
    dacvae_artifact = Path(loaded_codec_path) if loaded_codec_path is not None else None
    if dacvae_artifact is None and not isinstance(resolved_dacvae_source, HubDACVAESource):
        candidate = Path(requested_dacvae_model).expanduser()
        if candidate.is_dir():
            candidate = candidate / "weights.pth"
        if candidate.is_file():
            dacvae_artifact = candidate

    source_hashes = package_source_hashes(Path(__file__).parent)

    result = {
        "schema_version": 5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence": non_claim_evidence(
            result_kind="complete_waveform_solver_comparison",
            synthetic=False,
            hardware_measured=True,
        ),
        "definitions": {
            "ttft": "Internal first completed ODE step; not playable audio.",
            "ttfa": "Complete non-streaming waveform availability on CPU.",
            "streaming": False,
            "wer_gate": (
                "WER is reported when ASR evaluation runs. A quality pass/fail result exists "
                "only when maximum_wer is explicitly configured."
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
                "huggingface_hub"
                if isinstance(resolved_dacvae_source, HubDACVAESource)
                else "local"
            ),
            "hf_id": (
                resolved_dacvae_description.identifier
                if isinstance(resolved_dacvae_source, HubDACVAESource)
                else None
            ),
            "requested_source": requested_dacvae_model,
            "revision": resolved_dacvae_description.revision,
            "filename": resolved_dacvae_description.filename,
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
            "language": language,
            "language_pair": {
                "target": language,
                "reference": None,
                "cross_lingual": False,
            },
            "profile": profile,
            "num_steps": base_config.num_steps if nfe_budget is None else None,
            "nfe_budget": nfe_budget,
            "requested_duration_s": duration,
            "warmup_runs": warmup_runs,
            "measured_runs": runs,
            "seed": seed,
            "weights": "base" if base_weights else "ema",
            "compile_model": compile_model,
            "compile_mode": compile_mode if compile_model else None,
            "evaluate_asr": evaluate_asr,
            "maximum_wer": maximum_wer,
            "same_step_count_for_all_solvers": nfe_budget is None,
            "requested_cfg_scale": base_config.cfg_scale,
            "requested_cfg_mode": base_config.cfg_mode,
            "requested_cfg_scale_text": base_config.cfg_scale_text,
            "requested_cfg_scale_speaker": base_config.cfg_scale_speaker,
            "cfg_min_t": base_config.cfg_min_t,
            "cfg_max_t": base_config.cfg_max_t,
            "initial_noise_scale": base_config.initial_noise_scale,
            "temporal_rescale_k": base_config.temporal_rescale_k,
            "temporal_rescale_sigma": base_config.temporal_rescale_sigma,
            "target_latent_std": base_config.target_latent_std,
            "effective_cfg_scale": effective_cfg[0],
            "effective_cfg_mode": effective_cfg[1],
            "effective_cfg_scale_text": effective_cfg[2],
            "effective_cfg_scale_speaker": effective_cfg[3],
        },
        "environment": environment(),
        "source_sha256": source_hashes,
        "solvers": solver_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown_table(result), encoding="utf-8")
    tts.close()
    print(f"Saved JSON to {output_path}")
    print(f"Saved table to {markdown_path}")
    return result
