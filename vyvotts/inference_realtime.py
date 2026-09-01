"""Low-step inference built on the shared NAR-VAE runtime."""

from __future__ import annotations

import time
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import TracebackType

import torch

from vyvotts.caching import (
    CacheDiTRequestActiveError,
    CacheDiTSession,
    CacheDiTStats,
    assert_cache_dit_healthy,
    create_scm_context,
)
from vyvotts.checkpoint import HubCheckpointSource
from vyvotts.configuration import GenerationConfig, validate_cache_dit_options
from vyvotts.dacvae import HubDACVAESource
from vyvotts.inference import AudioReference, FlowMatchingTTSInference
from vyvotts.solvers.ode_solver import ODESolver


def _mark_compiled_cuda_graph_step() -> None:
    """Mark one logical request or fail clearly on an unsupported PyTorch build."""
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if not callable(marker):
        raise RuntimeError(
            "Compiled CUDA inference requires torch>=2.2 with "
            "torch.compiler.cudagraph_mark_step_begin()."
        )
    marker()


class RealtimeTTSInference(FlowMatchingTTSInference):
    """Shared TTS runtime with low-step profiles and synchronized latency data."""

    def __init__(
        self,
        flow_model_path: str | Path | HubCheckpointSource,
        dacvae_model: str | Path | HubDACVAESource | None = None,
        dacvae_backend: str = "bundled",
        device: str = "cuda",
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        latent_size: int = 128,
        model_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        text_vocab_size: int | None = None,
        text_num_layers: int = 6,
        speaker_patch_size: int | None = None,
        speaker_model_size: int = 512,
        speaker_num_layers: int = 4,
        speaker_num_heads: int = 8,
        speaker_intermediate_size: int = 2048,
        use_speaker_conditioning: bool | None = None,
        cache_mode: str | None = None,
        prefer_ema: bool = True,
        max_reference_seconds: float = 30.0,
        use_language_conditioning: bool | None = None,
        supported_languages: tuple[str, ...] | list[str] | None = None,
        text_model_size: int = 768,
        text_num_heads: int = 12,
        text_intermediate_size: int = 3072,
        timestep_embed_size: int = 256,
        adaln_rank: int = 128,
        norm_eps: float = 1e-6,
        use_duration_predictor: bool | None = None,
    ):
        super().__init__(
            flow_model_path=flow_model_path,
            dacvae_model=dacvae_model,
            dacvae_backend=dacvae_backend,
            device=device,
            latent_size=latent_size,
            model_size=model_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            text_vocab_size=text_vocab_size,
            text_num_layers=text_num_layers,
            speaker_patch_size=speaker_patch_size,
            speaker_model_size=speaker_model_size,
            speaker_num_layers=speaker_num_layers,
            speaker_num_heads=speaker_num_heads,
            speaker_intermediate_size=speaker_intermediate_size,
            use_speaker_conditioning=use_speaker_conditioning,
            prefer_ema=prefer_ema,
            max_reference_seconds=max_reference_seconds,
            use_language_conditioning=use_language_conditioning,
            use_duration_predictor=use_duration_predictor,
            supported_languages=supported_languages,
            text_model_size=text_model_size,
            text_num_heads=text_num_heads,
            text_intermediate_size=text_intermediate_size,
            timestep_embed_size=timestep_embed_size,
            adaln_rank=adaln_rank,
            norm_eps=norm_eps,
        )
        self.compile_model = False
        self.compile_mode = compile_mode
        self.cache_mode = cache_mode
        self._compiled_cache_session: CacheDiTSession | None = None
        self._uncompiled_backbone: torch.nn.Module | None = None
        self._uncompiled_decode = None
        self._cache_finalizer: weakref.finalize | None = None
        self._compiled_cache_failed = False

        if compile_model:
            self._enable_compilation(compile_mode)

    def _enable_compilation(self, compile_mode: str) -> None:
        """Compile EchoDiT and the decoder after installing cache hooks when requested."""
        assert_cache_dit_healthy(self.flow_model)
        print(f"\nPreparing EchoDiT and decoder for torch.compile (mode={compile_mode})...")
        original_backbone = self.flow_model.dit
        original_decode = self._decode
        cache_session = None
        try:
            backbone = original_backbone
            if self.cache_mode == "cache_dit":
                cache_session = CacheDiTSession(
                    self.flow_model,
                    num_steps=self.generation_profile("turbo").num_steps,
                )
                cache_session.__enter__()
                set_compile_configs = getattr(cache_session.api, "set_compile_configs", None)
                if not callable(set_compile_configs):
                    raise RuntimeError(
                        "The installed Cache-DiT version does not expose set_compile_configs()."
                    )
                set_compile_configs(
                    cuda_graphs=compile_mode in ("reduce-overhead", "max-autotune"),
                    use_fast_math=False,
                )
                backbone = cache_session.backbone

            compiled_backbone = torch.compile(
                backbone,
                mode=compile_mode,
                dynamic=False,
            )
            compiled_decode = torch.compile(
                original_decode,
                dynamic=False,
                options={
                    # Cache-DiT's coordinate-descent search is valuable for the
                    # repeated DiT blocks but prohibitively expensive for the
                    # convolutional decoder. Keep its fixed-shape CUDA Graph.
                    "triton.cudagraphs": compile_mode != "max-autotune-no-cudagraphs",
                    "coordinate_descent_tuning": False,
                    "coordinate_descent_check_all_directions": False,
                },
            )
            self.flow_model.dit = compiled_backbone
            self._decode = compiled_decode
            self._uncompiled_backbone = original_backbone
            self._uncompiled_decode = original_decode
            self._compiled_cache_session = cache_session
            if cache_session is not None:
                self._cache_finalizer = weakref.finalize(self, cache_session.close)
            self.compile_model = True
            print(
                "  Compilation is lazy; warm up with the same text and duration shape used in production."
            )
        except Exception as exc:
            cleanup_error: BaseException | None = None
            try:
                if cache_session is not None:
                    cache_session.close()
            except BaseException as error:
                cleanup_error = error
            finally:
                # Compilation and hook installation can both mutate these
                # references. Restore the eager runtime even when backend
                # teardown raises, while keeping the triggering error primary.
                self.flow_model.dit = original_backbone
                self._decode = original_decode
                self.compile_model = False
            failure = RuntimeError(
                "torch.compile could not prepare the EchoDiT backbone. "
                "Run without compile_model to use eager execution."
            )
            if cleanup_error is not None:
                note = f"Cache-DiT cleanup also failed: {cleanup_error}"
                add_note = getattr(failure, "add_note", None)
                if callable(add_note):
                    add_note(note)
                else:  # Python 3.10 compatibility
                    failure.args = (f"{failure.args[0]} {note}",)
            raise failure from exc

    def close(self) -> None:
        """Release persistent Cache-DiT hooks and restore eager modules."""
        cleanup_error: BaseException | None = None
        try:
            if self._compiled_cache_session is not None:
                self._compiled_cache_session.close()
        except CacheDiTRequestActiveError:
            # The compiled backbone still owns live backend hooks. Preserve
            # every reference so the caller can close again after inference.
            raise
        except BaseException as error:
            cleanup_error = error
        if self._uncompiled_backbone is not None:
            self.flow_model.dit = self._uncompiled_backbone
        if self._uncompiled_decode is not None:
            self._decode = self._uncompiled_decode
        if self._cache_finalizer is not None and self._cache_finalizer.alive:
            self._cache_finalizer.detach()
        self._compiled_cache_session = None
        self._uncompiled_backbone = None
        self._uncompiled_decode = None
        self._cache_finalizer = None
        self._compiled_cache_failed = False
        self.compile_model = False
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self) -> RealtimeTTSInference:
        """Return this runtime for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release persistent optimization hooks when leaving the context."""
        self.close()

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        num_steps: int = 64,
        solver: str = "heun",
        cfg_scale: float = 1.0,
        cfg_mode: str = "joint",
        cfg_scale_text: float | None = None,
        cfg_scale_speaker: float | None = None,
        cfg_min_t: float = 0.0,
        cfg_max_t: float = 1.0,
        initial_noise_scale: float = 1.0,
        temporal_rescale_k: float = 1.0,
        temporal_rescale_sigma: float = 2.5,
        target_latent_std: float | None = None,
        cache_mode: str | None = None,
        duration: float | None = None,
        show_progress: bool = True,
        reference_audio: AudioReference | None = None,
        reference_sample_rate: int | None = None,
        speaker_latent: torch.Tensor | None = None,
        language: str | None = None,
        reference_language: str | None = None,
    ) -> torch.Tensor:
        """Use the managed realtime lifecycle for the conventional synthesis API."""
        del show_progress
        return self.synthesize_fast(
            text,
            num_steps=num_steps,
            solver=solver,
            cfg_scale=cfg_scale,
            cfg_mode=cfg_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            initial_noise_scale=initial_noise_scale,
            target_latent_std=target_latent_std,
            temporal_rescale_k=temporal_rescale_k,
            temporal_rescale_sigma=temporal_rescale_sigma,
            duration=duration,
            cache_mode=cache_mode,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            speaker_latent=speaker_latent,
            language=language,
            reference_language=reference_language,
        )

    def synthesize_with_config(
        self,
        text: str,
        config: GenerationConfig,
        *,
        duration: float | None = None,
        show_progress: bool = True,
        reference_audio: AudioReference | None = None,
        reference_sample_rate: int | None = None,
        speaker_latent: torch.Tensor | None = None,
        language: str | None = None,
        reference_language: str | None = None,
    ) -> torch.Tensor:
        """Apply a typed profile without bypassing compiled cache request state."""
        del show_progress
        return self.synthesize_fast(
            text,
            config=config,
            duration=duration,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            speaker_latent=speaker_latent,
            language=language,
            reference_language=reference_language,
        )

    @torch.no_grad()
    def synthesize_batch(
        self,
        texts: list[str],
        num_steps: int = 32,
        cfg_scale: float = 1.0,
        solver: str = "euler",
        max_duration: float | None = None,
        languages: str | list[str] | tuple[str, ...] | None = None,
    ) -> list[torch.Tensor]:
        """Batch only when no persistent single-request Cache-DiT session is installed."""
        assert_cache_dit_healthy(self.flow_model)
        if getattr(self, "_compiled_cache_failed", False):
            raise RuntimeError(
                "The compiled Cache-DiT runtime was invalidated by a failed request. "
                "Call close() to restore eager inference, or construct a new runtime."
            )
        if getattr(self, "_compiled_cache_session", None) is not None:
            raise RuntimeError(
                "Compiled Cache-DiT currently supports one utterance per managed request; "
                "synthesize_batch cannot safely share its persistent cache state. Use a "
                "separate uncached runtime for exact-length batching."
            )
        if self.compile_model and self.device.type == "cuda":
            _mark_compiled_cuda_graph_step()
        return super().synthesize_batch(
            texts,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            solver=solver,
            max_duration=max_duration,
            languages=languages,
        )

    @torch.no_grad()
    def synthesize_fast(
        self,
        text: str,
        *,
        config: GenerationConfig | None = None,
        num_steps: int | None = None,
        solver: str | None = None,
        cfg_scale: float | None = None,
        cfg_mode: str | None = None,
        cfg_scale_text: float | None = None,
        cfg_scale_speaker: float | None = None,
        cfg_min_t: float | None = None,
        cfg_max_t: float | None = None,
        initial_noise_scale: float | None = None,
        target_latent_std: float | None = None,
        temporal_rescale_k: float | None = None,
        temporal_rescale_sigma: float | None = None,
        duration: float | None = None,
        return_timing: bool = False,
        cache_mode: str | None = None,
        predictor_order: int = 1,
        reference_audio: AudioReference | None = None,
        reference_sample_rate: int | None = None,
        speaker_latent: torch.Tensor | None = None,
        language: str | None = None,
        reference_language: str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Synthesize with a configured low-step profile and optional timings."""
        # Cache metrics are request-local. A failed request must not leave the
        # previous request's successful statistics visible to callers.
        self.last_cache_stats = CacheDiTStats()
        assert_cache_dit_healthy(self.flow_model)
        if getattr(self, "_compiled_cache_failed", False):
            raise RuntimeError(
                "The compiled Cache-DiT runtime was invalidated by a failed request. "
                "Call close() to restore eager inference, or construct a new runtime."
            )
        profile_config = config or self.generation_profile("fast")
        effective_cache_mode = (
            cache_mode
            if cache_mode is not None
            else self.cache_mode
            if self.cache_mode is not None
            else profile_config.cache_mode
        )
        config_cache_mode = (
            effective_cache_mode if effective_cache_mode in ("none", "cache_dit") else "none"
        )
        selected = profile_config.with_overrides(
            num_steps=num_steps,
            solver=solver,
            cfg_scale=cfg_scale,
            cfg_mode=cfg_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            initial_noise_scale=initial_noise_scale,
            target_latent_std=target_latent_std,
            temporal_rescale_k=temporal_rescale_k,
            temporal_rescale_sigma=temporal_rescale_sigma,
            cache_mode=config_cache_mode,
        )
        compiled_cache = getattr(self, "_compiled_cache_session", None)
        if (
            self.compile_model
            and compiled_cache is not None
            and effective_cache_mode != "cache_dit"
        ):
            raise RuntimeError(
                "A runtime compiled with Cache-DiT hooks cannot bypass them per request. "
                "Construct a separate compiled runtime with cache_mode='none'."
            )

        if return_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        request_started_at = time.perf_counter()
        timings: dict[str, float] = {}

        conditioning_started_at = request_started_at
        language_pair = self._resolve_language_pair(
            language,
            reference_language,
            has_reference=reference_audio is not None or speaker_latent is not None,
        )
        conditioning_ids = self._prepare_conditioning(text, language_pair.target)
        language_ids = self._language_ids(language_pair)
        resolved_speaker = self._resolve_speaker_latent(
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            speaker_latent=speaker_latent,
        )

        effective_cfg = self._effective_cfg(
            cfg_scale=selected.cfg_scale,
            cfg_mode=selected.cfg_mode,
            cfg_scale_text=selected.cfg_scale_text,
            cfg_scale_speaker=selected.cfg_scale_speaker,
            speaker_latent=resolved_speaker,
        )
        effective_cfg_scale, effective_cfg_mode, text_scale, speaker_scale = effective_cfg
        encoded_conditioning, predicted_frames, expected_token_durations = (
            self._encode_trajectory_conditioning(
                conditioning_ids,
                conditioning_mask=None,
                language_ids=language_ids,
                speaker_latent=resolved_speaker,
                cfg_scale=effective_cfg_scale,
                cfg_mode=effective_cfg_mode,
                cfg_scale_text=text_scale,
                cfg_scale_speaker=speaker_scale,
                needs_learned_duration=(
                    duration is None and getattr(self, "uses_learned_duration", False)
                ),
            )
        )
        estimated_duration, num_frames = self._resolve_duration_shape(
            text,
            duration,
            conditioning_ids,
            language_ids,
            resolved_speaker,
            predicted_frames,
        )
        token_durations = self._resolve_token_durations(
            conditioning_ids,
            num_frames=num_frames,
            language_ids=language_ids,
            speaker_latent=resolved_speaker,
            expected_token_durations=expected_token_durations,
        )
        latent_shape = (1, self.latent_size, num_frames)

        scm_ctx = None
        cache_dit_context = nullcontext(None)
        if effective_cache_mode in ("fast", "ultra"):
            if selected.solver != "euler":
                raise ValueError("Solver-level caching requires the Euler solver.")
            scm_ctx = create_scm_context(
                mask_policy=effective_cache_mode,
                num_steps=selected.num_steps,
                predictor_order=predictor_order,
            )
        elif effective_cache_mode == "cache_dit":
            validate_cache_dit_options(
                num_steps=selected.num_steps,
                solver=selected.solver,
                cfg_scale=effective_cfg_scale,
                cfg_mode=effective_cfg_mode,
                cfg_scale_text=text_scale,
                cfg_scale_speaker=speaker_scale,
                cfg_min_t=selected.cfg_min_t,
                cfg_max_t=selected.cfg_max_t,
            )
            if self.compile_model and compiled_cache is None:
                raise RuntimeError(
                    "A compiled turbo request requires constructing RealtimeTTSInference with "
                    "cache_mode='cache_dit' so hooks are installed before torch.compile."
                )
            cache_dit_context = (
                compiled_cache.request(selected.num_steps)
                if compiled_cache is not None
                else CacheDiTSession(
                    self.flow_model,
                    num_steps=selected.num_steps,
                )
            )
        elif effective_cache_mode != "none":
            raise ValueError("cache_mode must be 'none', 'cache_dit', 'fast', or 'ultra'.")

        if (
            self.compile_model
            and self.device.type == "cuda"
            and effective_cache_mode != "cache_dit"
        ):
            # One synthesis request invokes the compiled backbone many times.
            # Tell CUDA Graph Trees that those calls belong to the same logical
            # inference iteration so live solver-stage tensors are not reused.
            # Cache-DiT owns its graph lifecycle when its hooks are active.
            _mark_compiled_cuda_graph_step()

        ode_started_at: float | None = None
        first_step_finished_at = None
        first_step_recorded = False
        ode_start_event = None
        first_step_event = None
        if return_timing and self.device.type == "cuda":
            ode_start_event = torch.cuda.Event(enable_timing=True)
            first_step_event = torch.cuda.Event(enable_timing=True)

        def record_integration_start() -> None:
            """Close conditioning after invariant encoders, immediately before ODE work."""
            nonlocal ode_started_at
            if ode_started_at is not None:
                return
            if self.device.type == "cuda":
                # The duration heads and invariant encoders enqueue device work.
                # Synchronize at the stage boundary so neither stage borrows
                # latency from the other in reported measurements.
                torch.cuda.synchronize(self.device)
                if ode_start_event is not None:
                    ode_start_event.record()
            ode_started_at = time.perf_counter()
            timings["conditioning"] = ode_started_at - conditioning_started_at

        def record_first_step(step_idx: int, latents: torch.Tensor) -> None:
            del latents
            nonlocal first_step_finished_at, first_step_recorded
            if step_idx != 0 or first_step_recorded:
                return
            if self.device.type == "cuda":
                assert first_step_event is not None
                first_step_event.record()
            else:
                first_step_finished_at = time.perf_counter()
            first_step_recorded = True

        try:
            with cache_dit_context as cache_dit_session:
                prepared_conditioning, prepared_cfg_conditioning = (
                    self._finalize_trajectory_conditioning(
                        encoded_conditioning,
                        token_durations,
                    )
                )
                generated_latents = ODESolver.sample(
                    model=self.flow_model,
                    conditioning_ids=conditioning_ids,
                    num_steps=selected.num_steps,
                    latent_shape=latent_shape,
                    solver=selected.solver,
                    cfg_scale=effective_cfg_scale,
                    cfg_mode=effective_cfg_mode,
                    cfg_scale_text=text_scale,
                    cfg_scale_speaker=speaker_scale,
                    cfg_min_t=selected.cfg_min_t,
                    cfg_max_t=selected.cfg_max_t,
                    initial_noise_scale=selected.initial_noise_scale,
                    temporal_rescale_k=selected.temporal_rescale_k,
                    temporal_rescale_sigma=selected.temporal_rescale_sigma,
                    target_latent_std=selected.target_latent_std,
                    conditioning_mask=None,
                    speaker_latent=resolved_speaker,
                    language_ids=language_ids,
                    token_durations=token_durations,
                    prepared_conditioning=prepared_conditioning,
                    prepared_cfg_conditioning=prepared_cfg_conditioning,
                    fuse_cfg_branches=self.compile_model or effective_cache_mode == "cache_dit",
                    show_progress=False,
                    device=self.device,
                    scm_ctx=scm_ctx,
                    step_callback=record_first_step if return_timing else None,
                    integration_start_callback=(
                        record_integration_start if return_timing else None
                    ),
                )
        except BaseException:
            if getattr(self, "_compiled_cache_session", None) is not None:
                self._compiled_cache_failed = True
            raise
        if return_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        # Test doubles and older compatible solver shims may not expose the
        # explicit boundary. Preserve complete accounting in that case.
        if ode_started_at is None:
            ode_started_at = time.perf_counter()
            timings["conditioning"] = ode_started_at - conditioning_started_at

        if scm_ctx is not None:
            timings["cached_steps"] = float(scm_ctx.cached_steps)
            timings["total_steps"] = float(scm_ctx.total_steps)
            timings["cache_ratio"] = scm_ctx.cache_ratio
            self.last_cache_stats = CacheDiTStats(
                cached_steps=scm_ctx.cached_steps,
                executed_steps=scm_ctx.total_steps,
            )
        elif isinstance(cache_dit_session, CacheDiTSession):
            self.last_cache_stats = cache_dit_session.stats
            timings["cached_steps"] = float(cache_dit_session.stats.cached_steps)
            timings["total_steps"] = float(cache_dit_session.stats.executed_steps)
            timings["cache_ratio"] = cache_dit_session.stats.cache_ratio
            timings["baseline_block_calls"] = float(cache_dit_session.stats.baseline_block_calls)
            timings["estimated_block_calls"] = float(cache_dit_session.stats.estimated_block_calls)
            timings["block_work_reduction"] = cache_dit_session.stats.block_work_reduction
        else:
            self.last_cache_stats = CacheDiTStats()

        # This global-flow model emits no streamable token or audio before
        # integration finishes. TTFT is an internal first-step milestone.
        if ode_start_event is not None and first_step_event is not None and first_step_recorded:
            timings["ttft"] = (
                ode_started_at
                - request_started_at
                + ode_start_event.elapsed_time(first_step_event) / 1000.0
            )
        elif first_step_finished_at is not None:
            timings["ttft"] = first_step_finished_at - request_started_at

        decode_started_at = time.perf_counter()
        assert ode_started_at is not None
        timings["ode_sampling"] = decode_started_at - ode_started_at
        audio = self._decode(generated_latents)
        if return_timing and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        transfer_started_at = time.perf_counter()
        timings["decoding"] = transfer_started_at - decode_started_at
        audio = audio.squeeze().cpu()
        request_finished_at = time.perf_counter()
        timings["output_transfer"] = request_finished_at - transfer_started_at
        timings["ttfa"] = request_finished_at - request_started_at
        timings["total"] = timings["ttfa"]
        timings["audio_duration"] = audio.numel() / self.sample_rate

        if return_timing:
            return audio, timings
        return audio
