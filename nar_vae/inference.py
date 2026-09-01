import math
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from nar_vae.caching import CacheDiTSession, CacheDiTStats, assert_cache_dit_healthy
from nar_vae.checkpoint import (
    FlowCheckpoint,
    HubCheckpointSource,
)
from nar_vae.configuration import (
    GenerationConfig,
    cfg_guidance_active,
    load_inference_settings,
    validate_cache_dit_options,
)
from nar_vae.dacvae import HubDACVAESource, load_dacvae, normalize_dacvae_source
from nar_vae.dacvae_encoding import (
    derive_dacvae_posterior_seed,
    encode_dacvae_posterior_seeded,
)
from nar_vae.frozen_text_provider import FrozenTextProvider, FrozenTextProviderSpec
from nar_vae.languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_COUNT,
    CrossLingualUnsupportedError,
    LanguagePair,
    MultilingualUnsupportedError,
    language_id,
    normalize_languages,
)
from nar_vae.model_manifest import (
    MODEL_MANIFEST_FILENAME,
    load_model_manifest,
    validate_inference_manifest,
    validate_loaded_codec,
    validate_manifest_weight,
)
from nar_vae.model_presets import get_model_preset
from nar_vae.models.duration import ECHODIT_ARCHITECTURE_VERSION
from nar_vae.models.flow_matching import create_flow_matching_echodit
from nar_vae.objectives import (
    DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    RECTIFIED_FLOW_OBJECTIVE,
    VP_DIFFUSION_OBJECTIVE,
    normalize_generative_objective,
    validate_diffusion_schedule_shift,
)
from nar_vae.solvers.ode_solver import ODESolver
from nar_vae.tokenization import (
    PAD_TOKEN,
    TOKENIZER_NAME,
    TOTAL_VOCAB_SIZE,
    TextSpan,
    encode_tts_conditioning,
)
from nar_vae.voice import DEFAULT_MAX_REFERENCE_SECONDS

AudioReference = str | Path | torch.Tensor

_ACOUSTIC_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _normalize_acoustic_dtype(value: str | torch.dtype) -> torch.dtype:
    """Resolve the two inference precisions validated by the public runtime."""
    if value is torch.float32 or value is torch.bfloat16:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            return _ACOUSTIC_DTYPE_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError(
                "acoustic_dtype must be 'float32'/'fp32' or 'bfloat16'/'bf16'."
            ) from exc
    raise ValueError("acoustic_dtype must be a supported dtype name or torch.dtype.")


def _cast_acoustic_parameters(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast learned floating state without applying a dtype transform to buffers."""
    dtype = _normalize_acoustic_dtype(dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.is_floating_point(parameter):
                parameter.data = parameter.detach().to(
                    device=parameter.device,
                    dtype=dtype,
                )


def _runtime_generation_contract(runtime: object) -> tuple[str, float]:
    """Resolve objective metadata for compatible custom/test runtimes.

    Authenticated runtimes always set these values from their checkpoint manifest in
    ``__init__``.  Lightweight integrations that construct the runtime protocol via
    ``__new__`` may instead expose them on the acoustic model; complete absence is the
    unambiguous legacy rectified-flow contract.
    """
    model = getattr(runtime, "flow_model", None)
    objective = getattr(runtime, "generative_objective", None)
    if objective is None:
        objective = getattr(model, "generative_objective", RECTIFIED_FLOW_OBJECTIVE)
    schedule_shift = getattr(runtime, "diffusion_schedule_shift", None)
    if schedule_shift is None:
        schedule_shift = getattr(
            model,
            "diffusion_schedule_shift",
            DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
        )
    return (
        normalize_generative_objective(objective),
        validate_diffusion_schedule_shift(schedule_shift),
    )


@dataclass(frozen=True)
class _PreparedText:
    conditioning_ids: torch.Tensor
    conditioning_mask: torch.Tensor | None
    conditioning_features: torch.Tensor | None
    token_language_ids: torch.Tensor
    alignment_mask: torch.Tensor


class VoiceCloningUnsupportedError(RuntimeError):
    """Raised when reference audio is used with a text-only checkpoint."""


class LearnedDurationUnsupportedError(RuntimeError):
    """Raised when learned duration is requested without versioned trained weights."""


class FlowMatchingTTSInference:
    """
    Inference pipeline for versioned NAR-VAE TTS checkpoints.

    The canonical path uses:
    1. Reviewed provider-native phones and the pinned frozen multilingual provider
    2. A small learned adapter plus language/reference conditioning
    3. Strict VP diffusion sampled with analytic DDIM or an authenticated ODE trajectory
    4. The unchanged DACVAE decoder (continuous latents → audio)

    The compact-token frontend and built-in text Transformer remain available only for
    explicitly identified legacy scratch-token checkpoints.

    Args:
        flow_model_path: Path to trained EchoDiT checkpoint
        dacvae_model: Local codec artifact or Hugging Face repository ID
        dacvae_backend: DACVAE implementation ("bundled", "fast", or "auto")
        device: Device to run on (default: "cuda")
        latent_size: DACVAE latent dimension (default: 128)
        model_size: EchoDiT hidden size (default: 1024 for the xlarge topology)
        num_layers: Number of EchoDiT layers (default: 24 for xlarge)
        num_heads: Number of attention heads (default: 16 for xlarge)
        intermediate_size: MLP intermediate size (default: 4096 for xlarge)
        text_vocab_size: Vocabulary size (default: inferred from checkpoint)
        text_num_layers: Text encoder layers (default: 6)
        target_patch_size: Target-latent packing factor (default: inferred from checkpoint)
        speaker_patch_size: Speaker encoder patch size (inferred for clone-capable checkpoints)
        speaker_model_size: Speaker encoder hidden size (default: 512)
        speaker_num_layers: Speaker encoder layers (default: 4)
        speaker_num_heads: Speaker encoder attention heads (default: 8)
        speaker_intermediate_size: Speaker encoder MLP size (default: 2048)
        speaker_num_summary_tokens: Fixed reference token count (default: inferred)
        use_speaker_conditioning: Build speaker-conditioning state (default: inferred)
        use_duration_predictor: Use learned duration only when checkpoint metadata validates it
        acoustic_dtype: Acoustic EchoDiT precision: float32/fp32 or bfloat16/bf16. The immutable
            DACVAE and all acoustic-model buffers retain their authenticated dtypes.
    """

    def __init__(
        self,
        flow_model_path: str | Path | HubCheckpointSource,
        dacvae_model: str | Path | HubDACVAESource | None = None,
        dacvae_backend: str = "bundled",
        device: str = "cuda",
        # EchoDiT model config (must match training) - xlarge topology defaults
        latent_size: int = 128,
        model_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        text_vocab_size: int | None = None,
        text_num_layers: int | None = None,
        target_patch_size: int | None = None,
        # Speaker encoder config (must match training)
        speaker_patch_size: int | None = None,  # Inferred for speaker checkpoints
        speaker_model_size: int = 512,
        speaker_num_layers: int = 4,
        speaker_num_heads: int = 8,
        speaker_intermediate_size: int = 2048,
        use_speaker_conditioning: bool | None = None,
        prefer_ema: bool = True,
        max_reference_seconds: float = DEFAULT_MAX_REFERENCE_SECONDS,
        use_language_conditioning: bool | None = None,
        supported_languages: tuple[str, ...] | list[str] | None = None,
        text_model_size: int = 768,
        text_num_heads: int = 12,
        text_intermediate_size: int = 3072,
        timestep_embed_size: int = 256,
        adaln_rank: int = 128,
        norm_eps: float = 1e-6,
        use_duration_predictor: bool | None = None,
        frozen_text_provider: object | None = None,
        speaker_num_summary_tokens: int | None = None,
        acoustic_dtype: str | torch.dtype = "float32",
    ):
        self.acoustic_dtype = _normalize_acoustic_dtype(acoustic_dtype)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.latent_size = latent_size
        self.max_reference_seconds = max_reference_seconds
        self.settings = load_inference_settings()
        self.last_cache_stats = CacheDiTStats()
        preloaded_manifest = None

        def validate_before_deserialization(provenance):
            nonlocal preloaded_manifest
            manifest_path = provenance.manifest_path
            if manifest_path is None:
                manifest_path = provenance.path.parent / MODEL_MANIFEST_FILENAME
            preloaded_manifest = load_model_manifest(manifest_path)
            validate_manifest_weight(
                preloaded_manifest,
                provenance.path,
                selected_filename=provenance.selected_filename,
            )
            if provenance.ema_filename is not None:
                if provenance.base_path is None:
                    raise FileNotFoundError(
                        "A partial EMA checkpoint requires its full manifest-bound base "
                        f"checkpoint {provenance.base_filename!r}."
                    )
                validate_manifest_weight(
                    preloaded_manifest,
                    provenance.base_path,
                    selected_filename=provenance.base_filename,
                )

        checkpoint = FlowCheckpoint.load(
            flow_model_path,
            prefer_ema=prefer_ema,
            preload_validator=validate_before_deserialization,
        )
        checkpoint.validate_architecture()
        checkpoint_objective = checkpoint.generative_objective()
        checkpoint_schedule_shift = checkpoint.diffusion_schedule_shift()
        checkpoint_text_conditioning = checkpoint.text_conditioning()
        self.generative_objective = checkpoint_objective
        self.diffusion_schedule_shift = checkpoint_schedule_shift
        self.text_conditioning_mode = checkpoint_text_conditioning.mode
        checkpoint_target_patch_size = checkpoint.infer_target_patch_size()
        if target_patch_size is None:
            target_patch_size = checkpoint_target_patch_size
        elif target_patch_size != checkpoint_target_patch_size:
            raise ValueError(
                "Requested target_patch_size does not match checkpoint metadata: "
                f"{target_patch_size} != {checkpoint_target_patch_size}."
            )
        self.checkpoint_provenance = checkpoint.provenance
        if text_vocab_size is None:
            manifest_vocab_size = (
                int(preloaded_manifest.architecture["text_vocab_size"])
                if preloaded_manifest is not None
                else TOTAL_VOCAB_SIZE
            )
            text_vocab_size = checkpoint.infer_text_vocab_size(manifest_vocab_size)
        if checkpoint_text_conditioning.mode == "frozen_features":
            if text_num_layers is None:
                text_num_layers = 0
            elif text_num_layers != 0:
                raise ValueError(
                    "Frozen-feature checkpoints require text_num_layers=0 at inference."
                )
            conditioning_feature_size = checkpoint_text_conditioning.feature_size
        else:
            if text_num_layers is None:
                text_num_layers = 6
            conditioning_feature_size = None
        checkpoint_supports_voice_cloning = checkpoint.infer_speaker_conditioning(False)
        inferred_summary_tokens = checkpoint.infer_speaker_num_summary_tokens()
        if not isinstance(inferred_summary_tokens, int) or isinstance(
            inferred_summary_tokens, bool
        ):
            # Compatibility for protocol mocks/custom loaders. The authenticated
            # manifest comparison below remains authoritative for real runtimes.
            inferred_summary_tokens = (
                int(preloaded_manifest.architecture.get("speaker_num_summary_tokens", 0))
                if preloaded_manifest is not None
                else 0
            )
        if speaker_num_summary_tokens is None:
            speaker_num_summary_tokens = inferred_summary_tokens
        elif speaker_num_summary_tokens != inferred_summary_tokens:
            raise VoiceCloningUnsupportedError(
                "Requested speaker_num_summary_tokens does not match checkpoint metadata: "
                f"{speaker_num_summary_tokens} != {inferred_summary_tokens}."
            )
        self.speaker_num_summary_tokens = speaker_num_summary_tokens
        requested_speaker_patch_size = speaker_patch_size
        if checkpoint_supports_voice_cloning:
            inferred_patch_size = checkpoint.infer_speaker_patch_size(4)
            if (
                requested_speaker_patch_size is not None
                and requested_speaker_patch_size != inferred_patch_size
            ):
                raise VoiceCloningUnsupportedError(
                    f"Checkpoint speaker patch size is {inferred_patch_size}, but "
                    f"{requested_speaker_patch_size} was requested."
                )
            speaker_patch_size = inferred_patch_size
        else:
            speaker_patch_size = requested_speaker_patch_size or 4
        self.speaker_patch_size = speaker_patch_size
        if use_speaker_conditioning is None:
            use_speaker_conditioning = checkpoint_supports_voice_cloning
        elif use_speaker_conditioning and not checkpoint_supports_voice_cloning:
            raise VoiceCloningUnsupportedError(
                "The selected checkpoint was not trained with speaker conditioning. "
                "Choose a versioned speaker-conditioned checkpoint."
            )
        self.text_vocab_size = text_vocab_size
        self.supports_voice_cloning = bool(use_speaker_conditioning)

        checkpoint_language = checkpoint.language_capability()
        if use_language_conditioning is None:
            use_language_conditioning = checkpoint_language.enabled
        elif use_language_conditioning and not checkpoint_language.enabled:
            raise MultilingualUnsupportedError(
                "The selected checkpoint has no versioned language-conditioning state. "
                "Train or select a multilingual checkpoint."
            )
        elif checkpoint_language.enabled and not use_language_conditioning:
            raise MultilingualUnsupportedError(
                "Language conditioning cannot be disabled for a checkpoint trained with it."
            )
        checkpoint_languages = checkpoint_language.supported_languages
        if checkpoint_language.enabled:
            if supported_languages is not None:
                requested_languages = normalize_languages(supported_languages)
                if requested_languages != checkpoint_languages:
                    raise MultilingualUnsupportedError(
                        "Requested supported_languages do not match the checkpoint metadata: "
                        f"{requested_languages!r} != {checkpoint_languages!r}."
                    )
            resolved_languages = checkpoint_languages
        else:
            if supported_languages is not None:
                requested_languages = normalize_languages(supported_languages)
                if requested_languages != (DEFAULT_LANGUAGE,):
                    raise MultilingualUnsupportedError(
                        "A legacy checkpoint supports only English; "
                        f"received supported_languages={requested_languages!r}."
                    )
            resolved_languages = (DEFAULT_LANGUAGE,)
        self.uses_language_conditioning = bool(use_language_conditioning)
        self.supported_languages = resolved_languages
        self.supports_multilingual = len(resolved_languages) > 1
        checkpoint_duration = checkpoint.duration_capability()
        if use_duration_predictor is None:
            use_duration_predictor = checkpoint_duration.enabled
        elif use_duration_predictor and not checkpoint_duration.enabled:
            raise LearnedDurationUnsupportedError(
                "The selected checkpoint has no versioned, trained duration predictor."
            )
        elif checkpoint_duration.enabled and not use_duration_predictor:
            raise LearnedDurationUnsupportedError(
                "Learned duration cannot be disabled for an EchoDiT v2 duration checkpoint."
            )
        self.uses_learned_duration = bool(use_duration_predictor)
        checkpoint_alignment = checkpoint.monotonic_alignment_capability()
        self.uses_mas_duration = checkpoint_alignment.enabled
        reference_language_capability = checkpoint.reference_language_capability()
        self.supported_reference_languages = reference_language_capability.supported_languages
        checkpoint_language_pairs = reference_language_capability.supported_pairs
        self.supported_language_pairs = (
            checkpoint_language_pairs
            if checkpoint_language_pairs
            else (
                (LanguagePair(target=DEFAULT_LANGUAGE, reference=DEFAULT_LANGUAGE),)
                if self.supports_voice_cloning and not self.uses_language_conditioning
                else ()
            )
        )
        self.supports_cross_lingual = bool(
            self.supports_voice_cloning
            and self.uses_language_conditioning
            and reference_language_capability.enabled
            and any(pair.is_cross_lingual for pair in checkpoint_language_pairs)
        )
        self.checkpoint_path = checkpoint.path

        if dacvae_model is None:
            raise ValueError(
                "dacvae_model is required; pass the local codec artifact or the Hugging Face "
                "ID recorded by the checkpoint manifest."
            )
        provenance = checkpoint.provenance
        manifest_path = (
            provenance.manifest_path
            if provenance is not None and provenance.manifest_path is not None
            else checkpoint.path.parent / MODEL_MANIFEST_FILENAME
        )
        selected_filename = (
            provenance.selected_filename if provenance is not None else checkpoint.path.name
        )
        # A mocked/custom FlowCheckpoint loader may not invoke the callback. Retain a
        # strict fallback while the built-in loader authenticates before torch.load.
        self.model_manifest = preloaded_manifest or load_model_manifest(manifest_path)
        representation = self.model_manifest.representation
        if isinstance(dacvae_model, str) and representation["codec_revision"] is not None:
            codec_source = normalize_dacvae_source(
                dacvae_model,
                dacvae_revision=str(representation["codec_revision"]),
                dacvae_filename=str(representation["codec_filename"]),
            )
        else:
            codec_source = normalize_dacvae_source(dacvae_model)
        self.dacvae_source = codec_source
        inference_architecture = {
            "architecture_version": ECHODIT_ARCHITECTURE_VERSION,
            "latent_size": latent_size,
            "model_size": model_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "intermediate_size": intermediate_size,
            "text_model_size": text_model_size,
            "text_num_layers": text_num_layers,
            "text_num_heads": text_num_heads,
            "text_intermediate_size": text_intermediate_size,
            "speaker_model_size": speaker_model_size,
            "speaker_num_layers": speaker_num_layers,
            "speaker_num_heads": speaker_num_heads,
            "speaker_intermediate_size": speaker_intermediate_size,
            "timestep_embed_size": timestep_embed_size,
            "adaln_rank": adaln_rank,
            "text_vocab_size": text_vocab_size,
            "speaker_patch_size": speaker_patch_size,
            "speaker_num_summary_tokens": speaker_num_summary_tokens,
            "target_patch_size": target_patch_size,
            "use_speaker_conditioning": bool(use_speaker_conditioning),
            "use_mas_duration": checkpoint_alignment.enabled,
            "norm_eps": float(norm_eps),
        }
        checkpoint_capabilities = {
            "speaker_conditioning": bool(use_speaker_conditioning),
            "language_conditioning": self.uses_language_conditioning,
            "supported_languages": list(self.supported_languages),
            "supported_reference_languages": list(self.supported_reference_languages),
            "supported_language_pairs": [
                list(pair.as_tuple()) for pair in checkpoint_language_pairs
            ],
            "duration_predictor": self.uses_learned_duration,
            "duration_predictor_hidden_size": checkpoint_duration.hidden_size,
            "duration_predictor_num_layers": checkpoint_duration.num_layers,
            "duration_predictor_use_speaker": checkpoint_duration.uses_speaker,
            "monotonic_alignment": checkpoint_alignment.enabled,
            "duration_alignment_hidden_size": checkpoint_alignment.hidden_size,
        }
        checkpoint_generation = {
            "objective": checkpoint_objective,
            "diffusion_schedule_shift": checkpoint_schedule_shift,
        }
        checkpoint_text_manifest = dict(self.model_manifest.text_conditioning)
        checkpoint_text_manifest.update(
            {
                "mode": checkpoint_text_conditioning.mode,
                "feature_size": checkpoint_text_conditioning.feature_size,
                "adapter_version": checkpoint_text_conditioning.adapter_version,
            }
        )
        validate_inference_manifest(
            self.model_manifest,
            checkpoint_path=checkpoint.path,
            selected_filename=selected_filename,
            base_checkpoint_path=(
                provenance.base_path if provenance is not None else checkpoint.path
            ),
            base_filename=(
                provenance.base_filename if provenance is not None else checkpoint.path.name
            ),
            architecture=inference_architecture,
            capabilities=checkpoint_capabilities,
            generation=checkpoint_generation,
            text_conditioning=checkpoint_text_manifest,
            codec_source=codec_source,
            codec_backend=dacvae_backend,
        )

        if self.text_conditioning_mode == "frozen_features":
            expected_provider_spec = FrozenTextProviderSpec.from_manifest(self.model_manifest)
            if frozen_text_provider is None:
                frozen_text_provider = FrozenTextProvider.from_manifest(
                    self.model_manifest,
                    device=self.device,
                )
            actual_provider_spec = getattr(frozen_text_provider, "spec", None)
            if actual_provider_spec != expected_provider_spec:
                raise ValueError(
                    "Injected frozen_text_provider does not match the authenticated model "
                    "manifest text-conditioning contract."
                )
            validate_provider = getattr(
                frozen_text_provider,
                "validate_acoustic_contract",
                None,
            )
            if not callable(validate_provider) or not callable(
                getattr(frozen_text_provider, "encode", None)
            ):
                raise TypeError(
                    "frozen_text_provider must expose encode() and validate_acoustic_contract()."
                )
            validate_provider(self.model_manifest)
            self.frozen_text_provider = frozen_text_provider
            self.text_pad_token = int(frozen_text_provider.pad_token_id)
        else:
            if frozen_text_provider is not None:
                raise ValueError(
                    "frozen_text_provider cannot be supplied to a scratch-token checkpoint."
                )
            self.frozen_text_provider = None
            self.text_pad_token = PAD_TOKEN
        print(
            "Loading text frontend: "
            + (
                str(self.model_manifest.text_conditioning["encoder_id"])
                if self.text_conditioning_mode == "frozen_features"
                else TOKENIZER_NAME
            )
        )

        # Create EchoDiT model with built-in text encoder
        print("\nCreating EchoDiT model...")
        print(
            "EchoDiT text path: "
            + (
                f"frozen {conditioning_feature_size}-D features with a trainable adapter"
                if self.text_conditioning_mode == "frozen_features"
                else f"built-in {text_num_layers}-layer text encoder"
            )
        )
        print(
            "Speaker encoder: "
            f"patch_size={speaker_patch_size}, summary_tokens={speaker_num_summary_tokens}"
        )
        self.flow_model = create_flow_matching_echodit(
            latent_size=latent_size,
            model_size=model_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,  # Use explicit parameter
            text_vocab_size=text_vocab_size,
            text_conditioning_mode=self.text_conditioning_mode,
            conditioning_feature_size=conditioning_feature_size,
            text_model_size=text_model_size,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_intermediate_size=text_intermediate_size,
            # Speaker encoder config
            speaker_patch_size=speaker_patch_size,
            speaker_num_summary_tokens=speaker_num_summary_tokens,
            target_patch_size=target_patch_size,
            speaker_model_size=speaker_model_size,
            speaker_num_layers=speaker_num_layers,
            speaker_num_heads=speaker_num_heads,
            speaker_intermediate_size=speaker_intermediate_size,
            timestep_embed_size=timestep_embed_size,
            adaln_rank=adaln_rank,
            norm_eps=norm_eps,
            cfg_dropout=0.1,
            use_speaker_conditioning=use_speaker_conditioning,
            use_language_conditioning=self.uses_language_conditioning,
            supported_languages=self.supported_languages,
            supported_reference_languages=self.supported_reference_languages or None,
            supported_language_pairs=checkpoint_language_pairs or None,
            use_duration_predictor=self.uses_learned_duration,
            duration_predictor_hidden_size=checkpoint_duration.hidden_size or 256,
            duration_predictor_num_layers=checkpoint_duration.num_layers or 2,
            duration_predictor_use_speaker=checkpoint_duration.uses_speaker,
            use_mas_duration=checkpoint_alignment.enabled,
            duration_alignment_hidden_size=checkpoint_alignment.hidden_size or 64,
            generative_objective=checkpoint_objective,
            diffusion_schedule_shift=checkpoint_schedule_shift,
        )

        # Load checkpoint
        print(f"\nLoading EchoDiT checkpoint: {checkpoint.path}")
        checkpoint.load_into(self.flow_model)
        del checkpoint

        # Cast authenticated CPU weights before the device transfer. This avoids
        # staging a full FP32 acoustic model on the accelerator and never routes
        # complex/FP64/non-persistent buffers through a dtype conversion.
        _cast_acoustic_parameters(self.flow_model, self.acoustic_dtype)
        self.flow_model.to(self.device)
        self.flow_model.eval()

        self.dacvae = load_dacvae(
            codec_source,
            backend=dacvae_backend,
            device=self.device,
            freeze=True,
            expected_latent_size=latent_size,
            expected_sha256=self.model_manifest.representation["codec_sha256"],
        )
        validate_loaded_codec(self.model_manifest, self.dacvae)

        # Freeze all parameters
        for param in self.flow_model.parameters():
            param.requires_grad = False
        # DACVAE specs
        self.sample_rate = self.dacvae.sample_rate
        self.hop_length = self.dacvae.hop_length
        self.frame_rate = self.sample_rate / self.hop_length
        self._decode = self.dacvae.decode

        print(f"\n✓ Inference pipeline ready on {self.device}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Frame rate: {self.frame_rate:.1f} frames/sec")
        print(f"  Acoustic dtype: {self.acoustic_dtype}")
        print(f"  Objective: {self.generative_objective}")
        print(f"  Text conditioning: {self.text_conditioning_mode}")
        print(f"  Voice cloning: {'available' if self.supports_voice_cloning else 'unavailable'}")
        print(f"  Languages: {', '.join(self.supported_languages)}")
        print(
            "  Cross-lingual cloning: "
            f"{'available' if self.supports_cross_lingual else 'unavailable'}"
        )

    @classmethod
    def from_preset(cls, model_preset: str, **kwargs):
        """Construct an inference runtime with one packaged architecture family."""
        preset = get_model_preset(model_preset)
        architecture = preset.model_kwargs()
        conflicts = {
            name: (kwargs[name], expected)
            for name, expected in architecture.items()
            if name in kwargs
            and kwargs[name] != expected
            and not (name == "text_num_layers" and kwargs[name] == 0)
        }
        if kwargs.get("text_num_layers") == 0:
            architecture["text_num_layers"] = 0
        if conflicts:
            details = ", ".join(
                f"{name}={actual!r} (preset {expected!r})"
                for name, (actual, expected) in conflicts.items()
            )
            raise ValueError(
                f"Inference arguments conflict with model preset {model_preset!r}: {details}."
            )
        return cls(**{**architecture, **kwargs})

    def generation_profile(self, name: str = "quality") -> GenerationConfig:
        """Return one of the packaged, validated inference profiles."""
        return self.settings.profile(name)

    def _prepare_conditioning(
        self,
        text: str,
        language: str | None = None,
        *,
        normalized_text: str | None = None,
        phonemes: str | Sequence[str] | None = None,
        language_spans: Sequence[TextSpan | Mapping[str, object]] | None = None,
    ) -> _PreparedText:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip() and phonemes is None and not language_spans:
            raise ValueError("text, phonemes, or language_spans must contain speech content.")
        if getattr(self, "text_conditioning_mode", "scratch_tokens") == "frozen_features":
            provider = getattr(self, "frozen_text_provider", None)
            if provider is None:
                raise RuntimeError(
                    "Frozen-feature inference has no authenticated frozen text provider."
                )
            prepared = provider.encode(
                text.strip(),
                language=language,
                normalized_text=normalized_text,
                phonemes=phonemes,
                language_spans=language_spans,
            )
            if prepared.contract_sha256 != provider.spec.contract_sha256:
                raise RuntimeError(
                    "Frozen text provider returned conditioning from a different contract."
                )
            self._validate_token_language_capability(prepared.token_language_ids)
            return _PreparedText(
                conditioning_ids=prepared.conditioning_ids.to(self.device)[None, :],
                conditioning_mask=prepared.conditioning_mask.to(self.device)[None, :],
                conditioning_features=prepared.conditioning_features.to(
                    device=self.device,
                    dtype=self._acoustic_state_dtype(),
                )[None, :, :],
                token_language_ids=prepared.token_language_ids.to(self.device)[None, :],
                alignment_mask=prepared.alignment_mask.to(self.device)[None, :],
            )
        prepared = encode_tts_conditioning(
            text.strip(),
            vocab_size=getattr(self, "text_vocab_size", None),
            language=language,
            normalized_text=normalized_text,
            phonemes=phonemes,
            language_spans=language_spans,
        )
        self._validate_token_language_capability(prepared.token_language_ids)
        return _PreparedText(
            conditioning_ids=torch.tensor(
                [prepared.conditioning_ids], dtype=torch.long, device=self.device
            ),
            conditioning_mask=None,
            conditioning_features=None,
            token_language_ids=torch.tensor(
                [prepared.token_language_ids], dtype=torch.long, device=self.device
            ),
            alignment_mask=torch.tensor(
                [prepared.alignment_mask], dtype=torch.bool, device=self.device
            ),
        )

    def _validate_token_language_capability(self, values) -> None:
        """Reject code-switch spans outside the authenticated checkpoint capability."""
        token_languages = torch.as_tensor(values)
        if token_languages.ndim != 1 or token_languages.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("token_language_ids must be a rank-one integer axis.")
        if bool((token_languages < 0).any()) or bool((token_languages > LANGUAGE_COUNT).any()):
            raise ValueError(f"token_language_ids must be within [0, {LANGUAGE_COUNT}].")
        supported = getattr(self, "supported_languages", (DEFAULT_LANGUAGE,))
        allowed = {language_id(code) for code in supported}
        observed = {int(value) for value in token_languages.tolist() if int(value) != 0}
        if not observed.issubset(allowed):
            raise MultilingualUnsupportedError(
                "Conditioning contains code-switch token languages outside the checkpoint's "
                f"supported_languages capability: {sorted(observed - allowed)}."
            )

    def _resolve_language_pair(
        self,
        language: str | None,
        reference_language: str | None,
        *,
        has_reference: bool,
    ) -> LanguagePair:
        """Validate independent target-text and source-reference languages."""
        pair = LanguagePair.resolve(
            language,
            reference_language,
            has_reference=has_reference,
        )
        supported = getattr(self, "supported_languages", (DEFAULT_LANGUAGE,))
        if pair.target not in supported:
            raise MultilingualUnsupportedError(
                f"Checkpoint {self.checkpoint_path.name!r} does not support target language "
                f"{pair.target!r}. Supported languages: {', '.join(supported)}."
            )
        if has_reference and not getattr(self, "supports_voice_cloning", False):
            raise VoiceCloningUnsupportedError(
                "Reference-audio synthesis requires a speaker-conditioned checkpoint."
            )
        if has_reference:
            supported_pairs = {
                supported_pair.as_tuple()
                for supported_pair in getattr(self, "supported_language_pairs", ())
            }
            if pair.as_tuple() not in supported_pairs:
                raise CrossLingualUnsupportedError(
                    f"Checkpoint {self.checkpoint_path.name!r} does not declare trained "
                    f"target/reference language pair {pair.as_tuple()!r}. Supported pairs: "
                    f"{sorted(supported_pairs)}."
                )
        return pair

    def _language_ids(self, pair: LanguagePair) -> torch.Tensor | None:
        """Create target-language conditioning without coupling it to the speaker."""
        if not getattr(self, "uses_language_conditioning", False):
            return None
        return torch.tensor(
            [language_id(pair.target)],
            dtype=torch.long,
            device=self.device,
        )

    def estimate_duration(self, text: str, duration: float | None = None) -> float:
        """Estimate output length with the packaged, bounded duration policy."""
        return self.settings.duration.estimate(text, duration)

    def _resolve_duration_shape(
        self,
        text: str,
        duration: float | None,
        conditioning_ids: torch.Tensor,
        language_ids: torch.Tensor | None,
        speaker_latent: torch.Tensor | None,
        conditioning_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        predicted_frames: torch.Tensor | None = None,
    ) -> tuple[float, int]:
        """Resolve seconds and DACVAE frames without granting capability to legacy weights."""
        if duration is not None or not getattr(self, "uses_learned_duration", False):
            seconds = self.estimate_duration(text, duration)
            return seconds, int(seconds * self.frame_rate)

        predicted = predicted_frames
        if predicted is None:
            feature_kwargs = (
                {"conditioning_features": conditioning_features}
                if conditioning_features is not None
                else {}
            )
            predicted = self.flow_model.predict_duration_frames(
                conditioning_ids,
                attention_mask=conditioning_mask,
                speaker_latent=speaker_latent,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
                **feature_kwargs,
            )
        if predicted.numel() != 1:
            raise RuntimeError("Single-utterance duration prediction must return one frame count.")
        predicted_frames = float(predicted.item())
        if not math.isfinite(predicted_frames):
            raise RuntimeError("Learned duration prediction returned a non-finite frame count.")
        minimum_frames = max(1, math.ceil(self.settings.duration.minimum_seconds * self.frame_rate))
        maximum_frames = max(
            minimum_frames,
            math.floor(self.settings.duration.maximum_seconds * self.frame_rate),
        )
        if predicted_frames > maximum_frames:
            predicted_seconds = predicted_frames / self.frame_rate
            raise ValueError(
                f"Learned duration prediction {predicted_seconds:g}s exceeds the configured "
                f"maximum of {self.settings.duration.maximum_seconds:g}s. Split the text or "
                "select a duration policy validated for longer utterances."
            )
        num_frames = max(minimum_frames, int(predicted_frames))
        return num_frames / self.frame_rate, num_frames

    def _resolve_token_durations(
        self,
        conditioning_ids: torch.Tensor,
        *,
        num_frames: int,
        conditioning_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        expected_token_durations: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Predict an exact token-to-frame allocation for a MAS-trained checkpoint."""
        if not getattr(self, "uses_mas_duration", False):
            return None
        if expected_token_durations is None:
            feature_kwargs = (
                {"conditioning_features": conditioning_features}
                if conditioning_features is not None
                else {}
            )
            token_durations = self.flow_model.predict_token_duration_frames(
                conditioning_ids,
                attention_mask=conditioning_mask,
                speaker_latent=speaker_latent,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
                total_frames=num_frames,
                **feature_kwargs,
            )
        else:
            token_durations = self.flow_model.allocate_token_duration_frames(
                expected_token_durations,
                conditioning_mask,
                total_frames=num_frames,
                alignment_mask=alignment_mask,
            )
        if tuple(token_durations.shape) != tuple(conditioning_ids.shape):
            raise RuntimeError(
                "MAS token-duration prediction did not preserve the conditioning shape."
            )
        return token_durations

    def _encode_trajectory_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        *,
        conditioning_mask: torch.Tensor | None,
        language_ids: torch.Tensor | None,
        token_language_ids: torch.Tensor | None,
        alignment_mask: torch.Tensor | None,
        conditioning_features: torch.Tensor | None,
        speaker_latent: torch.Tensor | None,
        cfg_scale: float,
        cfg_mode: str,
        cfg_scale_text: float | None,
        cfg_scale_speaker: float | None,
        needs_learned_duration: bool,
    ) -> tuple[object | None, torch.Tensor | None, torch.Tensor | None]:
        """Encode request invariants once and optionally run the shared duration head."""
        encode = getattr(self.flow_model, "encode_inference_conditioning", None)
        finalize = getattr(self.flow_model, "finalize_inference_conditioning", None)
        predict = getattr(
            self.flow_model,
            "predict_duration_frames_and_token_weights",
            None,
        )
        needs_prediction = needs_learned_duration or getattr(self, "uses_mas_duration", False)
        if (
            not callable(encode)
            or not callable(finalize)
            or (needs_prediction and not callable(predict))
        ):
            return None, None, None
        guidance_active = cfg_guidance_active(
            cfg_scale=cfg_scale,
            cfg_mode=cfg_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
        )
        encode_kwargs = {}
        if conditioning_features is not None:
            encode_kwargs["conditioning_features"] = conditioning_features
        encoded = encode(
            conditioning_ids,
            conditioning_mask,
            speaker_latent,
            cfg_mode=cfg_mode if guidance_active else None,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            **encode_kwargs,
        )
        if not needs_prediction:
            return encoded, None, None
        predicted_frames, expected_token_durations = predict(encoded)
        return encoded, predicted_frames, expected_token_durations

    def _finalize_trajectory_conditioning(
        self,
        encoded: object | None,
        token_durations: torch.Tensor | None,
    ) -> tuple[object | None, object | None]:
        """Finish KV/MAS preparation without another invariant encoder pass."""
        if encoded is None:
            return None, None
        return self.flow_model.finalize_inference_conditioning(
            encoded,
            token_durations=token_durations,
        )

    def _acoustic_state_dtype(self) -> torch.dtype:
        flow_model = getattr(self, "flow_model", None)
        parameters = getattr(flow_model, "parameters", None)
        if not callable(parameters):
            # Preserve the historical FP32 protocol for lightweight runtimes
            # that inject inference helpers without constructing EchoDiT.
            return torch.float32
        parameter = next(parameters(), None)
        if parameter is None:
            return torch.float32
        if not torch.is_floating_point(parameter) or parameter.dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise RuntimeError(
                "The acoustic model must expose a float16, bfloat16, float32, or float64 "
                "state parameter."
            )
        return parameter.dtype

    def _decode_generated_latents(self, generated_latents: torch.Tensor) -> torch.Tensor:
        """Move acoustic solver state to the immutable codec's parameter dtype."""
        codec = getattr(self, "dacvae", None)
        if codec is None:
            # Lightweight protocol/test runtimes may inject only a decode callable.
            return self._decode(generated_latents)
        codec_parameters = getattr(codec, "parameters", None)
        if not callable(codec_parameters):
            return self._decode(generated_latents)
        codec_parameter = next(codec_parameters(), None)
        if codec_parameter is None or not torch.is_floating_point(codec_parameter):
            raise RuntimeError("DACVAE must expose a floating parameter for latent decoding.")
        return self._decode(
            generated_latents.to(
                device=codec_parameter.device,
                dtype=codec_parameter.dtype,
            )
        )

    def _align_speaker_latent(self, speaker_latent: torch.Tensor) -> torch.Tensor:
        if speaker_latent.ndim == 2:
            speaker_latent = speaker_latent.unsqueeze(0)
        if speaker_latent.ndim != 3:
            raise ValueError(
                "speaker_latent must have shape [channels, frames] or [1, channels, frames]."
            )
        if speaker_latent.shape[0] != 1:
            raise ValueError("Single-utterance inference accepts one speaker reference.")
        if speaker_latent.shape[1] != self.latent_size:
            raise ValueError(
                f"speaker_latent has {speaker_latent.shape[1]} channels; "
                f"expected {self.latent_size}."
            )
        if speaker_latent.shape[-1] == 0:
            raise ValueError("speaker_latent must contain at least one frame.")
        if not torch.isfinite(speaker_latent).all():
            raise ValueError("speaker_latent contains non-finite values.")

        remainder = speaker_latent.shape[-1] % self.speaker_patch_size
        if remainder:
            speaker_latent = F.pad(
                speaker_latent,
                (0, self.speaker_patch_size - remainder),
            )
        return speaker_latent.to(self.device, dtype=self._acoustic_state_dtype())

    @torch.inference_mode()
    def encode_reference_audio(
        self,
        reference_audio: AudioReference,
        *,
        sample_rate: int | None = None,
        max_seconds: float | None = None,
    ) -> torch.Tensor:
        """Encode one reference recording for a speaker-conditioned checkpoint."""
        if not self.supports_voice_cloning:
            raise VoiceCloningUnsupportedError(
                "Voice cloning is unavailable for this checkpoint. "
                f"{self.checkpoint_path.name!r} has no trained speaker-conditioning state. "
                "Use a versioned speaker-conditioned checkpoint."
            )

        if isinstance(reference_audio, (str, Path)):
            if sample_rate is not None:
                raise ValueError("sample_rate must be omitted when reference_audio is a file.")
            waveform, source_sample_rate = torchaudio.load(str(reference_audio))
        elif isinstance(reference_audio, torch.Tensor):
            if sample_rate is None:
                raise ValueError("sample_rate is required for a reference audio tensor.")
            waveform = reference_audio.detach().cpu()
            source_sample_rate = sample_rate
        else:
            raise TypeError("reference_audio must be a path or torch.Tensor.")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim != 2:
            raise ValueError("reference audio must have shape [samples] or [channels, samples].")
        if waveform.shape[-1] == 0:
            raise ValueError("reference audio is empty.")
        if source_sample_rate <= 0:
            raise ValueError("reference sample_rate must be positive.")
        waveform = waveform.float().mean(dim=0, keepdim=True)
        if not torch.isfinite(waveform).all():
            raise ValueError("reference audio contains non-finite samples.")

        if source_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                source_sample_rate,
                self.sample_rate,
            )

        reference_limit = self.max_reference_seconds if max_seconds is None else max_seconds
        if not math.isfinite(reference_limit) or reference_limit <= 0:
            raise ValueError("max reference duration must be finite and positive.")
        max_reference_samples = int(reference_limit * self.sample_rate)
        if max_reference_samples < self.hop_length:
            minimum_seconds = self.hop_length / self.sample_rate
            raise ValueError(
                "max reference duration is too short to encode one DACVAE frame; "
                f"use at least {minimum_seconds:.3f} seconds."
            )
        waveform = waveform[..., :max_reference_samples]
        if waveform.shape[-1] < self.hop_length:
            waveform = F.pad(waveform, (0, self.hop_length - waveform.shape[-1]))

        seed = derive_dacvae_posterior_seed(
            waveform,
            codec_sha256=self.model_manifest.representation["codec_sha256"],
        )
        speaker_latent = encode_dacvae_posterior_seeded(
            self.dacvae,
            waveform.unsqueeze(0).to(self.device),
            seed=seed,
        )
        return self._align_speaker_latent(speaker_latent)

    def _resolve_speaker_latent(
        self,
        *,
        reference_audio: AudioReference | None,
        reference_sample_rate: int | None,
        speaker_latent: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if reference_audio is not None and speaker_latent is not None:
            raise ValueError("Pass either reference_audio or speaker_latent, not both.")
        if reference_audio is not None:
            return self.encode_reference_audio(
                reference_audio,
                sample_rate=reference_sample_rate,
            )
        if speaker_latent is not None:
            if not self.supports_voice_cloning:
                raise VoiceCloningUnsupportedError(
                    "The selected checkpoint does not support speaker conditioning."
                )
            return self._align_speaker_latent(speaker_latent)
        return None

    def _effective_cfg(
        self,
        *,
        cfg_scale: float,
        cfg_mode: str,
        cfg_scale_text: float | None,
        cfg_scale_speaker: float | None,
        speaker_latent: torch.Tensor | None,
    ) -> tuple[float, str, float | None, float | None]:
        # With no speaker-conditioned model, the independent speaker branch is
        # identical to the conditional branch. Joint text CFG is algebraically
        # equivalent and avoids one redundant model evaluation.
        if (
            not getattr(self, "supports_voice_cloning", False)
            and speaker_latent is None
            and cfg_mode == "independent"
            and cfg_scale_text is not None
        ):
            return 1.0 + cfg_scale_text, "joint", None, None
        return cfg_scale, cfg_mode, cfg_scale_text, cfg_scale_speaker

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        # ODE solver settings; select them from checkpoint-specific evaluation.
        num_steps: int = 64,
        solver: str | None = None,
        # Neutral conditional inference; guidance requires checkpoint-specific calibration.
        cfg_scale: float = 1.0,
        cfg_mode: str = "joint",
        cfg_scale_text: float | None = None,
        cfg_scale_speaker: float | None = None,
        cfg_min_t: float = 0.0,
        cfg_max_t: float = 1.0,
        # Noise and rescaling
        initial_noise_scale: float = 1.0,
        temporal_rescale_k: float = 1.0,
        temporal_rescale_sigma: float = 2.5,
        # Optional only when the exact training-latent distribution justifies it.
        target_latent_std: float | None = None,
        cache_mode: str = "none",
        # Other
        duration: float | None = None,
        show_progress: bool = True,
        reference_audio: AudioReference | None = None,
        reference_sample_rate: int | None = None,
        speaker_latent: torch.Tensor | None = None,
        language: str | None = None,
        reference_language: str | None = None,
        normalized_text: str | None = None,
        phonemes: str | Sequence[str] | None = None,
        language_spans: Sequence[TextSpan | Mapping[str, object]] | None = None,
    ) -> torch.Tensor:
        """
        Synthesize speech from text.

        Args:
            text: Input text string
            num_steps: Number of ODE solver steps
            solver: ODE solver type ("euler", "midpoint", "heun", "rk4")
            cfg_scale: Joint guidance scale; 1.0 uses conditional prediction without CFG
            cfg_mode: CFG mode - "independent", "joint", or "alternating"
            cfg_scale_text: Optional text guidance scale for an explicitly selected CFG mode
            cfg_scale_speaker: Optional speaker guidance scale for an explicitly selected CFG mode
            cfg_min_t: Minimum timestep for CFG
            cfg_max_t: Maximum timestep for CFG
            initial_noise_scale: Initial noise scale
            temporal_rescale_k: Temporal rescaling (1.0 disables it)
            temporal_rescale_sigma: Temporal rescaling sigma
            target_latent_std: Optional checkpoint-specific final-latent standard deviation
            cache_mode: ``"cache_dit"`` enables DBCache for the EchoDiT blocks
            duration: Target duration in seconds (None = auto-estimate)
            show_progress: Whether to show progress bar
            reference_audio: Reference WAV path or waveform for voice cloning
            reference_sample_rate: Required when reference_audio is a tensor
            speaker_latent: Pre-encoded speaker reference, mutually exclusive
                with reference_audio
            language: Target text/speech language (defaults to English)
            reference_language: Source language spoken in the reference audio;
                defaults to ``language`` for backward-compatible same-language cloning
            normalized_text: Optional externally reviewed normalization of ``text``
            phonemes: Optional reviewed IPA/phone sequence used instead of orthography
            language_spans: Optional explicitly language-tagged spans for code switching

        Returns:
            Audio waveform tensor [samples]
        """
        # Never expose cache statistics from a previous request if validation,
        # cache setup, ODE sampling, or decoding fails for this one.
        self.last_cache_stats = CacheDiTStats()
        # Duration/MAS prediction uses the acoustic model before ODE sampling,
        # so reject a runtime poisoned by failed Cache-DiT cleanup immediately.
        assert_cache_dit_healthy(self.flow_model)
        generative_objective, diffusion_schedule_shift = _runtime_generation_contract(self)
        if solver is None:
            solver = "ddim" if generative_objective == VP_DIFFUSION_OBJECTIVE else "heun"
        language_pair = self._resolve_language_pair(
            language,
            reference_language,
            has_reference=reference_audio is not None or speaker_latent is not None,
        )
        prepared_text = self._prepare_conditioning(
            text,
            language_pair.target,
            normalized_text=normalized_text,
            phonemes=phonemes,
            language_spans=language_spans,
        )
        conditioning_ids = prepared_text.conditioning_ids
        conditioning_mask = prepared_text.conditioning_mask
        conditioning_features = prepared_text.conditioning_features
        token_language_ids = prepared_text.token_language_ids
        alignment_mask = prepared_text.alignment_mask
        language_ids = self._language_ids(language_pair)
        speaker_latent = self._resolve_speaker_latent(
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            speaker_latent=speaker_latent,
        )

        cfg_scale, cfg_mode, cfg_scale_text, cfg_scale_speaker = self._effective_cfg(
            cfg_scale=cfg_scale,
            cfg_mode=cfg_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            speaker_latent=speaker_latent,
        )
        encoded_conditioning, predicted_frames, expected_token_durations = (
            self._encode_trajectory_conditioning(
                conditioning_ids,
                conditioning_mask=conditioning_mask,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
                conditioning_features=conditioning_features,
                speaker_latent=speaker_latent,
                cfg_scale=cfg_scale,
                cfg_mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
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
            speaker_latent,
            conditioning_mask=conditioning_mask,
            conditioning_features=conditioning_features,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            predicted_frames=predicted_frames,
        )
        token_durations = self._resolve_token_durations(
            conditioning_ids,
            num_frames=num_frames,
            conditioning_mask=conditioning_mask,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            conditioning_features=conditioning_features,
            speaker_latent=speaker_latent,
            expected_token_durations=expected_token_durations,
        )
        latent_shape = (1, self.latent_size, num_frames)

        print(f"\nGenerating {estimated_duration:.2f}s of audio ({num_frames} frames)...")

        generation_started_at = time.perf_counter()

        if cache_mode not in ("none", "cache_dit"):
            raise ValueError("cache_mode must be 'none' or 'cache_dit'.")
        if cache_mode == "cache_dit":
            validate_cache_dit_options(
                num_steps=num_steps,
                solver=solver,
                cfg_scale=cfg_scale,
                cfg_mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_min_t=cfg_min_t,
                cfg_max_t=cfg_max_t,
            )

        cache_context = (
            CacheDiTSession(self.flow_model, num_steps=num_steps)
            if cache_mode == "cache_dit"
            else nullcontext(None)
        )

        # Generate latents with ODE solver
        # EchoDiT's built-in text encoder processes conditioning_ids
        with cache_context as cache_session:
            prepared_conditioning, prepared_cfg_conditioning = (
                self._finalize_trajectory_conditioning(
                    encoded_conditioning,
                    token_durations,
                )
            )
            generated_latents = ODESolver.sample(
                model=self.flow_model,
                conditioning_ids=conditioning_ids,
                num_steps=num_steps,
                latent_shape=latent_shape,
                solver=solver,
                # CFG parameters
                cfg_scale=cfg_scale,
                cfg_mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                cfg_min_t=cfg_min_t,
                cfg_max_t=cfg_max_t,
                # Noise and rescaling
                initial_noise_scale=initial_noise_scale,
                temporal_rescale_k=temporal_rescale_k,
                temporal_rescale_sigma=temporal_rescale_sigma,
                # Latent rescaling
                target_latent_std=target_latent_std,
                # Other
                conditioning_mask=conditioning_mask,
                conditioning_features=conditioning_features,
                speaker_latent=speaker_latent,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
                token_durations=token_durations,
                prepared_conditioning=prepared_conditioning,
                prepared_cfg_conditioning=prepared_cfg_conditioning,
                fuse_cfg_branches=cache_mode == "cache_dit",
                show_progress=show_progress,
                device=self.device,
                generative_objective=generative_objective,
                diffusion_schedule_shift=diffusion_schedule_shift,
            )
        self.last_cache_stats = (
            cache_session.stats if isinstance(cache_session, CacheDiTSession) else CacheDiTStats()
        )

        # Decode with DACVAE
        print("Decoding audio with DACVAE...")
        audio = self._decode_generated_latents(generated_latents)  # [1, 1, samples]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_time_s = time.perf_counter() - generation_started_at
        print(f"Generation time: {generation_time_s:.2f}s")

        # Return as 1D tensor
        audio = audio.squeeze().cpu()

        return audio

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
        normalized_text: str | None = None,
        phonemes: str | Sequence[str] | None = None,
        language_spans: Sequence[TextSpan | Mapping[str, object]] | None = None,
    ) -> torch.Tensor:
        """Synthesize using one named or custom typed generation profile."""
        return self.synthesize(
            text,
            num_steps=config.num_steps,
            solver=config.solver,
            cfg_scale=config.cfg_scale,
            cfg_mode=config.cfg_mode,
            cfg_scale_text=config.cfg_scale_text,
            cfg_scale_speaker=config.cfg_scale_speaker,
            cfg_min_t=config.cfg_min_t,
            cfg_max_t=config.cfg_max_t,
            initial_noise_scale=config.initial_noise_scale,
            temporal_rescale_k=config.temporal_rescale_k,
            temporal_rescale_sigma=config.temporal_rescale_sigma,
            target_latent_std=config.target_latent_std,
            cache_mode=config.cache_mode,
            duration=duration,
            show_progress=show_progress,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            speaker_latent=speaker_latent,
            language=language,
            reference_language=reference_language,
            normalized_text=normalized_text,
            phonemes=phonemes,
            language_spans=language_spans,
        )

    @torch.inference_mode()
    def synthesize_batch(
        self,
        texts: list[str],
        num_steps: int = 32,
        cfg_scale: float = 1.0,
        solver: str | None = None,
        max_duration: float | None = None,
        languages: str | list[str] | tuple[str, ...] | None = None,
        phonemes: Sequence[str | Sequence[str] | None] | None = None,
        language_spans: (Sequence[Sequence[TextSpan | Mapping[str, object]] | None] | None) = None,
    ) -> list[torch.Tensor]:
        """
        Synthesize multiple texts in batch.

        Args:
            texts: List of text strings
            num_steps: ODE solver steps
            cfg_scale: Guidance scale
            solver: Solver type
            max_duration: Optional validation ceiling per sample. ``None`` uses the
                configured inference ceiling. Requests above the selected ceiling fail;
                audio is never truncated to make a batch.

        Returns:
            List of audio tensors
        """
        self.last_cache_stats = CacheDiTStats()
        assert_cache_dit_healthy(self.flow_model)
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must contain only strings.")
        generative_objective, diffusion_schedule_shift = _runtime_generation_contract(self)
        if solver is None:
            solver = "ddim" if generative_objective == VP_DIFFUSION_OBJECTIVE else "euler"
        minimum_duration = self.settings.duration.minimum_seconds
        configured_maximum = self.settings.duration.maximum_seconds
        selected_maximum = configured_maximum if max_duration is None else max_duration
        try:
            valid_maximum = not isinstance(selected_maximum, bool) and math.isfinite(
                selected_maximum
            )
        except TypeError:
            valid_maximum = False
        if not valid_maximum:
            raise ValueError("max_duration must be a finite number.")
        if selected_maximum < minimum_duration:
            raise ValueError(f"max_duration must be at least {minimum_duration:g} seconds.")
        if selected_maximum > configured_maximum:
            raise ValueError(
                f"max_duration cannot exceed the configured maximum of "
                f"{configured_maximum:g} seconds."
            )
        if languages is None or isinstance(languages, str):
            selected_languages = [languages] * len(texts)
        else:
            selected_languages = list(languages)
            if len(selected_languages) != len(texts):
                raise ValueError("languages must contain one entry per text.")
        if phonemes is None:
            selected_phonemes: list[str | Sequence[str] | None] = [None] * len(texts)
        else:
            if isinstance(phonemes, (str, bytes)):
                raise TypeError("phonemes must contain one phone sequence per text.")
            selected_phonemes = list(phonemes)
            if len(selected_phonemes) != len(texts):
                raise ValueError("phonemes must contain one entry per text.")
        if language_spans is None:
            selected_spans: list[Sequence[TextSpan | Mapping[str, object]] | None] = [None] * len(
                texts
            )
        else:
            if isinstance(language_spans, (str, bytes)):
                raise TypeError("language_spans must contain one span sequence per text.")
            selected_spans = list(language_spans)
            if len(selected_spans) != len(texts):
                raise ValueError("language_spans must contain one entry per text.")

        # Resolve and validate every request before allocating generation state.
        # Requests are grouped by exact latent length, so each ODE call is a real
        # tensor batch without allowing padded target noise to affect valid audio.
        prepared_requests = []
        for index, (text, language, request_phonemes, request_spans) in enumerate(
            zip(texts, selected_languages, selected_phonemes, selected_spans)
        ):
            language_pair = self._resolve_language_pair(
                language,
                None,
                has_reference=False,
            )
            if (
                getattr(self, "text_conditioning_mode", "scratch_tokens") == "frozen_features"
                and getattr(self.frozen_text_provider.spec, "frozen_text_frontend", None)
                == "phonemes"
                and request_phonemes is None
                and not request_spans
            ):
                raise ValueError(
                    f"Frozen phoneme request {index} requires canonical phonemes or "
                    "phoneme-bearing language_spans; raw text fallback is forbidden."
                )
            prepared_text = self._prepare_conditioning(
                text,
                language_pair.target,
                phonemes=request_phonemes,
                language_spans=request_spans,
            )
            language_ids = self._language_ids(language_pair)
            resolved_duration, num_frames = self._resolve_duration_shape(
                text,
                None,
                prepared_text.conditioning_ids,
                language_ids,
                None,
                conditioning_mask=prepared_text.conditioning_mask,
                conditioning_features=prepared_text.conditioning_features,
                token_language_ids=prepared_text.token_language_ids,
                alignment_mask=prepared_text.alignment_mask,
            )
            if resolved_duration > selected_maximum:
                raise ValueError(
                    f"Request {index} resolves to {resolved_duration:g}s, exceeding the batch "
                    f"maximum of {selected_maximum:g}s. Increase max_duration within the "
                    "configured limit or split the text; batching never truncates audio."
                )
            prepared_requests.append((index, prepared_text, language_ids, num_frames))

        requests_by_frames: dict[
            int,
            list[tuple[int, _PreparedText, torch.Tensor | None, int]],
        ] = {}
        for request in prepared_requests:
            requests_by_frames.setdefault(request[3], []).append(request)

        audios: list[torch.Tensor] = [torch.empty(0) for _ in texts]
        effective_cfg = self._effective_cfg(
            cfg_scale=cfg_scale,
            cfg_mode="joint",
            cfg_scale_text=None,
            cfg_scale_speaker=None,
            speaker_latent=None,
        )
        effective_cfg_scale, effective_cfg_mode, text_scale, speaker_scale = effective_cfg

        for num_frames, requests in requests_by_frames.items():
            batch_size = len(requests)
            max_text_tokens = max(request[1].conditioning_ids.shape[1] for request in requests)
            has_text_padding = any(
                request[1].conditioning_ids.shape[1] != max_text_tokens for request in requests
            )
            feature_presence = [
                request[1].conditioning_features is not None for request in requests
            ]
            if any(feature_presence) and not all(feature_presence):
                raise RuntimeError(
                    "A batch cannot mix scratch-token and frozen-feature conditioning."
                )
            batched_conditioning = torch.full(
                (batch_size, max_text_tokens),
                getattr(self, "text_pad_token", PAD_TOKEN),
                dtype=torch.long,
                device=self.device,
            )
            conditioning_mask = (
                torch.zeros(
                    batch_size,
                    max_text_tokens,
                    dtype=torch.bool,
                    device=self.device,
                )
                if has_text_padding
                or any(request[1].conditioning_mask is not None for request in requests)
                else None
            )
            conditioning_features_batch = None
            if all(feature_presence):
                first_features = requests[0][1].conditioning_features
                assert first_features is not None
                conditioning_features_batch = torch.zeros(
                    batch_size,
                    max_text_tokens,
                    first_features.shape[-1],
                    dtype=first_features.dtype,
                    device=self.device,
                )
            token_language_batch = torch.zeros_like(batched_conditioning)
            alignment_batch = torch.zeros_like(batched_conditioning, dtype=torch.bool)
            for batch_index, (_, prepared_text, _, _) in enumerate(requests):
                conditioning_ids = prepared_text.conditioning_ids
                token_count = conditioning_ids.shape[1]
                batched_conditioning[batch_index, :token_count] = conditioning_ids[0]
                token_language_batch[batch_index, :token_count] = prepared_text.token_language_ids[
                    0
                ]
                alignment_batch[batch_index, :token_count] = prepared_text.alignment_mask[0]
                if conditioning_mask is not None:
                    if prepared_text.conditioning_mask is None:
                        conditioning_mask[batch_index, :token_count] = True
                    else:
                        conditioning_mask[batch_index, :token_count] = (
                            prepared_text.conditioning_mask[0]
                        )
                if conditioning_features_batch is not None:
                    assert prepared_text.conditioning_features is not None
                    conditioning_features_batch[batch_index, :token_count] = (
                        prepared_text.conditioning_features[0]
                    )

            language_batch = None
            if requests and requests[0][2] is not None:
                if any(request[2] is None for request in requests):
                    raise RuntimeError("A batch cannot mix language-conditioned and legacy rows.")
                language_batch = torch.cat(
                    [request[2] for request in requests if request[2] is not None],
                    dim=0,
                )

            token_durations = self._resolve_token_durations(
                batched_conditioning,
                num_frames=num_frames,
                conditioning_mask=conditioning_mask,
                language_ids=language_batch,
                token_language_ids=token_language_batch,
                alignment_mask=alignment_batch,
                conditioning_features=conditioning_features_batch,
            )

            generated_latents = ODESolver.sample(
                model=self.flow_model,
                conditioning_ids=batched_conditioning,
                num_steps=num_steps,
                solver=solver,
                latent_shape=(batch_size, self.latent_size, num_frames),
                cfg_scale=effective_cfg_scale,
                cfg_mode=effective_cfg_mode,
                cfg_scale_text=text_scale,
                cfg_scale_speaker=speaker_scale,
                cfg_min_t=0.0,
                cfg_max_t=1.0,
                initial_noise_scale=1.0,
                temporal_rescale_k=1.0,
                temporal_rescale_sigma=2.5,
                target_latent_std=None,
                conditioning_mask=conditioning_mask,
                conditioning_features=conditioning_features_batch,
                language_ids=language_batch,
                token_language_ids=token_language_batch,
                alignment_mask=alignment_batch,
                token_durations=token_durations,
                fuse_cfg_branches=False,
                show_progress=False,
                device=self.device,
                generative_objective=generative_objective,
                diffusion_schedule_shift=diffusion_schedule_shift,
            )
            decoded = self._decode_generated_latents(generated_latents)
            if decoded.ndim == 0 or decoded.shape[0] != batch_size:
                raise RuntimeError(
                    "DACVAE batch decode must preserve the generated batch dimension."
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            for batch_index, (request_index, _, _, _) in enumerate(requests):
                audios[request_index] = decoded[batch_index].squeeze().cpu()

        return audios

    def save_audio(self, audio: torch.Tensor, path: str):
        """
        Save audio to file.

        Args:
            audio: Audio tensor [samples] or [1, samples]
            path: Output file path (e.g., "output.wav")
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # [1, samples]

        torchaudio.save(path, audio, self.sample_rate)
        print(f"✓ Saved audio to {path}")

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        num_steps: int = 64,
        cfg_scale: float = 1.0,
        solver: str | None = None,
        duration: float | None = None,
        # Neutral conditional inference; guidance requires checkpoint-specific calibration.
        cfg_mode: str = "joint",
        cfg_scale_text: float | None = None,
        cfg_scale_speaker: float | None = None,
        cfg_min_t: float = 0.0,
        cfg_max_t: float = 1.0,
        initial_noise_scale: float = 1.0,
        temporal_rescale_k: float = 1.0,
        temporal_rescale_sigma: float = 2.5,
        target_latent_std: float | None = None,
        cache_mode: str = "none",
        reference_audio: AudioReference | None = None,
        reference_sample_rate: int | None = None,
        speaker_latent: torch.Tensor | None = None,
        language: str | None = None,
        reference_language: str | None = None,
        normalized_text: str | None = None,
        phonemes: str | Sequence[str] | None = None,
        language_spans: Sequence[TextSpan | Mapping[str, object]] | None = None,
    ):
        audio = self.synthesize(
            text=text,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            solver=solver,
            duration=duration,
            show_progress=True,
            cfg_mode=cfg_mode,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            initial_noise_scale=initial_noise_scale,
            temporal_rescale_k=temporal_rescale_k,
            temporal_rescale_sigma=temporal_rescale_sigma,
            target_latent_std=target_latent_std,
            cache_mode=cache_mode,
            reference_audio=reference_audio,
            reference_sample_rate=reference_sample_rate,
            speaker_latent=speaker_latent,
            language=language,
            reference_language=reference_language,
            normalized_text=normalized_text,
            phonemes=phonemes,
            language_spans=language_spans,
        )
        self.save_audio(audio, output_path)
