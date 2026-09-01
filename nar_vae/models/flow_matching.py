import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn as nn

from nar_vae.languages import (
    DEFAULT_LANGUAGE,
    LANGUAGE_CONDITIONING_VERSION,
    LANGUAGE_COUNT,
    LANGUAGE_REGISTRY_VERSION,
    NULL_LANGUAGE_ID,
    Language,
    LanguagePair,
    language_id,
    normalize_languages,
    resolve_language_pair_support,
)
from nar_vae.voice import (
    CROSS_LINGUAL_CAPABILITY_VERSION,
    SPEAKER_CONDITIONING_VERSION,
    SPEAKER_PATCH_LAYOUT_VERSION,
)

from .alignment import monotonic_alignment_search
from .dit import EchoDiT
from .duration import (
    DURATION_PREDICTOR_VERSION,
    ECHODIT_ARCHITECTURE_VERSION,
    MONOTONIC_ALIGNMENT_VERSION,
    DurationAlignmentOutput,
    EchoDurationAlignment,
    EchoDurationPredictor,
    allocate_positive_token_durations,
    expand_text_by_durations,
)


def _cfg_dropout_probability(value: float, *, name: str) -> float:
    """Validate a direct model-construction CFG dropout probability."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1].")
    return probability


@dataclass(frozen=True)
class PreparedConditioning:
    """Device-resident conditioning state reused across ODE evaluations."""

    text_mask: torch.Tensor | None
    speaker_mask: torch.Tensor | None
    kv_cache_text: list[tuple[torch.Tensor, torch.Tensor]]
    kv_cache_speaker: list[tuple[torch.Tensor, torch.Tensor]]
    frame_text_state: torch.Tensor | None

    def slice_batch(self, start: int, stop: int) -> "PreparedConditioning":
        """Return a batch slice as views without recomputing its encoders."""
        return PreparedConditioning(
            text_mask=(self.text_mask[start:stop] if self.text_mask is not None else None),
            speaker_mask=(self.speaker_mask[start:stop] if self.speaker_mask is not None else None),
            kv_cache_text=[
                (key[start:stop], value[start:stop]) for key, value in self.kv_cache_text
            ],
            kv_cache_speaker=[
                (key[start:stop], value[start:stop]) for key, value in self.kv_cache_speaker
            ],
            frame_text_state=(
                self.frame_text_state[start:stop] if self.frame_text_state is not None else None
            ),
        )


@dataclass(frozen=True)
class PreparedCFGConditioning:
    """Precomputed fused batches for one classifier-free-guidance mode."""

    mode: str
    branch_count: int
    variants: tuple[PreparedConditioning, ...]
    conditional: PreparedConditioning


@dataclass(frozen=True)
class EncodedConditioning:
    """Normalized encoder states before per-layer KV and MAS-frame projection."""

    text_mask: torch.Tensor | None
    speaker_mask: torch.Tensor | None
    text_state: torch.Tensor
    speaker_state: torch.Tensor

    def slice_batch(self, start: int, stop: int) -> "EncodedConditioning":
        """Return a contiguous batch slice without running either encoder again."""
        return EncodedConditioning(
            text_mask=(self.text_mask[start:stop] if self.text_mask is not None else None),
            speaker_mask=(self.speaker_mask[start:stop] if self.speaker_mask is not None else None),
            text_state=self.text_state[start:stop],
            speaker_state=self.speaker_state[start:stop],
        )

    def index_select_batch(self, indices: torch.Tensor) -> "EncodedConditioning":
        """Select arbitrary request rows while retaining device-resident encoder states."""
        indices = indices.to(device=self.text_state.device, dtype=torch.long)
        return EncodedConditioning(
            text_mask=(
                self.text_mask.index_select(0, indices) if self.text_mask is not None else None
            ),
            speaker_mask=(
                self.speaker_mask.index_select(0, indices)
                if self.speaker_mask is not None
                else None
            ),
            text_state=self.text_state.index_select(0, indices),
            speaker_state=self.speaker_state.index_select(0, indices),
        )


@dataclass(frozen=True)
class EncodedCFGConditioning:
    """One encoder pass containing conditional and optional fused CFG branches."""

    mode: str | None
    branch_count: int
    variants: tuple[EncodedConditioning, ...]
    conditional: EncodedConditioning

    @property
    def batch_size(self) -> int:
        return int(self.conditional.text_state.shape[0])

    def index_select_requests(self, indices: torch.Tensor) -> "EncodedCFGConditioning":
        """Select request rows from every branch-major CFG variant."""
        indices = indices.to(device=self.conditional.text_state.device, dtype=torch.long)
        batch_size = self.batch_size
        variant_indices = torch.cat(
            [indices + branch_index * batch_size for branch_index in range(self.branch_count)],
            dim=0,
        )
        variants = tuple(variant.index_select_batch(variant_indices) for variant in self.variants)
        return EncodedCFGConditioning(
            mode=self.mode,
            branch_count=self.branch_count,
            variants=variants,
            conditional=self.conditional.index_select_batch(indices),
        )


class FlowMatchingEchoDiT(nn.Module):
    """
    Flow Matching TTS with EchoDiT architecture.

    Wraps EchoDiT to work with flow matching training and inference.
    """

    def __init__(
        self,
        latent_size: int = 1024,  # DACVAE latent dimension
        model_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        #
        text_vocab_size: int = 152000,
        text_model_size: int = 768,
        text_num_layers: int = 6,
        text_num_heads: int = 12,
        text_intermediate_size: int = 3072,
        #
        speaker_patch_size: int = 4,
        speaker_model_size: int = 512,
        speaker_num_layers: int = 4,
        speaker_num_heads: int = 8,
        speaker_intermediate_size: int = 2048,
        #
        timestep_embed_size: int = 256,
        adaln_rank: int = 128,
        norm_eps: float = 1e-6,
        #
        cfg_dropout: float = 0.1,
        cfg_dropout_text: float | None = None,
        cfg_dropout_speaker: float | None = None,
        use_speaker_conditioning: bool = False,
        use_language_conditioning: bool = False,
        supported_languages: tuple[str, ...] | list[str] | None = None,
        supported_reference_languages: tuple[str, ...] | list[str] | None = None,
        supported_language_pairs: (
            Iterable[LanguagePair | Sequence[str | Language]] | LanguagePair | None
        ) = None,
        use_duration_predictor: bool = False,
        duration_predictor_hidden_size: int = 256,
        duration_predictor_num_layers: int = 2,
        duration_predictor_use_speaker: bool = False,
        use_mas_duration: bool = False,
        duration_alignment_hidden_size: int = 64,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.cfg_dropout = _cfg_dropout_probability(cfg_dropout, name="cfg_dropout")
        self.cfg_dropout_text = (
            self.cfg_dropout
            if cfg_dropout_text is None
            else _cfg_dropout_probability(cfg_dropout_text, name="cfg_dropout_text")
        )
        self.cfg_dropout_speaker = (
            self.cfg_dropout
            if cfg_dropout_speaker is None
            else _cfg_dropout_probability(cfg_dropout_speaker, name="cfg_dropout_speaker")
        )
        self.use_speaker_conditioning = use_speaker_conditioning
        self.speaker_patch_size = speaker_patch_size
        self.use_language_conditioning = use_language_conditioning
        self.use_duration_predictor = use_duration_predictor
        self.duration_predictor_use_speaker = duration_predictor_use_speaker
        self.use_mas_duration = use_mas_duration
        if use_mas_duration and not use_duration_predictor:
            raise ValueError("use_mas_duration requires use_duration_predictor=True.")
        if duration_predictor_use_speaker and not use_duration_predictor:
            raise ValueError("duration_predictor_use_speaker requires use_duration_predictor=True.")
        if duration_predictor_use_speaker and not use_speaker_conditioning:
            raise ValueError(
                "Speaker-conditioned duration prediction requires speaker conditioning."
            )
        self.supported_languages = (
            normalize_languages(supported_languages) if use_language_conditioning else ()
        )
        if (
            use_speaker_conditioning
            and use_language_conditioning
            and supported_reference_languages is None
            and supported_language_pairs is None
        ):
            raise ValueError(
                "Speaker-conditioned multilingual models require exact supported_language_pairs."
            )
        if (
            supported_reference_languages is not None or supported_language_pairs is not None
        ) and not (use_speaker_conditioning and use_language_conditioning):
            raise ValueError(
                "Reference-language pair support requires both speaker and language conditioning."
            )
        if use_speaker_conditioning and use_language_conditioning:
            (
                self.supported_reference_languages,
                self.supported_language_pairs,
            ) = resolve_language_pair_support(
                self.supported_languages,
                supported_reference_languages=supported_reference_languages,
                supported_language_pairs=supported_language_pairs,
            )
        else:
            self.supported_reference_languages = ()
            self.supported_language_pairs = ()

        # EchoDiT backbone
        self.dit = EchoDiT(
            latent_size=latent_size,
            model_size=model_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            norm_eps=norm_eps,
            text_vocab_size=text_vocab_size,
            text_model_size=text_model_size,
            text_num_layers=text_num_layers,
            text_num_heads=text_num_heads,
            text_intermediate_size=text_intermediate_size,
            speaker_patch_size=speaker_patch_size,
            speaker_model_size=speaker_model_size,
            speaker_num_layers=speaker_num_layers,
            speaker_num_heads=speaker_num_heads,
            speaker_intermediate_size=speaker_intermediate_size,
            timestep_embed_size=timestep_embed_size,
            adaln_rank=adaln_rank,
            num_languages=LANGUAGE_COUNT if use_language_conditioning else 0,
            use_speaker_conditioning=use_speaker_conditioning,
            use_duration_alignment=use_mas_duration,
        )
        if use_duration_predictor:
            self.duration_predictor = EchoDurationPredictor(
                text_size=text_model_size,
                hidden_size=duration_predictor_hidden_size,
                num_layers=duration_predictor_num_layers,
                speaker_size=(speaker_model_size if duration_predictor_use_speaker else None),
            )
            self.register_buffer(
                "echodit_architecture_version",
                torch.tensor(ECHODIT_ARCHITECTURE_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "duration_predictor_version",
                torch.tensor(DURATION_PREDICTOR_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "duration_predictor_hidden_size_metadata",
                torch.tensor(duration_predictor_hidden_size, dtype=torch.int32),
            )
            self.register_buffer(
                "duration_predictor_num_layers_metadata",
                torch.tensor(duration_predictor_num_layers, dtype=torch.int32),
            )
            self.register_buffer(
                "duration_predictor_uses_speaker_metadata",
                torch.tensor(int(duration_predictor_use_speaker), dtype=torch.int32),
            )
            if use_mas_duration:
                self.duration_alignment = EchoDurationAlignment(
                    text_size=text_model_size,
                    latent_size=latent_size,
                    hidden_size=duration_alignment_hidden_size,
                )
                self.register_buffer(
                    "duration_alignment_version",
                    torch.tensor(MONOTONIC_ALIGNMENT_VERSION, dtype=torch.int32),
                )
                self.register_buffer(
                    "duration_alignment_hidden_size_metadata",
                    torch.tensor(duration_alignment_hidden_size, dtype=torch.int32),
                )

        # Null embeddings for unconditional generation (CFG)
        self.register_buffer("null_text_embed", torch.zeros(1, 1, text_model_size))

        if use_speaker_conditioning:
            # Null speaker must have time_dim = patch_size (minimum for patching)
            self.register_buffer(
                "null_speaker_embed", torch.zeros(1, latent_size, speaker_patch_size)
            )
            self.register_buffer(
                "speaker_conditioning_version",
                torch.tensor(SPEAKER_CONDITIONING_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "speaker_patch_layout_version",
                torch.tensor(SPEAKER_PATCH_LAYOUT_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "speaker_patch_size_metadata",
                torch.tensor(speaker_patch_size, dtype=torch.int32),
            )

        if use_language_conditioning:
            supported_ids = tuple(language_id(code) for code in self.supported_languages)
            self.register_buffer(
                "language_conditioning_version",
                torch.tensor(LANGUAGE_CONDITIONING_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "language_registry_version",
                torch.tensor(LANGUAGE_REGISTRY_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "language_count_metadata",
                torch.tensor(LANGUAGE_COUNT, dtype=torch.int32),
            )
            self.register_buffer(
                "supported_language_ids_metadata",
                torch.tensor(supported_ids, dtype=torch.int32),
            )

        if self.supported_language_pairs:
            reference_ids = tuple(language_id(code) for code in self.supported_reference_languages)
            pair_ids = tuple(
                (language_id(pair.target), language_id(pair.reference))
                for pair in self.supported_language_pairs
            )
            self.register_buffer(
                "cross_lingual_capability_version",
                torch.tensor(CROSS_LINGUAL_CAPABILITY_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "reference_language_registry_version",
                torch.tensor(LANGUAGE_REGISTRY_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "supported_reference_language_ids_metadata",
                torch.tensor(reference_ids, dtype=torch.int32),
            )
            self.register_buffer(
                "supported_language_pair_ids_metadata",
                torch.tensor(pair_ids, dtype=torch.int32),
            )

    def _prepare_language_ids(
        self,
        language_ids: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Validate language IDs and apply the legacy English default."""
        uses_language_conditioning = getattr(self, "use_language_conditioning", False)
        if language_ids is None:
            if not uses_language_conditioning:
                return None
            return torch.full(
                (batch_size,),
                language_id(DEFAULT_LANGUAGE),
                dtype=torch.long,
                device=device,
            )
        if language_ids.ndim != 1 or language_ids.shape[0] != batch_size:
            raise ValueError("language_ids must have shape [batch].")
        language_ids = language_ids.to(device=device, dtype=torch.long)

        if not uses_language_conditioning:
            default_id = language_id(DEFAULT_LANGUAGE)
            if torch.any((language_ids != default_id) & (language_ids != NULL_LANGUAGE_ID)):
                raise RuntimeError(
                    "Non-English language IDs require a language-conditioned checkpoint."
                )
            return None

        supported = self.supported_language_ids_metadata.to(device=device, dtype=torch.long)
        valid = (language_ids[:, None] == supported[None, :]).any(dim=1)
        valid = valid | (language_ids == NULL_LANGUAGE_ID)
        if not bool(valid.all()):
            invalid = sorted(set(language_ids[~valid].detach().cpu().tolist()))
            raise ValueError(
                f"language_ids contain values not supported by this checkpoint: {invalid}."
            )
        return language_ids

    def prepare_inference_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_durations: torch.Tensor | None = None,
    ) -> PreparedConditioning:
        """Encode invariant text/speaker context once for an ODE trajectory."""
        encoded = self.encode_inference_conditioning(
            conditioning_ids,
            attention_mask,
            speaker_latent,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
        )
        prepared, _ = self.finalize_inference_conditioning(
            encoded,
            token_durations=token_durations,
        )
        return prepared

    def encode_inference_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        *,
        cfg_mode: str | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> EncodedCFGConditioning:
        """Encode all request and CFG invariants in one branch-major encoder batch."""
        if cfg_mode not in {None, "joint", "independent", "alternating"}:
            raise ValueError(f"Unknown CFG mode: {cfg_mode}")
        batch_size = conditioning_ids.shape[0]
        device = conditioning_ids.device
        text_mask = attention_mask.bool() if attention_mask is not None else None
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=batch_size,
            device=device,
        )

        def null_speaker_like(reference: torch.Tensor | None) -> torch.Tensor | None:
            if not self.use_speaker_conditioning or reference is None:
                return None
            null_speaker = self.null_speaker_embed.expand(batch_size, -1, -1)
            reference_frames = reference.shape[-1]
            if null_speaker.shape[-1] < reference_frames:
                null_speaker = torch.nn.functional.pad(
                    null_speaker,
                    (0, reference_frames - null_speaker.shape[-1]),
                )
            return null_speaker[..., :reference_frames]

        conditional = (conditioning_ids, speaker_latent, language_ids)
        if cfg_mode is None:
            branch_count = 1
            branch_variants = ((conditional,),)
        else:
            null_ids = torch.zeros_like(conditioning_ids)
            null_language_ids = torch.zeros_like(language_ids) if language_ids is not None else None
            null_speaker = null_speaker_like(speaker_latent)
            if cfg_mode == "joint":
                branch_count = 2
                branch_variants = (
                    (
                        conditional,
                        (null_ids, null_speaker, null_language_ids),
                    ),
                )
            elif cfg_mode == "independent":
                branch_count = 3
                branch_variants = (
                    (
                        conditional,
                        (null_ids, speaker_latent, null_language_ids),
                        (conditioning_ids, null_speaker, language_ids),
                    ),
                )
            else:
                branch_count = 2
                branch_variants = (
                    (
                        conditional,
                        (null_ids, speaker_latent, null_language_ids),
                    ),
                    (
                        conditional,
                        (conditioning_ids, null_speaker, language_ids),
                    ),
                )

        flat_branches = tuple(branch for variant in branch_variants for branch in variant)
        flat_conditioning_ids = torch.cat([branch[0] for branch in flat_branches], dim=0)
        flat_text_mask = (
            torch.cat([text_mask] * len(flat_branches), dim=0) if text_mask is not None else None
        )
        flat_language_ids = (
            torch.cat([branch[2] for branch in flat_branches], dim=0)
            if all(branch[2] is not None for branch in flat_branches)
            else None
        )
        flat_text_state = self.dit.encode_text(
            flat_conditioning_ids,
            flat_text_mask,
            flat_language_ids,
        )

        flat_speaker_latent = None
        if any(branch[1] is not None for branch in flat_branches):
            if not all(branch[1] is not None for branch in flat_branches):
                raise ValueError("Speaker CFG branches must either all be tensors or all be None.")
            flat_speaker_latent = torch.cat(
                [branch[1] for branch in flat_branches if branch[1] is not None],
                dim=0,
            )
        flat_speaker_mask = (
            torch.cat([speaker_mask] * len(flat_branches), dim=0)
            if speaker_mask is not None
            else None
        )
        flat_speaker_latent, flat_speaker_mask = self._prepare_speaker_inputs(
            flat_speaker_latent,
            flat_speaker_mask,
            batch_size=batch_size * len(flat_branches),
            device=device,
        )
        encode_speaker = getattr(self.dit, "encode_speaker", None)
        flat_speaker_state = (
            encode_speaker(flat_speaker_latent, flat_speaker_mask)
            if callable(encode_speaker)
            else flat_speaker_latent
        )

        variant_size = branch_count * batch_size
        variants = tuple(
            EncodedConditioning(
                text_mask=(
                    flat_text_mask[offset : offset + variant_size]
                    if flat_text_mask is not None
                    else None
                ),
                speaker_mask=(
                    flat_speaker_mask[offset : offset + variant_size]
                    if flat_speaker_mask is not None
                    else None
                ),
                text_state=flat_text_state[offset : offset + variant_size],
                speaker_state=flat_speaker_state[offset : offset + variant_size],
            )
            for offset in range(0, len(flat_branches) * batch_size, variant_size)
        )
        return EncodedCFGConditioning(
            mode=cfg_mode,
            branch_count=branch_count,
            variants=variants,
            conditional=variants[0].slice_batch(0, batch_size),
        )

    def finalize_inference_conditioning(
        self,
        encoded: EncodedCFGConditioning,
        *,
        token_durations: torch.Tensor | None = None,
    ) -> tuple[PreparedConditioning, PreparedCFGConditioning | None]:
        """Project encoded states after the runtime has fixed the exact frame allocation."""
        batch_size = encoded.batch_size
        if getattr(self, "use_mas_duration", False):
            if token_durations is None:
                raise ValueError(
                    "A MAS-duration checkpoint requires token_durations for ODE conditioning."
                )
            if tuple(token_durations.shape) != tuple(encoded.conditional.text_state.shape[:2]):
                raise ValueError("token_durations must have the encoded conditioning shape.")
        elif token_durations is not None:
            raise ValueError("token_durations require a duration-aligned NAR-VAE checkpoint.")

        def finalize_variant(variant: EncodedConditioning) -> PreparedConditioning:
            variant_durations = None
            if token_durations is not None:
                variant_durations = torch.cat([token_durations] * encoded.branch_count, dim=0)
            project_speaker = getattr(self.dit, "project_speaker_kv_cache", None)
            kv_cache_speaker = (
                project_speaker(variant.speaker_state)
                if callable(project_speaker)
                else self.dit.get_kv_cache_speaker(variant.speaker_state, variant.speaker_mask)
            )
            return PreparedConditioning(
                text_mask=variant.text_mask,
                speaker_mask=variant.speaker_mask,
                kv_cache_text=self.dit.project_text_kv_cache(variant.text_state),
                kv_cache_speaker=kv_cache_speaker,
                frame_text_state=(
                    expand_text_by_durations(variant.text_state, variant_durations)
                    if variant_durations is not None
                    else None
                ),
            )

        variants = tuple(finalize_variant(variant) for variant in encoded.variants)
        conditional = variants[0].slice_batch(0, batch_size)
        if encoded.mode is None:
            return conditional, None
        return conditional, PreparedCFGConditioning(
            mode=encoded.mode,
            branch_count=encoded.branch_count,
            variants=variants,
            conditional=conditional,
        )

    def _prepare_speaker_inputs(
        self,
        speaker_latent: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Validate speaker inputs and reduce frame masks to encoder patches."""
        if not self.use_speaker_conditioning:
            if speaker_latent is not None or speaker_mask is not None:
                raise ValueError("Speaker inputs require a speaker-conditioned NAR-VAE checkpoint.")
            return (
                torch.empty(
                    batch_size,
                    self.latent_size,
                    0,
                    device=device,
                    dtype=self.dit.in_proj.weight.dtype,
                ),
                None,
            )
        if self.use_speaker_conditioning and speaker_latent is not None:
            if speaker_latent.shape[2] % self.dit.speaker_patch_size:
                raise ValueError(
                    f"Speaker latent has {speaker_latent.shape[2]} frames; expected a multiple "
                    f"of patch size {self.dit.speaker_patch_size}."
                )
            num_speaker_patches = speaker_latent.shape[2] // self.dit.speaker_patch_size
            if speaker_mask is None:
                speaker_attention_mask = None
            else:
                if speaker_mask.ndim != 2 or speaker_mask.shape[0] != batch_size:
                    raise ValueError(
                        "speaker_mask must have shape [batch, frames] or [batch, patches]"
                    )
                speaker_attention_mask = speaker_mask.to(device=device, dtype=torch.bool)
                if speaker_attention_mask.shape[1] == speaker_latent.shape[2]:
                    speaker_latent = speaker_latent.masked_fill(
                        ~speaker_attention_mask[:, None, :],
                        0.0,
                    )
                    speaker_attention_mask = speaker_attention_mask.reshape(
                        batch_size,
                        num_speaker_patches,
                        self.dit.speaker_patch_size,
                    ).any(dim=-1)
                elif speaker_attention_mask.shape[1] != num_speaker_patches:
                    raise ValueError(
                        "speaker_mask length must match the speaker frame or patch count"
                    )
            return speaker_latent, speaker_attention_mask

        return self.null_speaker_embed.expand(batch_size, -1, -1), None

    def predict_log_duration(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``log1p`` DACVAE frame count with a trained EchoDiT v2 head."""
        text_state, text_mask, speaker_state, speaker_attention_mask = (
            self._encode_duration_conditioning(
                conditioning_ids,
                attention_mask,
                speaker_latent,
                speaker_mask,
                language_ids,
            )
        )
        return self._predict_log_duration_from_encoded(
            text_state,
            text_mask,
            speaker_state,
            speaker_attention_mask,
        )

    def _encode_duration_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        speaker_latent: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        language_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Encode complete duration conditioning once for total or token predictions."""
        if not getattr(self, "use_duration_predictor", False):
            raise RuntimeError(
                "Learned duration requires a versioned EchoDiT v2 duration checkpoint."
            )
        batch_size = conditioning_ids.shape[0]
        device = conditioning_ids.device
        text_mask = attention_mask.bool() if attention_mask is not None else None
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=batch_size,
            device=device,
        )
        text_state = self.dit.encode_text(conditioning_ids, text_mask, language_ids)

        speaker_state = None
        speaker_attention_mask = None
        if self.duration_predictor_use_speaker:
            speaker_latent, speaker_attention_mask = self._prepare_speaker_inputs(
                speaker_latent,
                speaker_mask,
                batch_size=batch_size,
                device=device,
            )
            speaker_state = self.dit.encode_speaker(speaker_latent, speaker_attention_mask)

        return text_state, text_mask, speaker_state, speaker_attention_mask

    def predict_expected_token_durations(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict positive floating-point frame contributions for each valid token."""
        if not getattr(self, "use_mas_duration", False):
            raise RuntimeError(
                "Token duration prediction requires a versioned MAS-duration checkpoint."
            )
        text_state, text_mask, speaker_state, speaker_attention_mask = (
            self._encode_duration_conditioning(
                conditioning_ids,
                attention_mask,
                speaker_latent,
                speaker_mask,
                language_ids,
            )
        )
        _, token_durations = self._predict_duration_components_from_encoded(
            text_state,
            text_mask,
            speaker_state,
            speaker_attention_mask,
        )
        return token_durations

    def _predict_log_duration_from_encoded(
        self,
        text_state: torch.Tensor,
        text_mask: torch.Tensor | None,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the duration head on encoder states that can also feed the velocity path."""
        return self.duration_predictor(text_state, text_mask, speaker_state, speaker_mask)

    def _predict_duration_components_from_encoded(
        self,
        text_state: torch.Tensor,
        text_mask: torch.Tensor | None,
        speaker_state: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return total and per-token duration values without a second encoder pass."""
        prediction = self.duration_predictor(
            text_state,
            text_mask,
            speaker_state,
            speaker_mask,
            return_token_durations=True,
        )
        assert isinstance(prediction, tuple)
        return prediction

    @staticmethod
    def _duration_frames_from_log(log_frames: torch.Tensor) -> torch.Tensor:
        """Convert validated ``log1p`` predictions to positive integral frame counts."""
        # Direct float-to-int conversion maps NaN, infinity, and overflow to a
        # platform sentinel. That sentinel can later be minimum-clamped into a
        # plausible duration, so validate in float64 before conversion.
        log_frames = log_frames.to(dtype=torch.float64)
        if not bool(torch.isfinite(log_frames).all()):
            raise RuntimeError("Learned duration prediction returned a non-finite log frame count.")
        frames = torch.expm1(log_frames)
        int64_exclusive_upper_bound = float(1 << 63)
        if not bool(torch.isfinite(frames).all()) or bool(
            (frames >= int64_exclusive_upper_bound).any()
        ):
            raise RuntimeError(
                "Learned duration prediction exceeds the representable frame-count range."
            )
        return frames.round().clamp(min=1).to(dtype=torch.long)

    def predict_duration_frames_and_token_weights(
        self,
        encoded: EncodedCFGConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict total frames and optional MAS weights from the encoded conditional rows."""
        if not getattr(self, "use_duration_predictor", False):
            raise RuntimeError(
                "Learned duration requires a versioned EchoDiT v2 duration checkpoint."
            )
        conditional = encoded.conditional
        speaker_state = conditional.speaker_state if self.duration_predictor_use_speaker else None
        speaker_mask = conditional.speaker_mask if self.duration_predictor_use_speaker else None
        if getattr(self, "use_mas_duration", False):
            log_frames, token_weights = self._predict_duration_components_from_encoded(
                conditional.text_state,
                conditional.text_mask,
                speaker_state,
                speaker_mask,
            )
        else:
            log_frames = self._predict_log_duration_from_encoded(
                conditional.text_state,
                conditional.text_mask,
                speaker_state,
                speaker_mask,
            )
            token_weights = None
        return self._duration_frames_from_log(log_frames), token_weights

    def predict_duration_frames(self, *args, **kwargs) -> torch.Tensor:
        """Predict a positive integral DACVAE frame count for each request."""
        log_frames = self.predict_log_duration(*args, **kwargs)
        return self._duration_frames_from_log(log_frames)

    def allocate_token_duration_frames(
        self,
        expected: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        total_frames: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Allocate exact positive MAS durations from already predicted token weights."""
        token_mask = (
            attention_mask.to(device=expected.device, dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(expected, dtype=torch.bool)
        )
        if total_frames is None:
            token_counts = token_mask.sum(dim=1, dtype=torch.long)
            total_frames = expected.sum(dim=1).round().to(dtype=torch.long)
            total_frames = torch.maximum(total_frames, token_counts)
        return allocate_positive_token_durations(expected, total_frames, token_mask)

    def predict_token_duration_frames(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        *,
        total_frames: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Allocate positive integer token durations that exactly match ``total_frames``."""
        expected = self.predict_expected_token_durations(
            conditioning_ids,
            attention_mask,
            speaker_latent,
            speaker_mask,
            language_ids,
        )
        return self.allocate_token_duration_frames(
            expected,
            attention_mask,
            total_frames=total_frames,
        )

    def forward_prepared(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: PreparedConditioning,
        latent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the velocity field with precomputed conditioning caches."""
        return self.dit(
            x=latents,
            t=timesteps,
            text_mask=conditioning.text_mask,
            speaker_mask=conditioning.speaker_mask,
            kv_cache_text=conditioning.kv_cache_text,
            kv_cache_speaker=conditioning.kv_cache_speaker,
            start_pos=None,
            kv_cache_latent=None,
            latent_mask=latent_mask,
            frame_text_state=conditioning.frame_text_state,
        )

    def prepare_fused_cfg_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        *,
        cfg_mode: str,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_durations: torch.Tensor | None = None,
    ) -> PreparedCFGConditioning:
        """Precompute every fixed CFG branch needed during one request."""
        encoded = self.encode_inference_conditioning(
            conditioning_ids,
            attention_mask,
            speaker_latent,
            cfg_mode=cfg_mode,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
        )
        _, prepared = self.finalize_inference_conditioning(
            encoded,
            token_durations=token_durations,
        )
        assert prepared is not None
        return prepared

    def forward_with_prepared_cfg(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditioning: PreparedCFGConditioning,
        *,
        cfg_scale: float,
        cfg_scale_text: float | None,
        cfg_scale_speaker: float | None,
        step_idx: int,
    ) -> torch.Tensor:
        """Apply CFG while reusing precomputed encoder and KV-cache outputs."""
        text_scale = cfg_scale if cfg_scale_text is None else cfg_scale_text
        speaker_scale = cfg_scale if cfg_scale_speaker is None else cfg_scale_speaker
        variant_index = step_idx % 2 if conditioning.mode == "alternating" else 0
        prediction = self.forward_prepared(
            torch.cat([latents] * conditioning.branch_count, dim=0),
            torch.cat([timesteps] * conditioning.branch_count, dim=0),
            conditioning.variants[variant_index],
        )
        branches = prediction.chunk(conditioning.branch_count, dim=0)

        if conditioning.mode == "joint":
            conditional, unconditional = branches
            return unconditional + cfg_scale * (conditional - unconditional)
        if conditioning.mode == "independent":
            conditional, unconditional_text, unconditional_speaker = branches
            return (
                conditional
                + text_scale * (conditional - unconditional_text)
                + speaker_scale * (conditional - unconditional_speaker)
            )

        conditional, unconditional = branches
        scale = text_scale if step_idx % 2 == 0 else speaker_scale
        return unconditional + scale * (conditional - unconditional)

    def forward(
        self,
        latents: torch.Tensor,  # [B, D=1024, T]
        conditioning_ids: torch.Tensor,  # [B, L] text token IDs
        timesteps: torch.Tensor,  # [B] timesteps in [0, 1]
        attention_mask: torch.Tensor | None = None,  # [B, L]
        speaker_latent: torch.Tensor | None = None,  # [B, D, T_speaker]
        use_cfg_dropout: bool = True,
        speaker_mask: torch.Tensor | None = None,  # [B, T_speaker] or [B, L_speaker]
        language_ids: torch.Tensor | None = None,  # [B] target-language IDs
        latent_mask: torch.Tensor | None = None,  # [B, T]
        return_duration_prediction: bool = False,
        return_duration_alignment: bool = False,
        duration_target_latents: torch.Tensor | None = None,
        token_durations: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | DurationAlignmentOutput]:
        """
        Forward pass for training.

        Args:
            latents: Noisy latents [B, latent_size, T]
            conditioning_ids: Text token IDs [B, L]
            timesteps: Timesteps [B] in range [0, 1]
            attention_mask: Text attention mask [B, L]
            speaker_latent: Speaker conditioning latents [B, D, T_speaker] (optional)
            use_cfg_dropout: Whether to apply CFG dropout
            speaker_mask: Valid speaker frames or patches (optional)
            language_ids: Stable target-language IDs (optional for legacy English)
            latent_mask: Valid target-latent positions for padded training batches
            return_duration_prediction: Return the v2 log-frame prediction for joint training
            return_duration_alignment: Return versioned per-token predictions and likelihoods
            duration_target_latents: Clean target latents used only by the alignment objective
            token_durations: Exact inference-time token-to-frame allocation for MAS checkpoints

        Returns:
            Predicted velocity, optionally paired with predicted log-frame count
        """
        B = latents.shape[0]
        device = latents.device
        if return_duration_alignment and not return_duration_prediction:
            raise ValueError("return_duration_alignment requires return_duration_prediction=True.")
        if return_duration_alignment and not getattr(self, "use_mas_duration", False):
            raise RuntimeError("Monotonic alignment requires a versioned MAS-duration checkpoint.")
        if return_duration_alignment and token_durations is not None:
            raise ValueError(
                "Training-time MAS alignment and inference token_durations are mutually exclusive."
            )
        if return_duration_alignment:
            if duration_target_latents is None:
                raise ValueError("MAS duration training requires clean duration_target_latents.")
            if duration_target_latents.shape != latents.shape:
                raise ValueError("duration_target_latents must have the noisy-latent shape.")

        # ``None`` means every token is valid. Keeping that sentinel avoids
        # materializing an all-True CUDA mask in every transformer block.
        if attention_mask is None:
            text_mask = None
        else:
            text_mask = attention_mask.bool()

        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=B,
            device=device,
        )

        # Duration uses the complete condition, before classifier-free dropout. Keep
        # those encoded states so the velocity path can reuse unchanged rows and only
        # re-encode the rows selected for CFG dropout.
        duration_prediction = None
        complete_text_state = None
        complete_speaker_state = None
        complete_speaker_latent = None
        complete_speaker_mask = None
        hard_alignment = None
        if return_duration_prediction:
            if not getattr(self, "use_duration_predictor", False):
                raise RuntimeError(
                    "Learned duration requires a versioned EchoDiT v2 duration checkpoint."
                )
            complete_text_state = self.dit.encode_text(
                conditioning_ids,
                text_mask,
                language_ids,
            )
            if self.duration_predictor_use_speaker:
                complete_speaker_latent, complete_speaker_mask = self._prepare_speaker_inputs(
                    speaker_latent,
                    speaker_mask,
                    batch_size=B,
                    device=device,
                )
                complete_speaker_state = self.dit.encode_speaker(
                    complete_speaker_latent,
                    complete_speaker_mask,
                )
            if return_duration_alignment:
                total_log_frames, token_durations = self._predict_duration_components_from_encoded(
                    complete_text_state,
                    text_mask,
                    complete_speaker_state,
                    complete_speaker_mask,
                )
                log_likelihoods = self.duration_alignment(
                    complete_text_state,
                    duration_target_latents,
                )
                alignment_token_mask = (
                    text_mask
                    if text_mask is not None
                    else torch.ones_like(conditioning_ids, dtype=torch.bool)
                )
                alignment_frame_mask = (
                    latent_mask
                    if latent_mask is not None
                    else torch.ones(
                        (B, latents.shape[-1]),
                        device=device,
                        dtype=torch.bool,
                    )
                )
                with torch.no_grad():
                    hard_alignment = monotonic_alignment_search(
                        log_likelihoods.detach(),
                        alignment_token_mask,
                        alignment_frame_mask,
                    )
                duration_prediction = DurationAlignmentOutput(
                    total_log_frames=total_log_frames,
                    token_durations=token_durations,
                    log_likelihoods=log_likelihoods,
                    hard_alignment=hard_alignment,
                )
            else:
                duration_prediction = self._predict_log_duration_from_encoded(
                    complete_text_state,
                    text_mask,
                    complete_speaker_state,
                    complete_speaker_mask,
                )

        # Independent classifier-free conditioning dropout per training row.
        # Text and speaker are dropped independently at separate rates
        text_cfg_mask = None
        speaker_cfg_mask = None
        text_condition_changed = False
        speaker_condition_changed = False
        if self.training and use_cfg_dropout:
            # Text dropout (independent)
            text_cfg_mask = torch.rand(B, device=device) < self.cfg_dropout_text
            if text_cfg_mask.any():
                text_condition_changed = True
                conditioning_ids = conditioning_ids.clone()
                conditioning_ids[text_cfg_mask] = 0  # Pad token
                if language_ids is not None:
                    language_ids = language_ids.clone()
                    language_ids[text_cfg_mask] = NULL_LANGUAGE_ID

            # Speaker dropout (independent, separate random draw)
            if self.use_speaker_conditioning and speaker_latent is not None:
                speaker_cfg_mask = torch.rand(B, device=device) < self.cfg_dropout_speaker
                if speaker_cfg_mask.any():
                    speaker_condition_changed = True
                    # Replace dropped speakers with null embedding
                    speaker_latent = speaker_latent.clone()
                    null_speaker = self.null_speaker_embed.expand(speaker_cfg_mask.sum(), -1, -1)
                    # Pad null_speaker to match speaker_latent time dimension
                    if null_speaker.shape[2] < speaker_latent.shape[2]:
                        null_speaker = torch.nn.functional.pad(
                            null_speaker,
                            (0, speaker_latent.shape[2] - null_speaker.shape[2]),
                            value=0.0,
                        )
                    speaker_latent[speaker_cfg_mask] = null_speaker[:, :, : speaker_latent.shape[2]]

        if complete_text_state is None:
            # Preserve the public preparation path for legacy subclasses and inference.
            conditioning = self.prepare_inference_conditioning(
                conditioning_ids,
                text_mask,
                speaker_latent,
                speaker_mask,
                language_ids,
                token_durations,
            )
        else:
            text_state = complete_text_state
            if text_condition_changed:
                assert text_cfg_mask is not None
                changed_rows = text_cfg_mask.nonzero(as_tuple=False).squeeze(1)
                changed_text_state = self.dit.encode_text(
                    conditioning_ids.index_select(0, changed_rows),
                    text_mask.index_select(0, changed_rows) if text_mask is not None else None,
                    (
                        language_ids.index_select(0, changed_rows)
                        if language_ids is not None
                        else None
                    ),
                )
                text_state = text_state.index_copy(0, changed_rows, changed_text_state)

            if complete_speaker_state is None:
                velocity_speaker_latent, velocity_speaker_mask = self._prepare_speaker_inputs(
                    speaker_latent,
                    speaker_mask,
                    batch_size=B,
                    device=device,
                )
                speaker_state = self.dit.encode_speaker(
                    velocity_speaker_latent,
                    velocity_speaker_mask,
                )
            else:
                velocity_speaker_mask = complete_speaker_mask
                speaker_state = complete_speaker_state
                if speaker_condition_changed:
                    assert speaker_cfg_mask is not None
                    changed_rows = speaker_cfg_mask.nonzero(as_tuple=False).squeeze(1)
                    changed_speaker_latent, changed_speaker_mask = self._prepare_speaker_inputs(
                        speaker_latent.index_select(0, changed_rows),
                        (
                            speaker_mask.index_select(0, changed_rows)
                            if speaker_mask is not None
                            else None
                        ),
                        batch_size=changed_rows.shape[0],
                        device=device,
                    )
                    changed_speaker_state = self.dit.encode_speaker(
                        changed_speaker_latent,
                        changed_speaker_mask,
                    )
                    speaker_state = speaker_state.index_copy(
                        0,
                        changed_rows,
                        changed_speaker_state,
                    )
            conditioning = PreparedConditioning(
                text_mask=text_mask,
                speaker_mask=velocity_speaker_mask,
                kv_cache_text=self.dit.project_text_kv_cache(text_state),
                kv_cache_speaker=self.dit.project_speaker_kv_cache(speaker_state),
                frame_text_state=(
                    torch.bmm(
                        hard_alignment.transpose(1, 2).to(dtype=text_state.dtype),
                        text_state,
                    )
                    if hard_alignment is not None
                    else (
                        expand_text_by_durations(
                            text_state,
                            token_durations,
                            target_frames=latents.shape[-1],
                        )
                        if getattr(self, "use_mas_duration", False) and token_durations is not None
                        else None
                    )
                ),
            )
        velocity = self.forward_prepared(latents, timesteps, conditioning, latent_mask)
        if duration_prediction is not None:
            return velocity, duration_prediction
        return velocity

    def forward_with_cfg(
        self,
        latents: torch.Tensor,
        conditioning_ids: torch.Tensor,
        timesteps: torch.Tensor,
        cfg_scale: float = 1.0,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        cfg_mode: str = "joint",
        cfg_scale_text: float | None = None,
        cfg_scale_speaker: float | None = None,
        step_idx: int | None = None,
        speaker_mask: torch.Tensor | None = None,
        fuse_cfg_branches: bool = False,
        language_ids: torch.Tensor | None = None,
        token_durations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance (inference).

        Supports three CFG modes from Echo paper:
        - joint: Two conditional/unconditional branches
        - independent: Three text/speaker guidance branches
        - alternating: Two branches whose guidance target alternates by step

        Branches run sequentially by default and can be fused for compiled inference.

        Args:
            latents: Noisy latents [B, D, T]
            conditioning_ids: Text token IDs [B, L]
            timesteps: Timesteps [B]
            cfg_scale: Joint guidance scale (used for joint mode)
            attention_mask: Text attention mask [B, L]
            speaker_latent: Speaker conditioning [B, D, T_speaker]
            cfg_mode: "joint", "independent", or "alternating"
            cfg_scale_text: Text guidance scale (for independent/alternating)
            cfg_scale_speaker: Speaker guidance scale (for independent/alternating)
            step_idx: Current step index (for alternating mode)
            speaker_mask: Valid speaker frames or patches (optional)
            fuse_cfg_branches: Batch CFG branches into one backbone call
            language_ids: Stable target-language IDs
            token_durations: Exact token-to-frame allocation for MAS checkpoints

        Returns:
            Guided velocity prediction [B, D, T]
        """
        if cfg_scale == 1.0 and cfg_mode == "joint":
            forward_kwargs = {}
            if token_durations is not None:
                forward_kwargs["token_durations"] = token_durations
            return self.forward(
                latents,
                conditioning_ids,
                timesteps,
                attention_mask,
                speaker_latent,
                use_cfg_dropout=False,
                speaker_mask=speaker_mask,
                language_ids=language_ids,
                **forward_kwargs,
            )

        B = latents.shape[0]
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=B,
            device=conditioning_ids.device,
        )
        # Default scales for independent/alternating
        if cfg_scale_text is None:
            cfg_scale_text = cfg_scale
        if cfg_scale_speaker is None:
            cfg_scale_speaker = cfg_scale

        def null_speaker_like(reference: torch.Tensor | None) -> torch.Tensor | None:
            if not self.use_speaker_conditioning or reference is None:
                return None
            null_speaker = self.null_speaker_embed.expand(B, -1, -1)
            reference_frames = reference.shape[-1]
            if null_speaker.shape[-1] < reference_frames:
                null_speaker = torch.nn.functional.pad(
                    null_speaker,
                    (0, reference_frames - null_speaker.shape[-1]),
                )
            return null_speaker[..., :reference_frames]

        def batched_predictions(
            text_branches: list[torch.Tensor],
            speaker_branches: list[torch.Tensor | None],
            language_branches: list[torch.Tensor | None],
        ) -> tuple[torch.Tensor, ...]:
            """Evaluate CFG branches sequentially or as one fixed batch.

            Compiled and Cache-DiT inference use one fixed transformer batch per
            solver evaluation. Other profiles keep the lower-memory sequential path.
            """
            branch_count = len(text_branches)
            if not branch_count == len(speaker_branches) == len(language_branches):
                raise ValueError("Text, speaker, and language CFG branches must align.")

            def forward_branch(
                text_branch: torch.Tensor,
                speaker_branch: torch.Tensor | None,
                language_branch: torch.Tensor | None,
            ) -> torch.Tensor:
                branch_kwargs = {}
                if language_branch is not None:
                    branch_kwargs["language_ids"] = language_branch
                if token_durations is not None:
                    branch_kwargs["token_durations"] = token_durations
                return self.forward(
                    latents,
                    text_branch,
                    timesteps,
                    attention_mask,
                    speaker_branch,
                    use_cfg_dropout=False,
                    speaker_mask=speaker_mask,
                    **branch_kwargs,
                )

            if not fuse_cfg_branches:
                return tuple(
                    forward_branch(text_branch, speaker_branch, language_branch)
                    for text_branch, speaker_branch, language_branch in zip(
                        text_branches,
                        speaker_branches,
                        language_branches,
                    )
                )

            batched_speakers = None
            if any(branch is not None for branch in speaker_branches):
                if not all(branch is not None for branch in speaker_branches):
                    raise ValueError(
                        "Speaker CFG branches must either all be tensors or all be None."
                    )
                batched_speakers = torch.cat(speaker_branches, dim=0)  # type: ignore[arg-type]

            batched_kwargs = {}
            if all(branch is not None for branch in language_branches):
                batched_kwargs["language_ids"] = torch.cat(language_branches, dim=0)
            if token_durations is not None:
                batched_kwargs["token_durations"] = torch.cat(
                    [token_durations] * branch_count,
                    dim=0,
                )
            prediction = self.forward(
                torch.cat([latents] * branch_count, dim=0),
                torch.cat(text_branches, dim=0),
                torch.cat([timesteps] * branch_count, dim=0),
                (
                    torch.cat([attention_mask] * branch_count, dim=0)
                    if attention_mask is not None
                    else None
                ),
                batched_speakers,
                use_cfg_dropout=False,
                speaker_mask=(
                    torch.cat([speaker_mask] * branch_count, dim=0)
                    if speaker_mask is not None
                    else None
                ),
                **batched_kwargs,
            )
            return prediction.chunk(branch_count, dim=0)

        null_language_ids = torch.zeros_like(language_ids) if language_ids is not None else None

        if cfg_mode == "joint":
            # Joint unconditional: drop both text and speaker
            null_ids = torch.zeros_like(conditioning_ids)
            null_speaker = null_speaker_like(speaker_latent)
            v_cond, v_uncond = batched_predictions(
                [conditioning_ids, null_ids],
                [speaker_latent, null_speaker],
                [language_ids, null_language_ids],
            )

            # CFG: v = v_uncond + scale * (v_cond - v_uncond)
            v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)

        elif cfg_mode == "independent":
            # Independent guidance: separate scales for text and speaker.
            # v = v_cond + w_text*(v_cond - v_uncond_text) + w_speaker*(v_cond - v_uncond_speaker)

            # Unconditional text (keep speaker)
            null_ids = torch.zeros_like(conditioning_ids)
            null_speaker = null_speaker_like(speaker_latent)
            v_cond, v_uncond_text, v_uncond_speaker = batched_predictions(
                [conditioning_ids, null_ids, conditioning_ids],
                [speaker_latent, speaker_latent, null_speaker],
                [language_ids, null_language_ids, language_ids],
            )

            # Independent CFG formula
            v_guided = (
                v_cond
                + cfg_scale_text * (v_cond - v_uncond_text)
                + cfg_scale_speaker * (v_cond - v_uncond_speaker)
            )

        elif cfg_mode == "alternating":
            # Alternating guidance: alternate between text and speaker each step (2x NFE)
            if step_idx is None:
                step_idx = 0

            if step_idx % 2 == 0:
                # Text guidance step
                null_ids = torch.zeros_like(conditioning_ids)
                v_cond, v_uncond = batched_predictions(
                    [conditioning_ids, null_ids],
                    [speaker_latent, speaker_latent],
                    [language_ids, null_language_ids],
                )
                v_guided = v_uncond + cfg_scale_text * (v_cond - v_uncond)
            else:
                # Speaker guidance step
                null_speaker = null_speaker_like(speaker_latent)
                v_cond, v_uncond = batched_predictions(
                    [conditioning_ids, conditioning_ids],
                    [speaker_latent, null_speaker],
                    [language_ids, language_ids],
                )
                v_guided = v_uncond + cfg_scale_speaker * (v_cond - v_uncond)

        else:
            raise ValueError(
                f"Unknown CFG mode: {cfg_mode}. Use 'joint', 'independent', or 'alternating'"
            )

        return v_guided

    @torch.no_grad()
    def encode_text(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode text and return KV cache."""
        if attention_mask is None:
            attention_mask = torch.ones_like(conditioning_ids, dtype=torch.bool)
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=conditioning_ids.shape[0],
            device=conditioning_ids.device,
        )
        if language_ids is None:
            return self.dit.get_kv_cache_text(conditioning_ids, attention_mask)
        return self.dit.get_kv_cache_text(conditioning_ids, attention_mask, language_ids)

    def get_num_params(self):
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        dit = sum(p.numel() for p in self.dit.parameters())
        text_encoder = sum(p.numel() for p in self.dit.text_encoder.parameters())
        speaker_encoder = (
            sum(p.numel() for p in self.dit.speaker_encoder.parameters())
            if self.dit.speaker_encoder is not None
            else 0
        )
        duration_predictor = (
            sum(p.numel() for p in self.duration_predictor.parameters())
            if self.use_duration_predictor
            else 0
        )
        duration_alignment = (
            sum(p.numel() for p in self.duration_alignment.parameters())
            if self.use_mas_duration
            else 0
        )

        return {
            "total": total,
            "dit": dit,
            "text_encoder": text_encoder,
            "speaker_encoder": speaker_encoder,
            "duration_predictor": duration_predictor,
            "duration_alignment": duration_alignment,
            "dit_blocks": dit - text_encoder - speaker_encoder,
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the model.
        This is required by HuggingFace Trainer.
        """
        if hasattr(self.dit, "gradient_checkpointing_enable"):
            self.dit.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            # Manual gradient checkpointing for EchoDiT
            if hasattr(self.dit, "gradient_checkpointing"):
                self.dit.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the model.
        This is required by HuggingFace Trainer.
        """
        if hasattr(self.dit, "gradient_checkpointing_disable"):
            self.dit.gradient_checkpointing_disable()
        else:
            # Manual gradient checkpointing for EchoDiT
            if hasattr(self.dit, "gradient_checkpointing"):
                self.dit.gradient_checkpointing = False


def create_flow_matching_echodit(
    latent_size: int = 1024,
    model_size: int = 1024,
    num_layers: int = 24,
    num_heads: int = 16,
    intermediate_size: int = 4096,
    text_vocab_size: int = 152000,
    cfg_dropout: float = 0.1,
    cfg_dropout_text: float | None = None,
    cfg_dropout_speaker: float | None = None,
    use_speaker_conditioning: bool = False,
    use_language_conditioning: bool = False,
    supported_languages: tuple[str, ...] | list[str] | None = None,
    supported_reference_languages: tuple[str, ...] | list[str] | None = None,
    supported_language_pairs: (
        Iterable[LanguagePair | Sequence[str | Language]] | LanguagePair | None
    ) = None,
    use_duration_predictor: bool = False,
    duration_predictor_hidden_size: int = 256,
    duration_predictor_num_layers: int = 2,
    duration_predictor_use_speaker: bool = False,
    use_mas_duration: bool = False,
    duration_alignment_hidden_size: int = 64,
    **kwargs,
) -> FlowMatchingEchoDiT:
    """
    Factory function to create FlowMatchingEchoDiT model.

    Args:
        latent_size: DACVAE latent dimension (1024)
        model_size: DiT hidden dimension
        num_layers: Number of DiT layers
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        text_vocab_size: Text vocabulary size
        cfg_dropout: Default classifier-free conditioning dropout rate
        cfg_dropout_text: Optional text-specific override of ``cfg_dropout``
        cfg_dropout_speaker: Optional speaker-specific override of ``cfg_dropout``
        use_speaker_conditioning: Enable speaker conditioning
        use_language_conditioning: Enable a learned target-language embedding
        supported_languages: Canonical codes or aliases represented by training data
        supported_reference_languages: Languages represented by speaker-reference audio
        supported_language_pairs: Exact target/reference pairs represented by training data
        use_duration_predictor: Add a versioned learned DACVAE frame-count head
        use_mas_duration: Add the independently versioned monotonic-alignment head
        duration_alignment_hidden_size: Fixed acoustic-projection and text-prior width
        **kwargs: Additional arguments

    Returns:
        FlowMatchingEchoDiT model
    """
    model = FlowMatchingEchoDiT(
        latent_size=latent_size,
        model_size=model_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        text_vocab_size=text_vocab_size,
        cfg_dropout=cfg_dropout,
        cfg_dropout_text=cfg_dropout_text,
        cfg_dropout_speaker=cfg_dropout_speaker,
        use_speaker_conditioning=use_speaker_conditioning,
        use_language_conditioning=use_language_conditioning,
        supported_languages=supported_languages,
        supported_reference_languages=supported_reference_languages,
        supported_language_pairs=supported_language_pairs,
        use_duration_predictor=use_duration_predictor,
        duration_predictor_hidden_size=duration_predictor_hidden_size,
        duration_predictor_num_layers=duration_predictor_num_layers,
        duration_predictor_use_speaker=duration_predictor_use_speaker,
        use_mas_duration=use_mas_duration,
        duration_alignment_hidden_size=duration_alignment_hidden_size,
        **kwargs,
    )

    return model
