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
from nar_vae.objectives import (
    DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY,
    GENERATIVE_OBJECTIVE_METADATA_KEY,
    RECTIFIED_FLOW_OBJECTIVE,
    VP_DIFFUSION_OBJECTIVE,
    normalize_generative_objective,
    objective_metadata_code,
    validate_diffusion_schedule_shift,
)
from nar_vae.tokenization import TOTAL_VOCAB_SIZE
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
    LEGACY_ECHODIT_ARCHITECTURE_VERSION,
    MONOTONIC_ALIGNMENT_VERSION,
    DurationAlignmentOutput,
    EchoDurationAlignment,
    EchoDurationPredictor,
    allocate_positive_token_durations,
    expand_text_by_durations,
)
from .text_conditioning import (
    FROZEN_FEATURE_TEXT_CONDITIONING,
    SCRATCH_TOKEN_TEXT_CONDITIONING,
    TEXT_CONDITIONING_ADAPTER_VERSION_KEY,
    TEXT_CONDITIONING_FEATURE_SIZE_KEY,
    TEXT_CONDITIONING_MODE_KEY,
    TEXT_CONDITIONING_VERSION,
    TEXT_CONDITIONING_VERSION_KEY,
    resolve_text_conditioning_metadata,
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
    global_language_state: torch.Tensor | None = None

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
            global_language_state=(
                self.global_language_state[start:stop]
                if self.global_language_state is not None
                else None
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
    alignment_mask: torch.Tensor | None = None
    global_language_state: torch.Tensor | None = None
    text_is_null: torch.Tensor | None = None

    def slice_batch(self, start: int, stop: int) -> "EncodedConditioning":
        """Return a contiguous batch slice without running either encoder again."""
        return EncodedConditioning(
            text_mask=(self.text_mask[start:stop] if self.text_mask is not None else None),
            alignment_mask=(
                self.alignment_mask[start:stop] if self.alignment_mask is not None else None
            ),
            speaker_mask=(self.speaker_mask[start:stop] if self.speaker_mask is not None else None),
            text_state=self.text_state[start:stop],
            speaker_state=self.speaker_state[start:stop],
            global_language_state=(
                self.global_language_state[start:stop]
                if self.global_language_state is not None
                else None
            ),
            text_is_null=(self.text_is_null[start:stop] if self.text_is_null is not None else None),
        )

    def index_select_batch(self, indices: torch.Tensor) -> "EncodedConditioning":
        """Select arbitrary request rows while retaining device-resident encoder states."""
        indices = indices.to(device=self.text_state.device, dtype=torch.long)
        return EncodedConditioning(
            text_mask=(
                self.text_mask.index_select(0, indices) if self.text_mask is not None else None
            ),
            alignment_mask=(
                self.alignment_mask.index_select(0, indices)
                if self.alignment_mask is not None
                else None
            ),
            speaker_mask=(
                self.speaker_mask.index_select(0, indices)
                if self.speaker_mask is not None
                else None
            ),
            text_state=self.text_state.index_select(0, indices),
            speaker_state=self.speaker_state.index_select(0, indices),
            global_language_state=(
                self.global_language_state.index_select(0, indices)
                if self.global_language_state is not None
                else None
            ),
            text_is_null=(
                self.text_is_null.index_select(0, indices)
                if self.text_is_null is not None
                else None
            ),
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
    Versioned continuous-velocity TTS with the EchoDiT architecture.

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
        text_vocab_size: int = TOTAL_VOCAB_SIZE,
        text_model_size: int = 768,
        text_num_layers: int = 6,
        text_num_heads: int = 12,
        text_intermediate_size: int = 3072,
        text_conditioning_mode: str = SCRATCH_TOKEN_TEXT_CONDITIONING,
        conditioning_feature_size: int | None = None,
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
        target_patch_size: int = 1,
        generative_objective: str = RECTIFIED_FLOW_OBJECTIVE,
        diffusion_schedule_shift: float = DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
        speaker_num_summary_tokens: int = 0,
        architecture_version: int = ECHODIT_ARCHITECTURE_VERSION,
    ):
        super().__init__()

        if architecture_version not in {
            LEGACY_ECHODIT_ARCHITECTURE_VERSION,
            ECHODIT_ARCHITECTURE_VERSION,
        }:
            raise ValueError(f"Unsupported EchoDiT architecture version: {architecture_version}.")
        if architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION and (
            text_conditioning_mode != SCRATCH_TOKEN_TEXT_CONDITIONING
            or target_patch_size != 1
            or speaker_num_summary_tokens != 0
            or generative_objective != RECTIFIED_FLOW_OBJECTIVE
            or diffusion_schedule_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT
        ):
            raise ValueError(
                "EchoDiT architecture v3 is an inference-only rectified-flow scratch-token "
                "topology with target_patch_size=1 and no speaker resampler."
            )

        self.latent_size = latent_size
        self.architecture_version = architecture_version
        self.text_conditioning = resolve_text_conditioning_metadata(
            text_conditioning_mode,
            conditioning_feature_size,
        )
        self.text_conditioning_mode = self.text_conditioning.mode
        self.conditioning_feature_size = self.text_conditioning.feature_size or None
        # Metadata is additive only for the new topology.  Its absence retains
        # the exact legacy meaning and state-dict schema: scratch token encoder.
        if self.text_conditioning.mode == FROZEN_FEATURE_TEXT_CONDITIONING:
            self.register_buffer(
                TEXT_CONDITIONING_VERSION_KEY,
                torch.tensor(TEXT_CONDITIONING_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                TEXT_CONDITIONING_MODE_KEY,
                torch.tensor(self.text_conditioning.mode_code, dtype=torch.int32),
            )
            self.register_buffer(
                TEXT_CONDITIONING_FEATURE_SIZE_KEY,
                torch.tensor(self.text_conditioning.feature_size, dtype=torch.int32),
            )
            self.register_buffer(
                TEXT_CONDITIONING_ADAPTER_VERSION_KEY,
                torch.tensor(self.text_conditioning.adapter_version, dtype=torch.int32),
            )
        self.generative_objective = normalize_generative_objective(generative_objective)
        self.diffusion_schedule_shift = validate_diffusion_schedule_shift(diffusion_schedule_shift)
        if (
            self.generative_objective == RECTIFIED_FLOW_OBJECTIVE
            and self.diffusion_schedule_shift != DEFAULT_DIFFUSION_SCHEDULE_SHIFT
        ):
            raise ValueError(
                "diffusion_schedule_shift is only meaningful for vp_diffusion_v checkpoints."
            )
        objective_code = objective_metadata_code(self.generative_objective)
        if objective_code is not None:
            self.register_buffer(
                GENERATIVE_OBJECTIVE_METADATA_KEY,
                torch.tensor(objective_code, dtype=torch.int32),
            )
        if self.generative_objective == VP_DIFFUSION_OBJECTIVE:
            self.register_buffer(
                DIFFUSION_SCHEDULE_SHIFT_METADATA_KEY,
                torch.tensor(self.diffusion_schedule_shift, dtype=torch.float64),
            )
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
            text_conditioning_mode=self.text_conditioning.mode,
            conditioning_feature_size=self.conditioning_feature_size,
            speaker_patch_size=speaker_patch_size,
            speaker_model_size=speaker_model_size,
            speaker_num_layers=speaker_num_layers,
            speaker_num_heads=speaker_num_heads,
            speaker_intermediate_size=speaker_intermediate_size,
            speaker_num_summary_tokens=speaker_num_summary_tokens,
            target_patch_size=target_patch_size,
            timestep_embed_size=timestep_embed_size,
            adaln_rank=adaln_rank,
            num_languages=LANGUAGE_COUNT if use_language_conditioning else 0,
            use_speaker_conditioning=use_speaker_conditioning,
            use_duration_alignment=use_mas_duration,
            architecture_version=architecture_version,
        )
        self.speaker_num_summary_tokens = self.dit.speaker_num_summary_tokens
        if architecture_version >= ECHODIT_ARCHITECTURE_VERSION:
            self.register_buffer(
                "echodit_architecture_version",
                torch.tensor(ECHODIT_ARCHITECTURE_VERSION, dtype=torch.int32),
            )
            self.register_buffer(
                "target_patch_size_metadata",
                torch.tensor(self.dit.target_patch_size, dtype=torch.int32),
            )
        if use_duration_predictor:
            self.duration_predictor = EchoDurationPredictor(
                text_size=text_model_size,
                hidden_size=duration_predictor_hidden_size,
                num_layers=duration_predictor_num_layers,
                speaker_size=(speaker_model_size if duration_predictor_use_speaker else None),
            )
            if architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
                self.register_buffer(
                    "echodit_architecture_version",
                    torch.tensor(architecture_version, dtype=torch.int32),
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

        if architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            self.register_buffer("null_text_embed", torch.zeros(1, 1, text_model_size))
        else:
            # Learned encoder-level null states make classifier-free branches explicit.
            # Reusing the historical null_text_embed key keeps old state dictionaries
            # loadable even though the tensor is now trainable and actually consumed.
            self.null_text_embed = nn.Parameter(torch.zeros(1, 1, text_model_size))

        if use_speaker_conditioning:
            # Retain the historical raw-latent sentinel as checkpoint metadata, but
            # CFG now substitutes a learned encoded state and skips a fake codec pass.
            self.register_buffer(
                "null_speaker_embed", torch.zeros(1, latent_size, speaker_patch_size)
            )
            if architecture_version >= ECHODIT_ARCHITECTURE_VERSION:
                self.null_speaker_state = nn.Parameter(torch.zeros(1, 1, speaker_model_size))
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
            if self.speaker_num_summary_tokens > 0:
                # Additive topology metadata: its absence means the exact legacy
                # variable-length speaker state and state-dict schema.
                self.register_buffer(
                    "speaker_resampler_version",
                    torch.tensor(1, dtype=torch.int32),
                )
                self.register_buffer(
                    "speaker_num_summary_tokens_metadata",
                    torch.tensor(self.speaker_num_summary_tokens, dtype=torch.int32),
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
        """Validate explicit utterance-level target language IDs."""
        uses_language_conditioning = getattr(self, "use_language_conditioning", False)
        if language_ids is None:
            if uses_language_conditioning:
                if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
                    return torch.full(
                        (batch_size,),
                        language_id(DEFAULT_LANGUAGE),
                        dtype=torch.long,
                        device=device,
                    )
                raise ValueError(
                    "language_ids are required by a language-conditioned NAR-VAE checkpoint."
                )
            return None
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

    def _prepare_token_language_ids(
        self,
        token_language_ids: torch.Tensor | None,
        *,
        conditioning_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Validate optional language IDs for individual text/span tokens."""
        if token_language_ids is None:
            return None
        if tuple(token_language_ids.shape) != tuple(conditioning_ids.shape):
            raise ValueError("token_language_ids must have shape [batch, token].")
        token_language_ids = token_language_ids.to(
            device=conditioning_ids.device,
            dtype=torch.long,
        )
        if not self.use_language_conditioning:
            default_id = language_id(DEFAULT_LANGUAGE)
            if torch.any(
                (token_language_ids != default_id) & (token_language_ids != NULL_LANGUAGE_ID)
            ):
                raise RuntimeError(
                    "Non-English token language IDs require a language-conditioned checkpoint."
                )
            return None
        supported = self.supported_language_ids_metadata.to(
            device=conditioning_ids.device,
            dtype=torch.long,
        )
        valid = (token_language_ids[..., None] == supported).any(dim=-1)
        valid = valid | (token_language_ids == NULL_LANGUAGE_ID)
        if not bool(valid.all()):
            invalid = sorted(set(token_language_ids[~valid].detach().cpu().tolist()))
            raise ValueError(
                f"token_language_ids contain values not supported by this checkpoint: {invalid}."
            )
        return token_language_ids

    @staticmethod
    def _prepare_alignment_mask(
        alignment_mask: torch.Tensor | None,
        *,
        conditioning_ids: torch.Tensor,
        text_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate a possibly non-contiguous subset of tokens allowed to own frames."""
        if alignment_mask is None:
            return None
        if tuple(alignment_mask.shape) != tuple(conditioning_ids.shape):
            raise ValueError("alignment_mask must have shape [batch, token].")
        alignment_mask = alignment_mask.to(device=conditioning_ids.device, dtype=torch.bool)
        if text_mask is not None and bool((alignment_mask & ~text_mask).any()):
            raise ValueError("alignment_mask cannot select padded text tokens.")
        if not bool(alignment_mask.any(dim=1).all()):
            raise ValueError("Every alignment_mask row must select at least one token.")
        return alignment_mask

    def _validate_legacy_v3_parallel_text_axes(
        self,
        *,
        conditioning_ids: torch.Tensor,
        text_mask: torch.Tensor | None,
        language_ids: torch.Tensor | None,
        token_language_ids: torch.Tensor | None,
        alignment_mask: torch.Tensor | None,
    ) -> None:
        """Authenticate modern compatibility axes as exact v3 semantic no-ops."""
        if token_language_ids is not None:
            prepared_token_languages = self._prepare_token_language_ids(
                token_language_ids,
                conditioning_ids=conditioning_ids,
            )
            if prepared_token_languages is not None and language_ids is not None:
                expected_languages = language_ids[:, None].expand_as(prepared_token_languages)
                valid_text = (
                    text_mask
                    if text_mask is not None
                    else torch.ones_like(conditioning_ids, dtype=torch.bool)
                )
                if bool(((prepared_token_languages != expected_languages) & valid_text).any()):
                    raise ValueError(
                        "EchoDiT architecture v3 token_language_ids must broadcast the "
                        "utterance-level language across every valid text token."
                    )
        if alignment_mask is not None:
            prepared_alignment = self._prepare_alignment_mask(
                alignment_mask,
                conditioning_ids=conditioning_ids,
                text_mask=text_mask,
            )
            expected_alignment = (
                text_mask
                if text_mask is not None
                else torch.ones_like(conditioning_ids, dtype=torch.bool)
            )
            if not torch.equal(prepared_alignment, expected_alignment):
                raise ValueError(
                    "EchoDiT architecture v3 alignment_mask must select every valid text token."
                )

    @staticmethod
    def _materialize_mask(
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones((batch_size, length), dtype=torch.bool, device=device)
        return mask.to(device=device, dtype=torch.bool)

    def _null_text_conditioning(
        self,
        *,
        batch_size: int,
        token_count: int,
        state_size: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one learned valid null token in a fixed-length padded state."""
        null_text_embed = self.null_text_embed
        if state_size is not None and state_size != null_text_embed.shape[-1]:
            raise RuntimeError("The learned null text state width does not match the encoder.")
        state_size = int(null_text_embed.shape[-1])
        state = torch.zeros(
            (batch_size, token_count, state_size),
            device=device,
            dtype=dtype,
        )
        state[:, :1] = null_text_embed.to(device=device, dtype=dtype)
        mask = torch.zeros((batch_size, token_count), device=device, dtype=torch.bool)
        mask[:, 0] = True
        return state, mask

    def _null_speaker_conditioning(
        self,
        *,
        batch_size: int,
        state_count: int,
        state_size: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return one learned timbre-null token without encoding zero codec latents."""
        if not self.use_speaker_conditioning:
            speaker_size = int(
                getattr(self.dit, "speaker_model_size", getattr(self, "latent_size", 0))
            )
            return (
                torch.empty(
                    (batch_size, 0, speaker_size),
                    device=device,
                    dtype=dtype,
                ),
                None,
            )
        null_speaker_state = self.null_speaker_state
        if state_size is not None and state_size != null_speaker_state.shape[-1]:
            raise RuntimeError("The learned null speaker state width does not match the encoder.")
        state_count = max(1, state_count)
        state = torch.zeros(
            (batch_size, state_count, null_speaker_state.shape[-1]),
            device=device,
            dtype=dtype,
        )
        state[:, :1] = null_speaker_state.to(device=device, dtype=dtype)
        if state_count == 1:
            return state, None
        mask = torch.zeros((batch_size, state_count), device=device, dtype=torch.bool)
        mask[:, 0] = True
        return state, mask

    @staticmethod
    def _compact_text_state(
        text_state: torch.Tensor,
        alignment_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack an arbitrary frame-owning token subset into a prefix-masked tensor."""
        if tuple(alignment_mask.shape) != tuple(text_state.shape[:2]):
            raise ValueError("alignment_mask must have the encoded text shape.")
        alignment_mask = alignment_mask.to(device=text_state.device, dtype=torch.bool)
        token_counts = alignment_mask.sum(dim=1)
        # Stable ranks map selected source tokens onto a contiguous prefix. Retain
        # the static padded length so compilation/DDP never incurs a GPU-to-CPU
        # shape synchronization for a data-dependent max token count.
        compact_indices = alignment_mask.cumsum(dim=1).sub(1).clamp_min(0)
        compact = text_state.new_zeros(text_state.shape)
        compact.scatter_add_(
            1,
            compact_indices[..., None].expand_as(text_state),
            text_state * alignment_mask[..., None].to(dtype=text_state.dtype),
        )
        compact_mask = (
            torch.arange(text_state.shape[1], device=text_state.device)[None, :]
            < token_counts[:, None]
        )
        return compact, compact_mask

    @staticmethod
    def _scatter_compact_rows(
        compact: torch.Tensor,
        alignment_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter compact token-leading tensors back to the public full-token shape."""
        if compact.shape[:2] != alignment_mask.shape:
            raise ValueError("Compact state and alignment_mask must have the same padded shape.")
        alignment_mask = alignment_mask.to(device=compact.device, dtype=torch.bool)
        compact_indices = alignment_mask.cumsum(dim=1).sub(1).clamp_min(0)
        trailing_dims = (1,) * (compact.ndim - 2)
        gather_indices = compact_indices.reshape(
            alignment_mask.shape[0],
            alignment_mask.shape[1],
            *trailing_dims,
        ).expand_as(compact)
        full = compact.gather(1, gather_indices)
        return full * alignment_mask.reshape(
            alignment_mask.shape[0],
            alignment_mask.shape[1],
            *trailing_dims,
        ).to(dtype=compact.dtype)

    def prepare_inference_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_durations: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> PreparedConditioning:
        """Encode invariant text/speaker context once for an ODE trajectory."""
        encoded = self.encode_inference_conditioning(
            conditioning_ids,
            attention_mask,
            speaker_latent,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            conditioning_features=conditioning_features,
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
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> EncodedCFGConditioning:
        """Encode the real conditions once, then assemble learned-null CFG branches."""
        if cfg_mode not in {None, "joint", "independent", "alternating"}:
            raise ValueError(f"Unknown CFG mode: {cfg_mode}")
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            return self._encode_legacy_v3_inference_conditioning(
                conditioning_ids,
                attention_mask,
                speaker_latent,
                cfg_mode=cfg_mode,
                speaker_mask=speaker_mask,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
                conditioning_features=conditioning_features,
            )
        batch_size = conditioning_ids.shape[0]
        device = conditioning_ids.device
        text_mask = attention_mask.bool() if attention_mask is not None else None
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=batch_size,
            device=device,
        )
        token_language_ids = self._prepare_token_language_ids(
            token_language_ids,
            conditioning_ids=conditioning_ids,
        )
        alignment_mask = self._prepare_alignment_mask(
            alignment_mask,
            conditioning_ids=conditioning_ids,
            text_mask=text_mask,
        )

        encode_text_kwargs = {}
        if token_language_ids is not None:
            encode_text_kwargs["token_language_ids"] = token_language_ids
        if conditioning_features is not None:
            encode_text_kwargs["conditioning_features"] = conditioning_features
        text_state = self.dit.encode_text(
            conditioning_ids,
            text_mask,
            language_ids,
            **encode_text_kwargs,
        )
        encode_global_language = getattr(self.dit, "encode_global_language", None)
        global_language_state = (
            encode_global_language(language_ids, batch_size=batch_size, device=device)
            if callable(encode_global_language)
            else None
        )
        speaker_state, speaker_attention_mask = self._encode_speaker_conditioning(
            speaker_latent,
            speaker_mask,
            batch_size=batch_size,
            device=device,
        )

        conditional = EncodedConditioning(
            text_mask=text_mask,
            alignment_mask=alignment_mask,
            speaker_mask=speaker_attention_mask,
            text_state=text_state,
            speaker_state=speaker_state,
            global_language_state=global_language_state,
            text_is_null=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )
        if cfg_mode is None:
            return EncodedCFGConditioning(
                mode=None,
                branch_count=1,
                variants=(conditional,),
                conditional=conditional,
            )

        null_text_state, null_text_mask = self._null_text_conditioning(
            batch_size=batch_size,
            token_count=text_state.shape[1],
            state_size=text_state.shape[2],
            device=text_state.device,
            dtype=text_state.dtype,
        )
        null_speaker_state, null_speaker_mask = self._null_speaker_conditioning(
            batch_size=batch_size,
            state_count=speaker_state.shape[1],
            state_size=speaker_state.shape[2],
            device=speaker_state.device,
            dtype=speaker_state.dtype,
        )
        null_global_state = (
            torch.zeros_like(global_language_state) if global_language_state is not None else None
        )
        null_alignment_mask = torch.zeros_like(null_text_mask)
        null_alignment_mask[:, 0] = True
        text_null = EncodedConditioning(
            text_mask=null_text_mask,
            alignment_mask=null_alignment_mask,
            speaker_mask=speaker_attention_mask,
            text_state=null_text_state,
            speaker_state=speaker_state,
            global_language_state=null_global_state,
            text_is_null=torch.ones(batch_size, dtype=torch.bool, device=device),
        )
        speaker_null = EncodedConditioning(
            text_mask=text_mask,
            alignment_mask=alignment_mask,
            speaker_mask=null_speaker_mask,
            text_state=text_state,
            speaker_state=null_speaker_state,
            global_language_state=global_language_state,
            text_is_null=torch.zeros(batch_size, dtype=torch.bool, device=device),
        )
        fully_null = EncodedConditioning(
            text_mask=null_text_mask,
            alignment_mask=null_alignment_mask,
            speaker_mask=null_speaker_mask,
            text_state=null_text_state,
            speaker_state=null_speaker_state,
            global_language_state=null_global_state,
            text_is_null=torch.ones(batch_size, dtype=torch.bool, device=device),
        )

        def combine(branches: tuple[EncodedConditioning, ...]) -> EncodedConditioning:
            def combine_mask(name: str, length: int) -> torch.Tensor | None:
                values = [getattr(branch, name) for branch in branches]
                if all(value is None for value in values):
                    return None
                return torch.cat(
                    [
                        self._materialize_mask(
                            value,
                            batch_size=batch_size,
                            length=length,
                            device=device,
                        )
                        for value in values
                    ],
                    dim=0,
                )

            global_states = [branch.global_language_state for branch in branches]
            return EncodedConditioning(
                text_mask=combine_mask("text_mask", text_state.shape[1]),
                alignment_mask=combine_mask("alignment_mask", text_state.shape[1]),
                speaker_mask=combine_mask("speaker_mask", speaker_state.shape[1]),
                text_state=torch.cat([branch.text_state for branch in branches], dim=0),
                speaker_state=torch.cat([branch.speaker_state for branch in branches], dim=0),
                global_language_state=(
                    torch.cat(
                        [
                            state if state is not None else torch.zeros_like(global_states[0])
                            for state in global_states
                        ],
                        dim=0,
                    )
                    if global_states[0] is not None
                    else None
                ),
                text_is_null=torch.cat(
                    [
                        branch.text_is_null
                        if branch.text_is_null is not None
                        else torch.zeros(batch_size, dtype=torch.bool, device=device)
                        for branch in branches
                    ],
                    dim=0,
                ),
            )

        if cfg_mode == "joint":
            branch_count = 2
            variants = (combine((conditional, fully_null)),)
        elif cfg_mode == "independent":
            branch_count = 3
            variants = (combine((conditional, text_null, speaker_null)),)
        else:
            branch_count = 2
            variants = (
                combine((conditional, text_null)),
                combine((conditional, speaker_null)),
            )
        return EncodedCFGConditioning(
            mode=cfg_mode,
            branch_count=branch_count,
            variants=variants,
            conditional=conditional,
        )

    def _encode_legacy_v3_inference_conditioning(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        speaker_latent: torch.Tensor | None,
        *,
        cfg_mode: str | None,
        speaker_mask: torch.Tensor | None,
        language_ids: torch.Tensor | None,
        token_language_ids: torch.Tensor | None,
        alignment_mask: torch.Tensor | None,
        conditioning_features: torch.Tensor | None,
    ) -> EncodedCFGConditioning:
        """Reproduce the authenticated architecture-v3 encoder/CFG graph exactly."""
        if conditioning_features is not None:
            raise ValueError("EchoDiT architecture v3 does not accept frozen text features.")
        batch_size = conditioning_ids.shape[0]
        device = conditioning_ids.device
        text_mask = attention_mask.bool() if attention_mask is not None else None
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=batch_size,
            device=device,
        )
        # The schema-2 runtime supplies parallel compatibility axes so modern batch
        # assembly remains shape-safe. Architecture v3 itself conditioned text only
        # with the utterance-level language ID and treated every non-padding token as
        # alignable, so validate those facts and deliberately keep them out of the
        # historical encoder graph.
        self._validate_legacy_v3_parallel_text_axes(
            conditioning_ids=conditioning_ids,
            text_mask=text_mask,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
        )

        def null_speaker_like(reference: torch.Tensor | None) -> torch.Tensor | None:
            if not self.use_speaker_conditioning or reference is None:
                return None
            null_speaker = self.null_speaker_embed.to(
                device=reference.device,
                dtype=reference.dtype,
            ).expand(batch_size, -1, -1)
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
        flat_speaker_state = self.dit.encode_speaker(
            flat_speaker_latent,
            flat_speaker_mask,
        )

        variant_size = branch_count * batch_size
        variants = tuple(
            EncodedConditioning(
                text_mask=(
                    flat_text_mask[offset : offset + variant_size]
                    if flat_text_mask is not None
                    else None
                ),
                alignment_mask=None,
                speaker_mask=(
                    flat_speaker_mask[offset : offset + variant_size]
                    if flat_speaker_mask is not None
                    else None
                ),
                text_state=flat_text_state[offset : offset + variant_size],
                speaker_state=flat_speaker_state[offset : offset + variant_size],
                global_language_state=None,
                text_is_null=None,
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
                if variant.text_is_null is not None and bool(variant.text_is_null.any()):
                    totals = variant_durations.sum(dim=1)
                    variant_durations = variant_durations.clone()
                    variant_durations[variant.text_is_null] = 0
                    variant_durations[variant.text_is_null, 0] = totals[variant.text_is_null]
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
                global_language_state=variant.global_language_state,
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

        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            if speaker_mask is not None:
                raise ValueError("speaker_mask requires speaker_latent.")
            speaker_encoder = self.dit.speaker_encoder
            if speaker_encoder is None:
                raise RuntimeError("EchoDiT v3 speaker state is internally inconsistent.")
            return (
                self.null_speaker_embed.to(
                    device=device,
                    dtype=speaker_encoder.in_proj.weight.dtype,
                ).expand(batch_size, -1, -1),
                None,
            )

        raise ValueError(
            "speaker_latent is required here; use learned null_speaker_state for an "
            "unconditional speaker branch."
        )

    def _encode_speaker_conditioning(
        self,
        speaker_latent: torch.Tensor | None,
        speaker_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode a real reference once or return the learned encoder-level null state."""
        if not self.use_speaker_conditioning:
            if speaker_latent is not None or speaker_mask is not None:
                raise ValueError("Speaker inputs require a speaker-conditioned NAR-VAE checkpoint.")
            return self._null_speaker_conditioning(
                batch_size=batch_size,
                state_count=0,
                device=device,
                dtype=self.dit.in_proj.weight.dtype,
            )
        if speaker_latent is None and self.architecture_version >= ECHODIT_ARCHITECTURE_VERSION:
            if speaker_mask is not None:
                raise ValueError("speaker_mask requires speaker_latent.")
            return self._null_speaker_conditioning(
                batch_size=batch_size,
                state_count=max(
                    1,
                    int(getattr(self.dit, "speaker_num_summary_tokens", 0)),
                ),
                device=device,
                dtype=self.dit.in_proj.weight.dtype,
            )

        prepared_latent, patch_mask = self._prepare_speaker_inputs(
            speaker_latent,
            speaker_mask,
            batch_size=batch_size,
            device=device,
        )
        encode_speaker = getattr(self.dit, "encode_speaker", None)
        if not callable(encode_speaker):
            # Compatibility for small protocol fakes used by downstream callers.
            return prepared_latent, patch_mask
        speaker_state = encode_speaker(prepared_latent, patch_mask)
        summary_token_count = int(getattr(self.dit, "speaker_num_summary_tokens", 0))
        if summary_token_count > 0:
            if speaker_state.shape[1] != summary_token_count:
                raise RuntimeError(
                    "Speaker resampler output length does not match speaker_num_summary_tokens."
                )
            # The source padding mask has been consumed by cross-attention and
            # every learned summary query is a valid downstream conditioning token.
            return speaker_state, None
        if patch_mask is None:
            return speaker_state, None
        if speaker_state.shape[1] == patch_mask.shape[1] + 1:
            global_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            return speaker_state, torch.cat((global_mask, patch_mask), dim=1)
        if speaker_state.shape[1] != patch_mask.shape[1]:
            raise RuntimeError(
                "Speaker encoder output length must equal the patch count or include one "
                "leading global timbre token."
            )
        return speaker_state, patch_mask

    def predict_log_duration(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``log1p`` DACVAE frame count with a trained EchoDiT v2 head."""
        text_state, text_mask, _, speaker_state, speaker_attention_mask = (
            self._encode_duration_conditioning(
                conditioning_ids,
                attention_mask,
                speaker_latent,
                speaker_mask,
                language_ids,
                token_language_ids,
                alignment_mask,
                conditioning_features,
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
        token_language_ids: torch.Tensor | None,
        alignment_mask: torch.Tensor | None,
        conditioning_features: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
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
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            if conditioning_features is not None:
                raise ValueError("EchoDiT architecture v3 does not accept frozen text features.")
            self._validate_legacy_v3_parallel_text_axes(
                conditioning_ids=conditioning_ids,
                text_mask=text_mask,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
            )
            # Origin architecture v3 ran the duration and MAS heads over every
            # cl100k/control token. Never apply v4 alignment compaction here.
            text_state = self.dit.encode_text(conditioning_ids, text_mask, language_ids)
            full_alignment_mask = self._materialize_mask(
                text_mask,
                batch_size=batch_size,
                length=conditioning_ids.shape[1],
                device=device,
            )
            speaker_state = None
            speaker_attention_mask = None
            if self.duration_predictor_use_speaker:
                speaker_state, speaker_attention_mask = self._encode_speaker_conditioning(
                    speaker_latent,
                    speaker_mask,
                    batch_size=batch_size,
                    device=device,
                )
            return (
                text_state,
                text_mask,
                full_alignment_mask,
                speaker_state,
                speaker_attention_mask,
            )
        token_language_ids = self._prepare_token_language_ids(
            token_language_ids,
            conditioning_ids=conditioning_ids,
        )
        alignment_mask = self._prepare_alignment_mask(
            alignment_mask,
            conditioning_ids=conditioning_ids,
            text_mask=text_mask,
        )
        encode_text_kwargs = {}
        if token_language_ids is not None:
            encode_text_kwargs["token_language_ids"] = token_language_ids
        if conditioning_features is not None:
            encode_text_kwargs["conditioning_features"] = conditioning_features
        full_text_state = self.dit.encode_text(
            conditioning_ids,
            text_mask,
            language_ids,
            **encode_text_kwargs,
        )
        effective_alignment_mask = (
            alignment_mask
            if alignment_mask is not None
            else self._materialize_mask(
                text_mask,
                batch_size=batch_size,
                length=conditioning_ids.shape[1],
                device=device,
            )
        )
        text_state, duration_text_mask = self._compact_text_state(
            full_text_state,
            effective_alignment_mask,
        )

        speaker_state = None
        speaker_attention_mask = None
        if self.duration_predictor_use_speaker:
            speaker_state, speaker_attention_mask = self._encode_speaker_conditioning(
                speaker_latent,
                speaker_mask,
                batch_size=batch_size,
                device=device,
            )

        return (
            text_state,
            duration_text_mask,
            effective_alignment_mask,
            speaker_state,
            speaker_attention_mask,
        )

    def predict_expected_token_durations(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        speaker_latent: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict positive floating-point frame contributions for each valid token."""
        if not getattr(self, "use_mas_duration", False):
            raise RuntimeError(
                "Token duration prediction requires a versioned MAS-duration checkpoint."
            )
        text_state, text_mask, full_alignment_mask, speaker_state, speaker_attention_mask = (
            self._encode_duration_conditioning(
                conditioning_ids,
                attention_mask,
                speaker_latent,
                speaker_mask,
                language_ids,
                token_language_ids,
                alignment_mask,
                conditioning_features,
            )
        )
        _, token_durations = self._predict_duration_components_from_encoded(
            text_state,
            text_mask,
            speaker_state,
            speaker_attention_mask,
        )
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            return token_durations
        return self._scatter_compact_rows(token_durations, full_alignment_mask)

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
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
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
        alignment_mask = (
            conditional.alignment_mask
            if conditional.alignment_mask is not None
            else self._materialize_mask(
                conditional.text_mask,
                batch_size=conditional.text_state.shape[0],
                length=conditional.text_state.shape[1],
                device=conditional.text_state.device,
            )
        )
        duration_text_state, duration_text_mask = self._compact_text_state(
            conditional.text_state,
            alignment_mask,
        )
        if getattr(self, "use_mas_duration", False):
            log_frames, token_weights = self._predict_duration_components_from_encoded(
                duration_text_state,
                duration_text_mask,
                speaker_state,
                speaker_mask,
            )
            token_weights = self._scatter_compact_rows(token_weights, alignment_mask)
        else:
            log_frames = self._predict_log_duration_from_encoded(
                duration_text_state,
                duration_text_mask,
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
        alignment_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Allocate exact positive MAS durations from already predicted token weights."""
        selected_mask = alignment_mask if alignment_mask is not None else attention_mask
        token_mask = (
            selected_mask.to(device=expected.device, dtype=torch.bool)
            if selected_mask is not None
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
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
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
            token_language_ids,
            alignment_mask,
            conditioning_features,
        )
        return self.allocate_token_duration_frames(
            expected,
            attention_mask,
            total_frames=total_frames,
            alignment_mask=alignment_mask,
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
            global_language_state=conditioning.global_language_state,
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
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> PreparedCFGConditioning:
        """Precompute every fixed CFG branch needed during one request."""
        encoded = self.encode_inference_conditioning(
            conditioning_ids,
            attention_mask,
            speaker_latent,
            cfg_mode=cfg_mode,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            conditioning_features=conditioning_features,
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
        token_language_ids: torch.Tensor | None = None,  # [B, L] span-language IDs
        alignment_mask: torch.Tensor | None = None,  # [B, L] frame-owning text tokens
        conditioning_features: torch.Tensor | None = None,  # [B, L, F] frozen text states
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | DurationAlignmentOutput]:
        """
        Forward pass for training.

        Args:
            latents: Noisy latents [B, latent_size, T]
            conditioning_ids: Text token IDs [B, L]
            conditioning_features: Optional cached frozen-backbone states [B, L, F]
            timesteps: Timesteps [B] in range [0, 1]
            attention_mask: Text attention mask [B, L]
            speaker_latent: Speaker conditioning latents [B, D, T_speaker] (optional)
            use_cfg_dropout: Whether to apply CFG dropout
            speaker_mask: Valid speaker frames or patches (optional)
            language_ids: Stable target-language IDs (optional for legacy English)
            token_language_ids: Optional per-token language IDs for code-switched text
            alignment_mask: Possibly non-contiguous tokens allowed to own acoustic frames
            latent_mask: Valid target-latent positions for padded training batches
            return_duration_prediction: Return the v2 log-frame prediction for joint training
            return_duration_alignment: Return versioned per-token predictions and likelihoods
            duration_target_latents: Clean target latents used only by the alignment objective
            token_durations: Exact inference-time token-to-frame allocation for MAS checkpoints

        Returns:
            Predicted velocity, optionally paired with predicted log-frame count
        """
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            if self.training:
                raise RuntimeError(
                    "EchoDiT architecture v3 is authenticated for inference only; "
                    "legacy training is unsupported."
                )
            if (
                return_duration_prediction
                or return_duration_alignment
                or duration_target_latents is not None
            ):
                raise RuntimeError(
                    "EchoDiT architecture v3 does not expose current training-return paths."
                )
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
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            if conditioning_features is not None:
                raise ValueError("EchoDiT architecture v3 does not accept frozen text features.")
            self._validate_legacy_v3_parallel_text_axes(
                conditioning_ids=conditioning_ids,
                text_mask=text_mask,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
            )
            # Compatibility axes are deliberately excluded from the origin v3
            # encoder/duration graph after exact no-op validation.
            token_language_ids = None
            alignment_mask = None
        else:
            token_language_ids = self._prepare_token_language_ids(
                token_language_ids,
                conditioning_ids=conditioning_ids,
            )
            alignment_mask = self._prepare_alignment_mask(
                alignment_mask,
                conditioning_ids=conditioning_ids,
                text_mask=text_mask,
            )
        effective_alignment_mask = (
            alignment_mask
            if alignment_mask is not None
            else self._materialize_mask(
                text_mask,
                batch_size=B,
                length=conditioning_ids.shape[1],
                device=device,
            )
        )

        # Encode each real condition exactly once. Duration/MAS uses compacted
        # frame-owning tokens, while the complete state remains available to DiT.
        encode_text_kwargs = {}
        if token_language_ids is not None:
            encode_text_kwargs["token_language_ids"] = token_language_ids
        if conditioning_features is not None:
            encode_text_kwargs["conditioning_features"] = conditioning_features
        complete_text_state = self.dit.encode_text(
            conditioning_ids,
            text_mask,
            language_ids,
            **encode_text_kwargs,
        )
        encode_global_language = getattr(self.dit, "encode_global_language", None)
        global_language_state = (
            encode_global_language(language_ids, batch_size=B, device=device)
            if callable(encode_global_language)
            else None
        )
        complete_speaker_state, complete_speaker_mask = self._encode_speaker_conditioning(
            speaker_latent,
            speaker_mask,
            batch_size=B,
            device=device,
        )

        duration_prediction = None
        hard_alignment = None
        if return_duration_prediction:
            if not getattr(self, "use_duration_predictor", False):
                raise RuntimeError(
                    "Learned duration requires a versioned EchoDiT v2 duration checkpoint."
                )
            duration_text_state, duration_text_mask = self._compact_text_state(
                complete_text_state,
                effective_alignment_mask,
            )
            duration_speaker_state = (
                complete_speaker_state if self.duration_predictor_use_speaker else None
            )
            duration_speaker_mask = (
                complete_speaker_mask if self.duration_predictor_use_speaker else None
            )
            if return_duration_alignment:
                total_log_frames, compact_token_durations = (
                    self._predict_duration_components_from_encoded(
                        duration_text_state,
                        duration_text_mask,
                        duration_speaker_state,
                        duration_speaker_mask,
                    )
                )
                compact_log_likelihoods = self.duration_alignment(
                    duration_text_state,
                    duration_target_latents,
                )
                alignment_frame_mask = (
                    latent_mask.to(device=device, dtype=torch.bool)
                    if latent_mask is not None
                    else torch.ones(
                        (B, latents.shape[-1]),
                        device=device,
                        dtype=torch.bool,
                    )
                )
                with torch.no_grad():
                    compact_hard_alignment = monotonic_alignment_search(
                        compact_log_likelihoods.detach(),
                        duration_text_mask,
                        alignment_frame_mask,
                    )
                predicted_token_durations = self._scatter_compact_rows(
                    compact_token_durations,
                    effective_alignment_mask,
                )
                log_likelihoods = self._scatter_compact_rows(
                    compact_log_likelihoods,
                    effective_alignment_mask,
                )
                hard_alignment = self._scatter_compact_rows(
                    compact_hard_alignment,
                    effective_alignment_mask,
                )
                duration_prediction = DurationAlignmentOutput(
                    total_log_frames=total_log_frames,
                    token_durations=predicted_token_durations,
                    log_likelihoods=log_likelihoods,
                    hard_alignment=hard_alignment,
                )
            else:
                duration_prediction = self._predict_log_duration_from_encoded(
                    duration_text_state,
                    duration_text_mask,
                    duration_speaker_state,
                    duration_speaker_mask,
                )

        text_state = complete_text_state
        velocity_text_mask = text_mask
        speaker_state = complete_speaker_state
        velocity_speaker_mask = complete_speaker_mask
        text_cfg_mask = None
        speaker_cfg_mask = None
        if self.training and use_cfg_dropout:
            text_cfg_mask = torch.rand(B, device=device) < self.cfg_dropout_text
            if bool(text_cfg_mask.any()):
                null_text_state, null_text_mask = self._null_text_conditioning(
                    batch_size=B,
                    token_count=text_state.shape[1],
                    state_size=text_state.shape[2],
                    device=text_state.device,
                    dtype=text_state.dtype,
                )
                text_state = torch.where(text_cfg_mask[:, None, None], null_text_state, text_state)
                materialized_text_mask = self._materialize_mask(
                    velocity_text_mask,
                    batch_size=B,
                    length=text_state.shape[1],
                    device=device,
                )
                velocity_text_mask = torch.where(
                    text_cfg_mask[:, None],
                    null_text_mask,
                    materialized_text_mask,
                )
                if global_language_state is not None:
                    global_language_state = torch.where(
                        text_cfg_mask[:, None],
                        torch.zeros_like(global_language_state),
                        global_language_state,
                    )

            if self.use_speaker_conditioning and speaker_latent is not None:
                speaker_cfg_mask = torch.rand(B, device=device) < self.cfg_dropout_speaker
                if bool(speaker_cfg_mask.any()):
                    null_speaker_state, null_speaker_mask = self._null_speaker_conditioning(
                        batch_size=B,
                        state_count=speaker_state.shape[1],
                        state_size=speaker_state.shape[2],
                        device=speaker_state.device,
                        dtype=speaker_state.dtype,
                    )
                    speaker_state = torch.where(
                        speaker_cfg_mask[:, None, None],
                        null_speaker_state,
                        speaker_state,
                    )
                    materialized_speaker_mask = self._materialize_mask(
                        velocity_speaker_mask,
                        batch_size=B,
                        length=speaker_state.shape[1],
                        device=device,
                    )
                    null_speaker_mask = self._materialize_mask(
                        null_speaker_mask,
                        batch_size=B,
                        length=speaker_state.shape[1],
                        device=device,
                    )
                    velocity_speaker_mask = torch.where(
                        speaker_cfg_mask[:, None],
                        null_speaker_mask,
                        materialized_speaker_mask,
                    )

        frame_text_state = None
        if hard_alignment is not None:
            frame_text_state = torch.bmm(
                hard_alignment.transpose(1, 2).to(dtype=complete_text_state.dtype),
                complete_text_state,
            )
        elif getattr(self, "use_mas_duration", False) and token_durations is not None:
            frame_text_state = expand_text_by_durations(
                complete_text_state,
                token_durations,
                target_frames=latents.shape[-1],
            )
        if frame_text_state is not None and text_cfg_mask is not None and bool(text_cfg_mask.any()):
            frame_null = self.null_text_embed.to(
                device=frame_text_state.device,
                dtype=frame_text_state.dtype,
            ).expand(B, frame_text_state.shape[1], -1)
            frame_text_state = torch.where(
                text_cfg_mask[:, None, None],
                frame_null,
                frame_text_state,
            )

        project_speaker = getattr(self.dit, "project_speaker_kv_cache", None)
        kv_cache_speaker = (
            project_speaker(speaker_state)
            if callable(project_speaker)
            else self.dit.get_kv_cache_speaker(speaker_state, velocity_speaker_mask)
        )
        conditioning = PreparedConditioning(
            text_mask=velocity_text_mask,
            speaker_mask=velocity_speaker_mask,
            kv_cache_text=self.dit.project_text_kv_cache(text_state),
            kv_cache_speaker=kv_cache_speaker,
            frame_text_state=frame_text_state,
            global_language_state=global_language_state,
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
        token_language_ids: torch.Tensor | None = None,
        alignment_mask: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
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
            token_language_ids: Optional per-token language IDs for code-switched text
            alignment_mask: Possibly non-contiguous tokens allowed to own acoustic frames
            conditioning_features: Cached frozen-backbone states aligned to conditioning IDs

        Returns:
            Guided velocity prediction [B, D, T]
        """
        # CFG is assembled only from explicit learned encoded-null states. The real
        # text/reference encoders run once, independently of branch count.
        if cfg_scale == 1.0 and cfg_mode == "joint":
            return self.forward(
                latents,
                conditioning_ids,
                timesteps,
                attention_mask,
                speaker_latent,
                use_cfg_dropout=False,
                speaker_mask=speaker_mask,
                language_ids=language_ids,
                token_durations=token_durations,
                token_language_ids=token_language_ids,
                alignment_mask=alignment_mask,
                conditioning_features=conditioning_features,
            )

        encoded = self.encode_inference_conditioning(
            conditioning_ids,
            attention_mask,
            speaker_latent,
            cfg_mode=cfg_mode,
            speaker_mask=speaker_mask,
            language_ids=language_ids,
            token_language_ids=token_language_ids,
            alignment_mask=alignment_mask,
            conditioning_features=conditioning_features,
        )
        _, prepared_cfg = self.finalize_inference_conditioning(
            encoded,
            token_durations=token_durations,
        )
        assert prepared_cfg is not None
        resolved_step = 0 if step_idx is None else step_idx
        if fuse_cfg_branches:
            return self.forward_with_prepared_cfg(
                latents,
                timesteps,
                prepared_cfg,
                cfg_scale=cfg_scale,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_speaker=cfg_scale_speaker,
                step_idx=resolved_step,
            )

        variant_index = resolved_step % 2 if cfg_mode == "alternating" else 0
        fused_variant = prepared_cfg.variants[variant_index]
        batch_size = latents.shape[0]
        predictions = tuple(
            self.forward_prepared(
                latents,
                timesteps,
                fused_variant.slice_batch(
                    branch_index * batch_size,
                    (branch_index + 1) * batch_size,
                ),
            )
            for branch_index in range(prepared_cfg.branch_count)
        )
        text_scale = cfg_scale if cfg_scale_text is None else cfg_scale_text
        speaker_scale = cfg_scale if cfg_scale_speaker is None else cfg_scale_speaker
        if cfg_mode == "joint":
            conditional, unconditional = predictions
            return unconditional + cfg_scale * (conditional - unconditional)
        if cfg_mode == "independent":
            conditional, unconditional_text, unconditional_speaker = predictions
            return (
                conditional
                + text_scale * (conditional - unconditional_text)
                + speaker_scale * (conditional - unconditional_speaker)
            )
        conditional, unconditional = predictions
        scale = text_scale if resolved_step % 2 == 0 else speaker_scale
        return unconditional + scale * (conditional - unconditional)

    @torch.no_grad()
    def encode_text(
        self,
        conditioning_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        language_ids: torch.Tensor | None = None,
        token_language_ids: torch.Tensor | None = None,
        conditioning_features: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Encode text and return KV cache."""
        if attention_mask is None:
            attention_mask = torch.ones_like(conditioning_ids, dtype=torch.bool)
        language_ids = self._prepare_language_ids(
            language_ids,
            batch_size=conditioning_ids.shape[0],
            device=conditioning_ids.device,
        )
        if self.architecture_version == LEGACY_ECHODIT_ARCHITECTURE_VERSION:
            if conditioning_features is not None:
                raise ValueError("EchoDiT architecture v3 does not accept frozen text features.")
            self._validate_legacy_v3_parallel_text_axes(
                conditioning_ids=conditioning_ids,
                text_mask=attention_mask,
                language_ids=language_ids,
                token_language_ids=token_language_ids,
                alignment_mask=None,
            )
            token_language_ids = None
        else:
            token_language_ids = self._prepare_token_language_ids(
                token_language_ids,
                conditioning_ids=conditioning_ids,
            )
        return self.dit.get_kv_cache_text(
            conditioning_ids,
            attention_mask,
            language_ids,
            token_language_ids=token_language_ids,
            conditioning_features=conditioning_features,
        )

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
    text_vocab_size: int = TOTAL_VOCAB_SIZE,
    text_conditioning_mode: str = SCRATCH_TOKEN_TEXT_CONDITIONING,
    conditioning_feature_size: int | None = None,
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
    target_patch_size: int = 1,
    generative_objective: str = RECTIFIED_FLOW_OBJECTIVE,
    diffusion_schedule_shift: float = DEFAULT_DIFFUSION_SCHEDULE_SHIFT,
    speaker_num_summary_tokens: int = 0,
    architecture_version: int = ECHODIT_ARCHITECTURE_VERSION,
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
        text_conditioning_mode: ``scratch_tokens`` or token-aligned ``frozen_features``
        conditioning_feature_size: Frozen feature width; required only in frozen-feature mode
        target_patch_size: Exact temporal target-latent packing factor
        cfg_dropout: Default classifier-free conditioning dropout rate
        cfg_dropout_text: Optional text-specific override of ``cfg_dropout``
        cfg_dropout_speaker: Optional speaker-specific override of ``cfg_dropout``
        use_speaker_conditioning: Enable speaker conditioning
        speaker_num_summary_tokens: Fixed speaker-summary token count; zero keeps legacy topology
        architecture_version: Exact authenticated EchoDiT topology version
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
        text_conditioning_mode=text_conditioning_mode,
        conditioning_feature_size=conditioning_feature_size,
        target_patch_size=target_patch_size,
        generative_objective=generative_objective,
        diffusion_schedule_shift=diffusion_schedule_shift,
        cfg_dropout=cfg_dropout,
        cfg_dropout_text=cfg_dropout_text,
        cfg_dropout_speaker=cfg_dropout_speaker,
        use_speaker_conditioning=use_speaker_conditioning,
        speaker_num_summary_tokens=speaker_num_summary_tokens,
        architecture_version=architecture_version,
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
