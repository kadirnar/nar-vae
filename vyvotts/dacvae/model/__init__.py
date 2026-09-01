# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import audiotools

from .base import CodecMixin, DACFile
from .dacvae import DAC, DACVAE
from .discriminator import Discriminator

# BaseModel uses these allowlists while loading a bundled checkpoint. Keep the
# integration beside the implementation that actually requires audiotools.
audiotools.ml.BaseModel.INTERN += ["dacvae.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]

__all__ = ["CodecMixin", "DAC", "DACFile", "DACVAE", "Discriminator"]
