"""CPU-only tests for explicit Flash Attention 3 dispatch."""

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import nar_vae.kernels as kernels


class FlashAttentionDispatchTest(unittest.TestCase):
    def _kernel_state(self):
        return (
            patch.object(kernels, "_FA3_LOAD_ATTEMPTED", False),
            patch.object(kernels, "_HAS_FA3", False),
            patch.object(kernels, "_fa3_kernel", None),
            patch.object(kernels, "USING_FA3", False),
        )

    def test_fa3_is_disabled_by_default(self):
        get_kernel = Mock()
        fake_package = types.SimpleNamespace(get_kernel=get_kernel)
        environment = dict(os.environ)
        environment.pop("NAR_VAE_USE_FA3", None)

        state = self._kernel_state()
        with (
            state[0],
            state[1],
            state[2],
            state[3],
            patch.dict(os.environ, environment, clear=True),
            patch.dict(sys.modules, {"kernels": fake_package}),
        ):
            self.assertIsNone(kernels._load_fa3_kernel())

        get_kernel.assert_not_called()

    def test_fa3_requires_explicit_opt_in_and_a_pinned_revision(self):
        loaded = object()
        get_kernel = Mock(return_value=loaded)
        fake_package = types.SimpleNamespace(get_kernel=get_kernel)
        state = self._kernel_state()

        with (
            state[0],
            state[1],
            state[2],
            state[3],
            patch.dict(os.environ, {"NAR_VAE_USE_FA3": "1"}, clear=True),
            patch.dict(sys.modules, {"kernels": fake_package}),
        ):
            self.assertIs(kernels._load_fa3_kernel(), loaded)

        get_kernel.assert_called_once_with(
            "kernels-community/flash-attn3",
            revision="557701fc200e8964180fafa996316fbd72b854d6",
        )


if __name__ == "__main__":
    unittest.main()
