"""Portable regression tests for the Gemma 4 MLX cache bridge policy."""

import unittest
from types import SimpleNamespace

from sglang.srt.hardware_backend.mlx.native_cache import (
    uses_model_native_cache,
    validate_model_native_cache_config,
)
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")


class TestGemma4NativeCachePolicy(unittest.TestCase):
    def test_public_wrapper_and_text_backbone_are_recognized(self):
        self.assertTrue(
            uses_model_native_cache(
                SimpleNamespace(
                    model_type="gemma4", args=SimpleNamespace(model_type="gemma4")
                )
            )
        )
        self.assertTrue(
            uses_model_native_cache(
                SimpleNamespace(
                    model_type="wrapper",
                    language_model=SimpleNamespace(
                        args=SimpleNamespace(model_type="gemma4_text")
                    ),
                )
            )
        )
        self.assertFalse(uses_model_native_cache(SimpleNamespace(model_type="gemma3")))

    def test_all_unsafe_features_are_reported_together(self):
        with self.assertRaisesRegex(
            ValueError,
            r"--disable-radix-cache.*--disable-overlap-schedule.*"
            r"--chunked-prefill-size=-1",
        ):
            validate_model_native_cache_config(
                disable_radix_cache=False,
                disable_overlap_schedule=False,
                chunked_prefill_size=1024,
            )

    def test_conservative_configuration_is_accepted(self):
        validate_model_native_cache_config(
            disable_radix_cache=True,
            disable_overlap_schedule=True,
            chunked_prefill_size=-1,
        )


if __name__ == "__main__":
    unittest.main()
