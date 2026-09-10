import unittest
from unittest.mock import patch

import builtins

import torch

from sglang.srt.layers.moe.token_dispatcher.moriep import (
    DeepEPMode,
    init_mori_op,
    validate_mori_dispatch_capacity,
)


class MoriDispatchCapacityTest(unittest.TestCase):
    def test_rejects_capacities_below_wave64(self):
        for capacity in (-1, 0, 32, 63):
            with self.subTest(capacity=capacity):
                with self.assertRaisesRegex(ValueError, "must be at least 64"):
                    validate_mori_dispatch_capacity(capacity)

    def test_accepts_reported_capacities(self):
        for capacity in (64, 128, 256):
            with self.subTest(capacity=capacity):
                validate_mori_dispatch_capacity(capacity)

    def test_invalid_capacity_is_rejected_before_mori_dispatch_config(self):
        original_import = builtins.__import__

        def reject_mori_import(name, *args, **kwargs):
            if name == "mori":
                raise AssertionError("MoRI must not be imported for an invalid capacity")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_mori_import):
            with self.assertRaisesRegex(ValueError, "must be at least 64"):
                init_mori_op(
                    group=None,
                    router_topk=8,
                    num_experts=8,
                    num_local_experts=1,
                    hidden_size=128,
                    params_dtype=torch.bfloat16,
                    num_max_dispatch_tokens_per_rank=32,
                    deepep_mode=DeepEPMode.NORMAL,
                )


if __name__ == "__main__":
    unittest.main()
