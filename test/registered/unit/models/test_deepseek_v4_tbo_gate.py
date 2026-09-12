"""Regression tests for the DeepSeek-V4 TBO topology gate."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.models.deepseek_v4 as deepseek_v4
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV4TboGate(CustomTestCase):
    @staticmethod
    def _forward_batch():
        mode = SimpleNamespace(is_extend_without_speculative=lambda: True)
        return SimpleNamespace(
            can_run_tbo=True,
            tbo_children=(object(), object()),
            global_forward_mode=mode,
        )

    def _can_run_tbo(self, *, backend, attn_tp_size, attn_dp_size):
        model = object.__new__(deepseek_v4.DeepseekV4Model)
        model.pp_group = SimpleNamespace(world_size=1)
        parallel = SimpleNamespace(
            attn_tp_size=attn_tp_size,
            attn_dp_size=attn_dp_size,
        )
        with (
            patch("sglang.srt.layers.moe.is_tbo_enabled", return_value=True),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=False),
            patch.object(deepseek_v4, "get_moe_a2a_backend", return_value=backend),
            patch.object(deepseek_v4, "get_parallel", return_value=parallel),
        ):
            return model._can_run_tbo(self._forward_batch())

    def test_non_ep_attention_tp_rejected(self):
        self.assertFalse(
            self._can_run_tbo(
                backend=MoeA2ABackend.NONE,
                attn_tp_size=4,
                attn_dp_size=4,
            )
        )

    def test_non_ep_attention_tp_one_remains_enabled(self):
        self.assertTrue(
            self._can_run_tbo(
                backend=MoeA2ABackend.NONE,
                attn_tp_size=1,
                attn_dp_size=8,
            )
        )

    def test_ep_attention_tp_remains_enabled(self):
        self.assertTrue(
            self._can_run_tbo(
                backend=MoeA2ABackend.DEEPEP,
                attn_tp_size=4,
                attn_dp_size=4,
            )
        )


if __name__ == "__main__":
    unittest.main()
