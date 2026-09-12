import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    _resolve_dflash_draft_cell_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDflashDraftKvBudgetTopology(unittest.TestCase):
    def _resolve(self, *, tp_size: int, attn_tp_size: int) -> tuple[int, list[int]]:
        observed_tp_sizes = []
        draft_model_config = SimpleNamespace(
            dtype=torch.bfloat16,
            head_dim=128,
            v_head_dim=128,
            get_num_kv_heads=lambda width: observed_tp_sizes.append(width) or 16 // width,
        )
        parallel = SimpleNamespace(tp_size=tp_size, attn_tp_size=attn_tp_size)
        model = SimpleNamespace(kv_cache_dtype="auto")
        spec = SimpleNamespace(
            speculative_draft_kv_cache_dtype=None,
            speculative_draft_attention_backend=None,
        )

        with (
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.get_parallel",
                return_value=parallel,
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.get_model",
                return_value=model,
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components."
                "spec_aux_hidden_state.get_spec",
                return_value=spec,
            ),
            patch(
                "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
                return_value=("auto", torch.bfloat16),
            ),
        ):
            result = _resolve_dflash_draft_cell_size(
                draft_model_config=draft_model_config,
                draft_num_layers=5,
            )

        return result, observed_tp_sizes

    def test_dp_attention_uses_attention_tp_width(self):
        result, observed = self._resolve(tp_size=16, attn_tp_size=1)

        self.assertEqual(observed, [1])
        self.assertEqual(result, 16 * (128 + 128) * 5 * 2)

    def test_partial_dp_attention_uses_attention_tp_width(self):
        result, observed = self._resolve(tp_size=16, attn_tp_size=4)

        self.assertEqual(observed, [4])
        self.assertEqual(result, 4 * (128 + 128) * 5 * 2)

    def test_without_dp_attention_widths_are_unchanged(self):
        result, observed = self._resolve(tp_size=16, attn_tp_size=16)

        self.assertEqual(observed, [16])
        self.assertEqual(result, (128 + 128) * 5 * 2)


if __name__ == "__main__":
    unittest.main()
