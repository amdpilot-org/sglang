import json
import os
import unittest

import torch

from sglang.kernels.ops.attention.prefill_attention import context_attention_fwd
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestPrefillAttentionRaggedGQA(CustomTestCase):
    SEQ_LENS = (1, 7, 8, 12, 127, 128)
    HEAD_DIMS = (64, 80, 96, 128)
    NUM_Q_HEADS = 8
    NUM_KV_HEADS = (8, 4, 2, 1)
    WARMUP_ITERATIONS = 3
    TIMED_ITERATIONS = 10
    MAX_ABS_ERROR_LIMIT = 0.05
    MEAN_ABS_ERROR_LIMIT = 0.005

    def setUp(self):
        torch.manual_seed(42)

    def _reference_attention(self, q, k, v, is_causal, group_size):
        q_heads_first = q.permute(1, 0, 2).float()
        k_heads_first = k.permute(1, 0, 2).float().repeat_interleave(group_size, dim=0)
        v_heads_first = v.permute(1, 0, 2).float().repeat_interleave(group_size, dim=0)
        return torch.nn.functional.scaled_dot_product_attention(
            q_heads_first,
            k_heads_first,
            v_heads_first,
            is_causal=is_causal,
        ).permute(1, 0, 2)

    def _run_case(self, head_dim, num_kv_heads, is_causal):
        device = get_device()
        total_tokens = sum(self.SEQ_LENS)
        q = torch.randn(
            total_tokens,
            self.NUM_Q_HEADS,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        k = torch.randn(
            total_tokens,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        v = torch.randn_like(k)
        output = torch.zeros_like(q)
        starts = torch.tensor(
            [sum(self.SEQ_LENS[:index]) for index in range(len(self.SEQ_LENS))],
            dtype=torch.int32,
            device=device,
        )
        lengths = torch.tensor(self.SEQ_LENS, dtype=torch.int32, device=device)

        def kernel():
            context_attention_fwd(
                q,
                k,
                v,
                output,
                starts,
                lengths,
                max(self.SEQ_LENS),
                is_causal=is_causal,
            )

        for _ in range(self.WARMUP_ITERATIONS):
            kernel()
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(self.TIMED_ITERATIONS):
            kernel()
        end_event.record()
        torch.cuda.synchronize()

        max_abs_error = 0.0
        abs_error_sum = 0.0
        element_count = 0
        offset = 0
        for length in self.SEQ_LENS:
            reference = self._reference_attention(
                q[offset : offset + length],
                k[offset : offset + length],
                v[offset : offset + length],
                is_causal,
                self.NUM_Q_HEADS // num_kv_heads,
            )
            error = (output[offset : offset + length].float() - reference).abs()
            max_abs_error = max(max_abs_error, error.max().item())
            abs_error_sum += error.sum().item()
            element_count += error.numel()
            offset += length

        return {
            "head_dim": head_dim,
            "num_q_heads": self.NUM_Q_HEADS,
            "num_kv_heads": num_kv_heads,
            "gqa_ratio": self.NUM_Q_HEADS // num_kv_heads,
            "is_causal": is_causal,
            "seq_lens": list(self.SEQ_LENS),
            "max_abs_error": max_abs_error,
            "mean_abs_error": abs_error_sum / element_count,
            "warm_iterations": self.WARMUP_ITERATIONS,
            "timed_iterations": self.TIMED_ITERATIONS,
            "mean_warm_ms": start_event.elapsed_time(end_event) / self.TIMED_ITERATIONS,
        }

    def test_context_attention_ragged_gqa_bf16(self):
        results = []
        for head_dim in self.HEAD_DIMS:
            for num_kv_heads in self.NUM_KV_HEADS:
                for is_causal in (True, False):
                    result = self._run_case(head_dim, num_kv_heads, is_causal)
                    results.append(result)
                    self.assertLessEqual(
                        result["max_abs_error"], self.MAX_ABS_ERROR_LIMIT
                    )
                    self.assertLessEqual(
                        result["mean_abs_error"], self.MEAN_ABS_ERROR_LIMIT
                    )

        report_path = os.environ.get("SGLANG_PREFILL_RAGGED_GQA_REPORT")
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "gpu": torch.cuda.get_device_name(0),
                        "dtype": "bfloat16",
                        "reference": "float32 Torch scaled_dot_product_attention",
                        "timing_method": "3 warmups and 10 timed calls with CUDA events",
                        "results": results,
                    },
                    handle,
                    indent=2,
                )
                handle.write("\n")


if __name__ == "__main__":
    unittest.main()
