from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.gather_spec_extras import gather_spec_extras
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


class TestOverlapHiddenStateHandoff(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm is required")
    def test_stash_conversion_and_gather_match_reference(self):
        device = torch.device("cuda")
        pool_size = 8
        rows = 5
        draft_tokens = 3
        hidden_dim = 17
        topk = 2
        indices = torch.tensor([1, 3, 4, 2, 0], device=device, dtype=torch.int64)
        conversion_cases = (
            (torch.bfloat16, (torch.float32, torch.float16)),
            (torch.float16, (torch.bfloat16, torch.float32)),
            (torch.float32, (torch.bfloat16, torch.float16)),
        )

        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            for buffer_dtype, payload_dtypes in conversion_cases:
                with self.subTest(
                    buffer_dtype=buffer_dtype, payload_dtypes=payload_dtypes
                ):
                    req_to_token_pool = SimpleNamespace(
                        req_to_token=torch.zeros(
                            (pool_size, 16), dtype=torch.int64, device=device
                        )
                    )
                    future_map = FutureMap(
                        device=device,
                        spec_algo=SpeculativeAlgorithm.EAGLE,
                        req_to_token_pool=req_to_token_pool,
                        needs_cpu_seq_lens=False,
                    )
                    initial_hidden = torch.randn(
                        (rows, draft_tokens, hidden_dim),
                        dtype=buffer_dtype,
                        device=device,
                    )
                    future_map.stash(
                        indices,
                        RelayPayload(
                            bonus_tokens=torch.arange(rows, device=device),
                            topk_p=torch.rand((rows, topk), device=device),
                            topk_index=torch.randint(
                                0, 1000, (rows, topk), device=device
                            ),
                            hidden_states=initial_hidden,
                        ),
                    )

                    self.assertEqual(future_map.hidden_states_buf.dtype, buffer_dtype)
                    self.assertEqual(
                        tuple(future_map.hidden_states_buf.shape),
                        (pool_size, draft_tokens, hidden_dim),
                    )
                    self.assertTrue(future_map.hidden_states_buf.is_contiguous())
                    self.assertEqual(
                        future_map.hidden_states_buf.stride(),
                        (draft_tokens * hidden_dim, hidden_dim, 1),
                    )

                    for payload_dtype in payload_dtypes:
                        with self.subTest(payload_dtype=payload_dtype):
                            hidden_source = (
                                torch.randn(
                                    (rows, draft_tokens, hidden_dim * 2),
                                    dtype=payload_dtype,
                                    device=device,
                                )[:, :, ::2]
                            )
                            self.assertFalse(hidden_source.is_contiguous())
                            future_map.stash(
                                indices,
                                RelayPayload(
                                    bonus_tokens=torch.arange(rows, device=device),
                                    topk_p=torch.rand((rows, topk), device=device),
                                    topk_index=torch.randint(
                                        0, 1000, (rows, topk), device=device
                                    ),
                                    hidden_states=hidden_source,
                                ),
                            )

                            expected = torch.empty_like(future_map.hidden_states_buf)
                            expected[indices] = hidden_source.to(buffer_dtype)
                            torch.testing.assert_close(
                                future_map.hidden_states_buf[indices],
                                expected[indices],
                                rtol=0,
                                atol=0,
                            )

                            gathered = gather_spec_extras(
                                indices,
                                future_map.topk_p_buf,
                                future_map.topk_index_buf,
                                future_map.output_tokens_buf,
                                future_map.hidden_states_buf,
                            )[3]
                            self.assertEqual(gathered.dtype, buffer_dtype)
                            self.assertEqual(
                                tuple(gathered.shape),
                                (rows, draft_tokens, hidden_dim),
                            )
                            self.assertTrue(gathered.is_contiguous())
                            torch.testing.assert_close(
                                gathered,
                                expected[indices],
                                rtol=0,
                                atol=0,
                            )


if __name__ == "__main__":
    unittest.main()
