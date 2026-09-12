import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.utils import (
    cal_padded_tokens,
    pad_dsa_cache_seqlens,
)
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipIf(not torch.cuda.is_available(), "GPU is required")
class TestDSAPaddingMetadata(CustomTestCase):
    def test_attention_tp_alignment_covers_physical_query_rows(self):
        # This is the eager speculative shape transition from the issue: DSA
        # planned three real rows, while MLP-sync pads the physical query to the
        # attention-TP width (four rows for TP=2).
        batch = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=3,
            input_ids=torch.arange(3, device="cuda"),
            req_pool_indices=torch.arange(3, device="cuda"),
            seq_lens=torch.ones(3, dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.arange(3, device="cuda"),
            seq_lens_sum=3,
        )
        batch.global_num_tokens_cpu = [3]
        batch.dp_padding_mode = DpPaddingMode.SUM_LEN
        parallel = SimpleNamespace(
            attn_tp_size=2,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_rank=0,
        )
        planned_cache_seqlens = torch.tensor(
            [11, 12, 13], dtype=torch.int32, device="cuda"
        )

        with patch(
            "sglang.srt.layers.attention.dsa.utils.get_parallel",
            return_value=parallel,
        ):
            physical_rows = cal_padded_tokens(batch)
            padded_cache_seqlens = pad_dsa_cache_seqlens(
                batch, planned_cache_seqlens
            )

        self.assertEqual(physical_rows, 4)
        self.assertEqual(padded_cache_seqlens.shape[0], physical_rows)
        torch.testing.assert_close(
            padded_cache_seqlens,
            torch.tensor([11, 12, 13, 0], dtype=torch.int32, device="cuda"),
        )
        # Both downstream DSA consumers derive row offsets from this metadata.
        cu_seqlens = torch.nn.functional.pad(
            padded_cache_seqlens.cumsum(0), (1, 0)
        )
        self.assertEqual(cu_seqlens.shape[0], physical_rows + 1)


if __name__ == "__main__":
    unittest.main()
