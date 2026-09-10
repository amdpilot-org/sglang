"""gfx942 coverage for AITER's FP8 paged MQA logits kernel.

DeepGEMM's SM90/SM100 ``fp8_paged_mqa_logits`` path is architecture-gated and
does not run on MI300X.  This test exercises the installed AITER path with a
small valid cache and a bounded longer-row/page-map matrix, compares against an
independent dequantized dot-product reference, and verifies sentinel bounds.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-a-test-1-gpu-small-amd")


_RUNNABLE = is_hip()
if _RUNNABLE:
    try:
        from aiter.jit.utils.chip_info import get_gfx
        from aiter.ops.triton.attention.pa_mqa_logits import (
            deepgemm_fp8_paged_mqa_logits,
        )
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

        _RUNNABLE = get_gfx() == "gfx942"
    except Exception:
        _RUNNABLE = False


HEAD_DIM = 128
SENTINEL = -12345.0


def _make_case(batch_size, next_n, num_heads, context_lengths, max_model_len, seed):
    device = "cuda"
    torch.manual_seed(seed)
    context_tensor = torch.tensor(
        context_lengths, dtype=torch.int32, device=device
    )
    total_pages = sum(context_lengths)
    page_map = torch.full(
        (batch_size, max_model_len), -1, dtype=torch.int32, device=device
    )
    page_offset = 0
    for batch_index, context_length in enumerate(context_lengths):
        page_map[batch_index, :context_length] = torch.arange(
            page_offset, page_offset + context_length, device=device
        )
        page_offset += context_length

    query = torch.randn(
        batch_size, next_n, num_heads, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    query_fp8 = query.to(get_fp8_e4m3_dtype()).contiguous()
    keys = torch.randn(
        total_pages, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    key_amax = keys.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    key_scales = torch.pow(
        2.0, torch.ceil(torch.log2(key_amax / torch.finfo(get_fp8_e4m3_dtype()).max))
    )
    keys_fp8 = (keys / key_scales).to(get_fp8_e4m3_dtype()).contiguous()
    fused_cache = torch.zeros(
        total_pages, 1, 1, HEAD_DIM + 4, dtype=torch.uint8, device=device
    )
    fused_cache[..., :HEAD_DIM] = keys_fp8.view(torch.uint8).reshape(
        total_pages, 1, 1, HEAD_DIM
    )
    fused_cache[..., HEAD_DIM:] = (
        key_scales.reshape(total_pages, 1)
        .view(torch.uint8)
        .reshape(total_pages, 1, 1, 4)
    )
    weights = torch.randn(
        batch_size * next_n, num_heads, device=device, dtype=torch.float32
    ).contiguous()
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        SENTINEL,
        device=device,
        dtype=torch.float32,
    )
    return {
        "query_fp8": query_fp8,
        "keys_fp8": keys_fp8,
        "key_scales": key_scales,
        "fused_cache": fused_cache,
        "weights": weights,
        "context_lengths": context_tensor,
        "page_map": page_map,
        "logits": logits,
    }


def _run_kernel(case, max_model_len):
    deepgemm_fp8_paged_mqa_logits(
        case["query_fp8"],
        case["fused_cache"],
        case["weights"],
        case["logits"],
        case["context_lengths"],
        case["page_map"],
        max_model_len,
        Preshuffle=False,
        KVBlockSize=1,
        ChunkK=128,
    )
    torch.cuda.synchronize()


def _assert_reference(case, batch_size, next_n, num_heads, max_model_len):
    query_float = case["query_fp8"].float()
    keys_float = (
        case["keys_fp8"].reshape(-1, HEAD_DIM).float()
        * case["key_scales"].reshape(-1, 1)
    )
    maximum_error = 0.0
    for batch_index in range(batch_size):
        context_length = int(case["context_lengths"][batch_index].item())
        for next_index in range(next_n):
            row_index = batch_index * next_n + next_index
            valid_length = context_length - next_n + next_index
            pages = case["page_map"][batch_index, : valid_length + 1].long()
            keys = keys_float[pages]
            dots = torch.einsum(
                "hd,td->ht", query_float[batch_index, next_index], keys
            )
            expected = (
                torch.relu(dots) * case["weights"][row_index][:, None]
            ).sum(dim=0)
            actual = case["logits"][row_index, : valid_length + 1].float()
            maximum_error = max(
                maximum_error, float((actual - expected).abs().max().item())
            )
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            assert torch.all(
                case["logits"][row_index, valid_length + 1 : context_length]
                == float("-inf")
            )
            assert torch.all(
                case["logits"][row_index, context_length:] == SENTINEL
            )
            assert torch.all(case["page_map"][batch_index, context_length:] == -1)
    return maximum_error


@unittest.skipUnless(_RUNNABLE, "requires gfx942 with AITER paged MQA logits")
class TestAiterPagedMqaLogits(CustomTestCase):
    def test_small_valid_cache(self):
        batch_size, next_n, num_heads = 1, 1, 32
        context_lengths = [64]
        max_model_len = 256
        case = _make_case(
            batch_size, next_n, num_heads, context_lengths, max_model_len, 123
        )
        _run_kernel(case, max_model_len)
        error = _assert_reference(
            case, batch_size, next_n, num_heads, max_model_len
        )
        self.assertLess(error, 2e-2)

    def test_bounded_page_map_matrix(self):
        batch_size, next_n, num_heads = 4, 2, 32
        context_lengths = [65, 129, 193, 257]
        max_model_len = 512
        case = _make_case(
            batch_size, next_n, num_heads, context_lengths, max_model_len, 321
        )
        _run_kernel(case, max_model_len)
        error = _assert_reference(
            case, batch_size, next_n, num_heads, max_model_len
        )
        self.assertLess(error, 2e-2)


if __name__ == "__main__":
    unittest.main()
