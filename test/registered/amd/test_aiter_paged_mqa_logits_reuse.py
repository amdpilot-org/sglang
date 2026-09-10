"""gfx942 output-storage coverage for AITER FP8 paged MQA logits.

The existing numerical work covers a fresh explicit output buffer.  This test
adds the distinct execution-representation case: reuse one output buffer for
two different input batches while preserving its float32 dtype, contiguous
layout, stable address, and non-aliasing contract.
"""

import unittest

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-a-test-1-gpu-small-amd")


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


BATCH_SIZE = 2
NEXT_N = 1
NUM_HEADS = 32
HEAD_DIM = 128
MAX_MODEL_LEN = 256
CONTEXT_LENGTHS = (130, 193)
GUARD_ELEMENTS = 16
SENTINEL = -12345.0


def _make_case(seed):
    device = "cuda"
    torch.manual_seed(seed)
    context_lengths = torch.tensor(
        CONTEXT_LENGTHS, dtype=torch.int32, device=device
    )
    total_tokens = int(context_lengths.sum())
    page_map = torch.full(
        (BATCH_SIZE, MAX_MODEL_LEN), -1, dtype=torch.int32, device=device
    )
    token_offset = 0
    for batch_index, context_length in enumerate(CONTEXT_LENGTHS):
        page_map[batch_index, :context_length] = torch.arange(
            token_offset, token_offset + context_length, device=device
        )
        token_offset += context_length

    fp8_dtype = get_fp8_e4m3_dtype()
    query = torch.randn(
        BATCH_SIZE,
        NEXT_N,
        NUM_HEADS,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    query_fp8 = query.to(fp8_dtype).contiguous()
    keys = torch.randn(
        total_tokens, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    key_amax = keys.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    key_scales = torch.pow(
        2.0,
        torch.ceil(
            torch.log2(key_amax / torch.finfo(fp8_dtype).max)
        ),
    )
    keys_fp8 = (keys / key_scales).to(fp8_dtype).contiguous()
    fused_cache = torch.zeros(
        total_tokens, 1, 1, HEAD_DIM + 4, dtype=torch.uint8, device=device
    )
    fused_cache[..., :HEAD_DIM] = keys_fp8.view(torch.uint8).reshape(
        total_tokens, 1, 1, HEAD_DIM
    )
    fused_cache[..., HEAD_DIM:] = (
        key_scales.reshape(total_tokens, 1)
        .view(torch.uint8)
        .reshape(total_tokens, 1, 1, 4)
    )
    weights = torch.randn(
        BATCH_SIZE * NEXT_N,
        NUM_HEADS,
        device=device,
        dtype=torch.float32,
    ).contiguous()
    return {
        "query_fp8": query_fp8,
        "keys_fp8": keys_fp8,
        "key_scales": key_scales,
        "fused_cache": fused_cache,
        "weights": weights,
        "context_lengths": context_lengths,
        "page_map": page_map,
    }


def _guarded_output():
    storage = torch.full(
        (
            GUARD_ELEMENTS
            + BATCH_SIZE * NEXT_N * MAX_MODEL_LEN
            + GUARD_ELEMENTS,
        ),
        SENTINEL,
        device="cuda",
        dtype=torch.float32,
    )
    output = storage[
        GUARD_ELEMENTS : GUARD_ELEMENTS + BATCH_SIZE * NEXT_N * MAX_MODEL_LEN
    ].view(BATCH_SIZE * NEXT_N, MAX_MODEL_LEN)
    return storage, output


def _dispatch(case, output):
    expected_shape = (BATCH_SIZE * NEXT_N, MAX_MODEL_LEN)
    if output.dtype != torch.float32:
        raise ValueError(
            f"out_logits dtype must be float32, got {output.dtype}"
        )
    if output.shape != expected_shape:
        raise ValueError(
            f"out_logits shape must be {expected_shape}, got {tuple(output.shape)}"
        )
    if not output.is_contiguous():
        raise ValueError("out_logits must be contiguous")
    if output.device.type != "cuda":
        raise ValueError(f"out_logits must be CUDA, got {output.device}")

    deepgemm_fp8_paged_mqa_logits(
        case["query_fp8"],
        case["fused_cache"],
        case["weights"],
        output,
        case["context_lengths"],
        case["page_map"],
        MAX_MODEL_LEN,
        Preshuffle=False,
        KVBlockSize=1,
        ChunkK=128,
    )
    torch.cuda.synchronize()


def _reference(case):
    query_float = case["query_fp8"].float()
    keys_float = (
        case["keys_fp8"].reshape(-1, HEAD_DIM).float()
        * case["key_scales"].reshape(-1, 1)
    )
    expected = torch.zeros(
        BATCH_SIZE * NEXT_N, MAX_MODEL_LEN, device="cuda", dtype=torch.float32
    )
    for batch_index in range(BATCH_SIZE):
        context_length = CONTEXT_LENGTHS[batch_index]
        pages = case["page_map"][batch_index, :context_length].long()
        keys = keys_float[pages]
        dots = torch.einsum(
            "hd,td->ht", query_float[batch_index, 0], keys
        )
        expected[batch_index, :context_length] = (
            torch.relu(dots) * case["weights"][batch_index][:, None]
        ).sum(dim=0)
    return expected


def _assert_output_contract(case, storage, output):
    assert output.dtype == torch.float32
    assert output.is_contiguous()
    assert torch.equal(
        storage[:GUARD_ELEMENTS],
        torch.full_like(storage[:GUARD_ELEMENTS], SENTINEL),
    )
    assert torch.equal(
        storage[-GUARD_ELEMENTS:],
        torch.full_like(storage[-GUARD_ELEMENTS:], SENTINEL),
    )

    output_start = output.data_ptr()
    output_end = output_start + output.numel() * output.element_size()
    for name in ("query_fp8", "fused_cache", "weights", "page_map"):
        tensor = case[name]
        tensor_start = tensor.data_ptr()
        tensor_end = tensor_start + tensor.numel() * tensor.element_size()
        assert output_end <= tensor_start or tensor_end <= output_start, (
            f"out_logits aliases {name}"
        )


def _assert_reference(case, output):
    expected = _reference(case)
    valid = torch.zeros_like(output, dtype=torch.bool)
    for batch_index, context_length in enumerate(CONTEXT_LENGTHS):
        valid[batch_index, :context_length] = True
    torch.testing.assert_close(
        output[valid], expected[valid], atol=2e-2, rtol=2e-2
    )


@unittest.skipUnless(
    _RUNNABLE, "requires gfx942 with AITER paged MQA logits"
)
class TestAiterPagedMqaLogitsReuse(CustomTestCase):
    def test_fresh_and_reused_output_storage(self):
        first = _make_case(101)
        second = _make_case(202)

        fresh_storage, fresh_output = _guarded_output()
        _dispatch(first, fresh_output)
        _assert_reference(first, fresh_output)
        _assert_output_contract(first, fresh_storage, fresh_output)

        reused_storage, reused_output = _guarded_output()
        stable_address = reused_output.data_ptr()
        _dispatch(first, reused_output)
        first_snapshot = reused_output.clone()
        _dispatch(second, reused_output)

        self.assertEqual(reused_output.data_ptr(), stable_address)
        self.assertTrue(torch.equal(fresh_output, first_snapshot))
        _assert_reference(first, first_snapshot)
        _assert_reference(second, reused_output)
        _assert_output_contract(second, reused_storage, reused_output)

    def test_unsupported_output_variants_fail_clearly(self):
        case = _make_case(303)
        _, output = _guarded_output()
        wrong_dtype = torch.empty_like(output, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "dtype must be float32"):
            _dispatch(case, wrong_dtype)

        wide_storage = torch.empty(
            BATCH_SIZE * NEXT_N, MAX_MODEL_LEN * 2, device="cuda"
        )
        non_contiguous = wide_storage[:, ::2]
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            _dispatch(case, non_contiguous)


if __name__ == "__main__":
    unittest.main()
