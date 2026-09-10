"""Execution-representation coverage for Triton decode attention.

The decode kernels intentionally accept non-contiguous slot/head strides so a caller
can hand them packed K/V views. The last dimension is different: the kernels do not
receive a last-dim stride and must reject such views instead of reading wrong data.
"""

import unittest

import torch

from sglang.kernels.ops.attention.decode_attention import (
    decode_attention_fwd_grouped,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=4, suite="stage-b-test-1-gpu-small-amd")


BATCH = 2
QUERY_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 64
SEQUENCE_LENGTH = 128
MAX_KV_SPLITS = 4
SENTINEL = -12345.0
GUARD_ELEMENTS = 32


def _reference(q, k_buffer, v_buffer, kv_indptr, kv_indices):
    group_size = q.shape[1] // k_buffer.shape[1]
    output = torch.empty_like(q, dtype=torch.float32)
    scale = 1.0 / q.shape[-1] ** 0.5

    for batch in range(q.shape[0]):
        start = int(kv_indptr[batch])
        end = int(kv_indptr[batch + 1])
        indices = kv_indices[start:end]
        keys = k_buffer[indices].float()
        values = v_buffer[indices].float()
        for query_head in range(q.shape[1]):
            kv_head = query_head // group_size
            scores = q[batch, query_head].float() @ keys[:, kv_head].T
            scores *= scale
            weights = torch.softmax(scores, dim=-1)
            output[batch, query_head] = weights @ values[:, kv_head]

    return output.to(q.dtype)


def _make_case(packed, seed=2271, sequence_length=SEQUENCE_LENGTH):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    total_tokens = BATCH * sequence_length

    q = torch.randn(
        BATCH, QUERY_HEADS, HEAD_DIM, dtype=dtype, device=device, generator=generator
    )
    if packed:
        packed_kv = torch.randn(
            total_tokens,
            KV_HEADS,
            2 * HEAD_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        k_buffer = packed_kv[..., :HEAD_DIM]
        v_buffer = packed_kv[..., HEAD_DIM:]
    else:
        packed_kv = None
        k_buffer = torch.randn(
            total_tokens,
            KV_HEADS,
            HEAD_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        v_buffer = torch.randn(
            total_tokens,
            KV_HEADS,
            HEAD_DIM,
            dtype=dtype,
            device=device,
            generator=generator,
        )

    kv_indptr = torch.tensor(
        [0, sequence_length, 2 * sequence_length], dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)
    num_kv_splits = torch.full((BATCH,), 2, dtype=torch.int32, device=device)
    attn_logits = torch.empty(
        BATCH,
        QUERY_HEADS,
        MAX_KV_SPLITS,
        HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        BATCH, QUERY_HEADS, MAX_KV_SPLITS, dtype=torch.float32, device=device
    )

    return {
        "q": q,
        "k_buffer": k_buffer,
        "v_buffer": v_buffer,
        "packed_kv": packed_kv,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "attn_logits": attn_logits,
        "attn_lse": attn_lse,
    }


def _sentinel_output():
    elements = BATCH * QUERY_HEADS * HEAD_DIM
    storage = torch.full(
        (GUARD_ELEMENTS + elements + GUARD_ELEMENTS,),
        SENTINEL,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[GUARD_ELEMENTS : GUARD_ELEMENTS + elements].view(
        BATCH, QUERY_HEADS, HEAD_DIM
    )
    return storage, output


def _run(case, output):
    decode_attention_fwd_grouped(
        case["q"],
        case["k_buffer"],
        case["v_buffer"],
        output,
        case["kv_indptr"],
        case["kv_indices"],
        case["attn_logits"],
        case["attn_lse"],
        case["num_kv_splits"],
        MAX_KV_SPLITS,
        1.0 / HEAD_DIM**0.5,
        1.0,
    )


class TestTritonDecodeRepresentation(CustomTestCase):
    def test_packed_kv_matches_independent_reference(self):
        packed = _make_case(packed=True)
        unpacked = _make_case(packed=False)
        unpacked["q"].copy_(packed["q"])
        unpacked["k_buffer"].copy_(packed["k_buffer"])
        unpacked["v_buffer"].copy_(packed["v_buffer"])

        self.assertFalse(packed["k_buffer"].is_contiguous())
        self.assertFalse(packed["v_buffer"].is_contiguous())
        self.assertEqual(
            packed["k_buffer"].untyped_storage().data_ptr(),
            packed["v_buffer"].untyped_storage().data_ptr(),
        )

        packed_storage, packed_output = _sentinel_output()
        unpacked_storage, unpacked_output = _sentinel_output()
        _run(packed, packed_output)
        _run(unpacked, unpacked_output)
        torch.cuda.synchronize()

        reference = _reference(
            packed["q"],
            packed["k_buffer"],
            packed["v_buffer"],
            packed["kv_indptr"],
            packed["kv_indices"],
        )
        torch.testing.assert_close(
            packed_output.float(),
            reference.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            unpacked_output.float(),
            reference.float(),
            rtol=1e-2,
            atol=1e-2,
        )
        self.assertTrue(torch.all(packed_storage[:GUARD_ELEMENTS] == SENTINEL))
        self.assertTrue(torch.all(packed_storage[-GUARD_ELEMENTS:] == SENTINEL))
        self.assertTrue(torch.all(unpacked_storage[:GUARD_ELEMENTS] == SENTINEL))
        self.assertTrue(torch.all(unpacked_storage[-GUARD_ELEMENTS:] == SENTINEL))

    def test_packed_kv_graph_replay_keeps_output_address(self):
        case = _make_case(packed=True)
        _, output = _sentinel_output()

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            _run(case, output)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run(case, output)
        output_address = output.data_ptr()
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(output.data_ptr(), output_address)
        reference = _reference(
            case["q"],
            case["k_buffer"],
            case["v_buffer"],
            case["kv_indptr"],
            case["kv_indices"],
        )
        torch.testing.assert_close(
            output.float(), reference.float(), rtol=1e-2, atol=1e-2
        )

    def test_non_contiguous_last_dim_fails_clearly(self):
        case = _make_case(packed=True)
        _, output = _sentinel_output()

        q_storage = torch.empty(
            BATCH, QUERY_HEADS, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        strided_q = q_storage[..., ::2]
        invalid_q = dict(case)
        invalid_q["q"] = strided_q
        with self.assertRaisesRegex(ValueError, "q must have a contiguous last dim"):
            _run(invalid_q, output)

        k_storage = torch.empty(
            case["k_buffer"].shape[0],
            case["k_buffer"].shape[1],
            2 * case["k_buffer"].shape[2],
            dtype=case["k_buffer"].dtype,
            device=case["k_buffer"].device,
        )
        strided_k = k_storage[..., ::2]
        invalid_k = dict(case)
        invalid_k["k_buffer"] = strided_k
        with self.assertRaisesRegex(
            ValueError, "KV buffer must have a contiguous last dim"
        ):
            _run(invalid_k, output)


if __name__ == "__main__":
    unittest.main()
