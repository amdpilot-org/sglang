"""Bounded GPU checks for ragged target-verify attention shapes."""

import unittest

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


class TestRaggedVerifyExtendAttention(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(36481)

    def _build_fixture(self, layout):
        head_num = 4
        kv_head_num = 2
        head_dim = 32
        verify_lens = layout.verify_lens.cpu().tolist()
        prefix_lens = [7 + 2 * request for request in range(layout.bs)]
        sequence_lens = [
            prefix + verify for prefix, verify in zip(prefix_lens, verify_lens)
        ]
        request_starts = [0]
        for sequence_len in sequence_lens[:-1]:
            request_starts.append(request_starts[-1] + sequence_len)

        total_tokens = sum(sequence_lens)
        k_buffer = torch.randn(
            total_tokens, kv_head_num, head_dim, dtype=torch.float32, device=self.device
        )
        v_buffer = torch.randn(
            total_tokens, kv_head_num, head_dim, dtype=torch.float32, device=self.device
        )
        q = torch.randn(
            layout.graph_num_tokens,
            head_num,
            head_dim,
            dtype=torch.float32,
            device=self.device,
        )
        k_extend = torch.empty_like(k_buffer[: layout.graph_num_tokens])
        v_extend = torch.empty_like(v_buffer[: layout.graph_num_tokens])
        kv_indices = []
        for request, (prefix_len, verify_len, request_start) in enumerate(
            zip(prefix_lens, verify_lens, request_starts)
        ):
            kv_indices.extend(range(request_start, request_start + prefix_len))
            query_start = int(layout.qo_indptr_device[request])
            query_end = int(layout.qo_indptr_device[request + 1])
            buffer_start = request_start + prefix_len
            k_extend[query_start:query_end] = k_buffer[
                buffer_start : buffer_start + verify_len
            ]
            v_extend[query_start:query_end] = v_buffer[
                buffer_start : buffer_start + verify_len
            ]

        prefix_tensor = torch.tensor(prefix_lens, dtype=torch.int32, device=self.device)
        kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=self.device),
                torch.cumsum(prefix_tensor, dim=0, dtype=torch.int32),
            ]
        )
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device=self.device)
        return q, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices

    def _reference(self, fixture, layout):
        q, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices = fixture
        reference = torch.empty_like(q)
        head_dim = q.shape[-1]
        scale = head_dim**-0.5
        for request in range(layout.bs):
            query_start = int(layout.qo_indptr_device[request])
            query_end = int(layout.qo_indptr_device[request + 1])
            kv_start = int(kv_indptr[request])
            kv_end = int(kv_indptr[request + 1])
            prefix_len = kv_end - kv_start
            prefix_indices = kv_indices[kv_start:kv_end]
            k_prefix = k_buffer[prefix_indices]
            v_prefix = v_buffer[prefix_indices]
            k_full = torch.cat([k_prefix, k_extend[query_start:query_end]], dim=0)
            v_full = torch.cat([v_prefix, v_extend[query_start:query_end]], dim=0)
            k_full = k_full.repeat_interleave(q.shape[1] // k_full.shape[1], dim=1)
            v_full = v_full.repeat_interleave(q.shape[1] // v_full.shape[1], dim=1)
            for query_offset in range(query_end - query_start):
                visible_len = prefix_len + query_offset + 1
                scores = (
                    torch.einsum(
                        "hd,khd->hk",
                        q[query_start + query_offset],
                        k_full[:visible_len],
                    )
                    * scale
                )
                weights = torch.softmax(scores, dim=-1)
                reference[query_start + query_offset] = torch.einsum(
                    "hk,khd->hd", weights, v_full[:visible_len]
                )
        return reference

    def _run_kernel(self, fixture, layout, output, max_len_extend, extend_seq_lens_cpu):
        q, k_extend, v_extend, k_buffer, v_buffer, kv_indptr, kv_indices = fixture
        extend_attention_fwd(
            q,
            k_extend,
            v_extend,
            output,
            k_buffer,
            v_buffer,
            layout.qo_indptr_device,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=max_len_extend,
            k_scale=1.0,
            v_scale=1.0,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
        )

    def _assert_layout(self, layout):
        fixture = self._build_fixture(layout)
        q = fixture[0]
        verify_lens_cpu = layout.verify_lens.cpu().tolist()
        max_len_extend = max(verify_lens_cpu)
        eager = torch.empty_like(q)
        self._run_kernel(fixture, layout, eager, max_len_extend, verify_lens_cpu)
        reference = self._reference(fixture, layout)
        active_tokens = int(layout.qo_indptr_device[-1])
        torch.testing.assert_close(
            eager[:active_tokens], reference[:active_tokens], rtol=2e-3, atol=2e-3
        )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._run_kernel(
                    fixture, layout, eager, max_len_extend, verify_lens_cpu
                )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run_kernel(fixture, layout, eager, max_len_extend, verify_lens_cpu)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            eager[:active_tokens], reference[:active_tokens], rtol=2e-3, atol=2e-3
        )

        q.add_(0.25)
        graph.replay()
        torch.cuda.synchronize()
        mutated_reference = self._reference(fixture, layout)
        torch.testing.assert_close(
            eager[:active_tokens],
            mutated_reference[:active_tokens],
            rtol=2e-3,
            atol=2e-3,
        )

    def test_compact_and_padded_verify_shapes(self):
        compact_lens = torch.tensor([3, 1, 2, 4], dtype=torch.int32, device=self.device)
        compact = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=compact_lens, graph_num_tokens=16
        )
        padded = compact.padded_to_bucket(padded_bs=6, cap=4)
        self.assertEqual(padded.verify_lens.cpu().tolist(), [3, 1, 2, 4, 3, 3])
        self.assertEqual(
            padded.qo_indptr_device.cpu().tolist(), [0, 3, 4, 6, 10, 13, 16]
        )

        with self.subTest(layout="compact"):
            self._assert_layout(compact)
        with self.subTest(layout="padded"):
            self._assert_layout(padded)

    def test_zero_length_padding_sentinel_bounds(self):
        raw = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([4, 4], dtype=torch.int32, device=self.device),
            graph_num_tokens=8,
        )
        padded = raw.padded_to_bucket(padded_bs=6, cap=4)
        self.assertEqual(padded.verify_lens.cpu().tolist(), [4, 4, 0, 0, 0, 0])
        self.assertEqual(padded.qo_indptr_device.cpu().tolist(), [0, 4, 8, 8, 8, 8, 8])


if __name__ == "__main__":
    unittest.main()
