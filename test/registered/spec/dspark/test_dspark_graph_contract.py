"""Bounded GPU checks for DSpark compact-verify graph replay."""

import unittest

import torch

from sglang.kernels.ops.speculative.dspark import dspark_verify_window
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestDsparkGraphContract(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        torch.manual_seed(30734)

    def _capture(self, operation):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                operation()
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = operation()
        torch.cuda.synchronize()
        return graph, captured

    def _assert_compact_contract(self, layout, draft_block_ids, draft_tokens):
        eager = dspark_verify_window.CompactVerifyIds.triton(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=self.device,
        )
        reference = dspark_verify_window.CompactVerifyIds.torch(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=self.device,
        )
        request_ids, offsets, active = dspark_verify_window.CompactRowIndex.triton(
            verify_lens=layout.verify_lens,
            padded_total=layout.graph_num_tokens,
            device=self.device,
        )

        verify_lens = layout.verify_lens.tolist()
        expected_requests = torch.cat(
            [
                torch.full(
                    (length,), request_id, dtype=torch.int64, device=self.device
                )
                for request_id, length in enumerate(verify_lens)
            ]
        )
        expected_offsets = torch.cat(
            [
                torch.arange(length, device=self.device, dtype=torch.int64)
                for length in verify_lens
            ]
        )
        total_verify_tokens = sum(verify_lens)
        if total_verify_tokens < layout.graph_num_tokens:
            expected_requests = torch.cat(
                [
                    expected_requests,
                    torch.full(
                        (layout.graph_num_tokens - total_verify_tokens,),
                        layout.bs,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                ]
            )
            expected_offsets = torch.cat(
                [
                    expected_offsets,
                    torch.zeros(
                        layout.graph_num_tokens - total_verify_tokens,
                        dtype=torch.int64,
                        device=self.device,
                    ),
                ]
            )
        expected_active = torch.arange(
            layout.graph_num_tokens, device=self.device
        ) < int(layout.verify_lens.sum().item())

        self.assertTrue(torch.equal(eager, reference))
        self.assertTrue(torch.equal(request_ids, expected_requests))
        self.assertTrue(torch.equal(offsets, expected_offsets))
        self.assertTrue(torch.equal(active, expected_active))
        self.assertTrue(torch.equal(eager[~active], torch.zeros_like(eager[~active])))

        graph, replay = self._capture(
            lambda: dspark_verify_window.CompactVerifyIds.triton(
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                layout=layout,
                device=self.device,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(replay, eager))

        draft_block_ids.add_(17)
        draft_tokens.add_(23)
        graph.replay()
        torch.cuda.synchronize()
        mutated_reference = dspark_verify_window.CompactVerifyIds.torch(
            draft_block_ids=draft_block_ids,
            draft_tokens=draft_tokens,
            layout=layout,
            device=self.device,
        )
        self.assertTrue(torch.equal(replay, mutated_reference))

    def _assert_commit_contract(
        self,
        req_pool_indices,
        req_to_token,
        prefix_lens,
        block_pos_offsets,
        full_to_swa_mapping,
        commit_lens,
        stride,
    ):
        kwargs = {
            "req_pool_indices": req_pool_indices,
            "req_to_token": req_to_token,
            "prefix_lens": prefix_lens,
            "block_pos_offsets": block_pos_offsets,
            "full_to_swa_mapping": full_to_swa_mapping,
            "commit_lens": commit_lens,
            "stride": stride,
        }
        eager = dspark_verify_window.BuildCommitInjectLayout.triton(**kwargs)
        reference = dspark_verify_window.BuildCommitInjectLayout.torch(**kwargs)
        self.assertTrue(torch.equal(eager.swa_loc, reference.swa_loc))
        self.assertTrue(torch.equal(eager.positions, reference.positions))

        active = (
            torch.arange(stride, device=self.device).view(1, -1)
            < commit_lens.to(torch.int64).view(-1, 1)
        )
        self.assertTrue(
            torch.equal(
                eager.swa_loc.view(-1, stride)[~active],
                torch.full_like(eager.swa_loc.view(-1, stride)[~active], -1),
            )
        )

        graph, replay = self._capture(
            lambda: dspark_verify_window.BuildCommitInjectLayout.triton(**kwargs)
        )
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(replay.swa_loc, eager.swa_loc))
        self.assertTrue(torch.equal(replay.positions, eager.positions))

        commit_lens.copy_((commit_lens.to(torch.int64) + 1).clamp(max=stride))
        graph.replay()
        torch.cuda.synchronize()
        mutated_reference = dspark_verify_window.BuildCommitInjectLayout.torch(**kwargs)
        self.assertTrue(torch.equal(replay.swa_loc, mutated_reference.swa_loc))
        self.assertTrue(torch.equal(replay.positions, mutated_reference.positions))

    @unittest.skipUnless(
        hasattr(torch.cuda, "CUDAGraph"), "torch.cuda.CUDAGraph is unavailable"
    )
    def test_compact_and_padded_graph_replay(self):
        graph_num_tokens = 16
        gamma = 5
        compact_lens = torch.tensor([3, 1, 2, 4], dtype=torch.int32, device=self.device)
        compact_layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=compact_lens, graph_num_tokens=graph_num_tokens
        )
        padded_layout = compact_layout.padded_to_bucket(padded_bs=6, cap=gamma)

        compact_block_ids = torch.arange(
            4 * gamma, device=self.device, dtype=torch.int64
        ).reshape(4, gamma)
        compact_draft_tokens = compact_block_ids + 1000
        padded_block_ids = torch.arange(
            6 * gamma, device=self.device, dtype=torch.int64
        ).reshape(6, gamma)
        padded_draft_tokens = padded_block_ids + 2000

        with self.subTest(layout="compact"):
            self._assert_compact_contract(
                compact_layout, compact_block_ids, compact_draft_tokens
            )
        with self.subTest(layout="padded"):
            self._assert_compact_contract(
                padded_layout, padded_block_ids, padded_draft_tokens
            )

        stride = 5
        block_pos_offsets = torch.arange(stride, device=self.device, dtype=torch.int64)
        full_to_swa_mapping = torch.arange(256, device=self.device, dtype=torch.int64)
        for batch_size in (4, 6):
            req_pool_indices = torch.arange(
                batch_size, device=self.device, dtype=torch.int64
            )
            req_to_token = torch.arange(
                batch_size * 32, device=self.device, dtype=torch.int64
            ).reshape(batch_size, 32)
            prefix_lens = torch.arange(
                2, 2 + batch_size, device=self.device, dtype=torch.int64
            )
            commit_lens = torch.tensor(
                [2, 0, 5, 3, 1, 4][:batch_size],
                dtype=torch.int32,
                device=self.device,
            )
            with self.subTest(commit_batch_size=batch_size):
                self._assert_commit_contract(
                    req_pool_indices,
                    req_to_token,
                    prefix_lens,
                    block_pos_offsets,
                    full_to_swa_mapping,
                    commit_lens,
                    stride,
                )


if __name__ == "__main__":
    unittest.main()
