import unittest

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small-amd")


@unittest.skipIf(not torch.cuda.is_available(), "GPU is required")
class TestDSACudaGraphContextBoundary(unittest.TestCase):
    def _make_backend(self, *, context_len: int, row_width: int, draft_tokens: int):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.device = torch.device("cuda")
        backend.real_page_size = 64
        backend.speculative_num_draft_tokens = draft_tokens
        backend.max_context_len = context_len
        backend.req_to_token = torch.arange(
            row_width, dtype=torch.int32, device=backend.device
        ).view(1, row_width)
        backend.dsa_decode_impl = "triton"
        backend.init_cuda_graph_state(max_bs=1, max_num_tokens=max(draft_tokens, 1))
        return backend

    def test_target_verify_page_table_matches_req_to_token_row_width(self):
        # v0.5.14 allocated context_len + draft_tokens columns.  The reported
        # EAGLE request has two additional req_to_token reserve columns, making
        # the replay copy 614408 -> 614406 fail at the context boundary.
        backend = self._make_backend(
            context_len=614400, row_width=614408, draft_tokens=6
        )
        metadata = type(
            "Metadata",
            (),
            {"page_table_1": backend.decode_cuda_graph_metadata["page_table"]},
        )()

        width = backend._graph_page_table_width(metadata)
        page_indices = torch.repeat_interleave(
            backend.req_to_token[:, :width], repeats=6, dim=0
        )
        metadata.page_table_1[:, :width].copy_(page_indices)

        self.assertEqual(width, 614408)
        self.assertEqual(tuple(metadata.page_table_1.shape), (6, 614408))
        self.assertTrue(torch.equal(metadata.page_table_1, page_indices))

    def test_page_table_width_boundary_cases(self):
        for context_len, row_width, draft_tokens in (
            (64, 64, 0),  # no speculative reserve
            (64, 70, 6),  # reserve exactly equals the legacy allocation
            (64, 72, 6),  # reserve extends beyond context_len + draft_tokens
        ):
            with self.subTest(
                context_len=context_len,
                row_width=row_width,
                draft_tokens=draft_tokens,
            ):
                backend = self._make_backend(
                    context_len=context_len,
                    row_width=row_width,
                    draft_tokens=draft_tokens,
                )
                page_table = backend.decode_cuda_graph_metadata["page_table"]
                self.assertEqual(page_table.shape[1], row_width)


if __name__ == "__main__":
    unittest.main()
