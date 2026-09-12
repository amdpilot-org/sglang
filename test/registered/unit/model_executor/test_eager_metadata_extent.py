import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _preplanned_batch(mode: ForwardMode) -> ForwardBatch:
    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=3,
        input_ids=torch.arange(3),
        positions=torch.arange(3),
        req_pool_indices=torch.arange(3),
        seq_lens=torch.ones(3, dtype=torch.int32),
        out_cache_loc=torch.arange(3),
        seq_lens_sum=3,
    )
    batch.mark_forward_metadata_ready()
    # Reproduce eager attention-TP padding after the non-replannable plan.
    batch.batch_size = 4
    batch.input_ids = torch.arange(4)
    batch.positions = torch.arange(4)
    batch.req_pool_indices = torch.arange(4)
    batch.seq_lens = torch.ones(4, dtype=torch.int32)
    batch.out_cache_loc = torch.arange(4)
    return batch


class TestEagerMetadataExtent(CustomTestCase):
    def _runner(self, backend: AttentionBackend) -> EagerRunner:
        runner = EagerRunner.__new__(EagerRunner)
        runner.enable_pdmux = True
        runner.model_runner = SimpleNamespace(
            attn_backend=backend,
            decode_attn_backend=backend,
            model=SimpleNamespace(forward=Mock()),
            ps=SimpleNamespace(attn_dcp_size=1),
            device="cpu",
            device_timer=None,
            _pp_kwargs=lambda _: {},
            _extend_forward_kwargs=lambda *_: {},
        )
        return runner

    def test_decode_rejects_stale_default_backend_before_model(self):
        backend = AttentionBackend()
        runner = self._runner(backend)
        runner._resolve_decode_pdmux = lambda: (backend, contextlib.nullcontext())

        with self.assertRaisesRegex(RuntimeError, "planned batch_size=3"):
            runner._execute_decode(_preplanned_batch(ForwardMode.DECODE))
        runner.model_runner.model.forward.assert_not_called()

    @patch(
        "sglang.srt.model_executor.runner.eager_runner.is_cp_active",
        return_value=False,
    )
    def test_extend_rejects_stale_default_backend_before_model(self, _):
        runner = self._runner(AttentionBackend())

        with self.assertRaisesRegex(RuntimeError, "physical batch_size=4"):
            runner._execute_extend(_preplanned_batch(ForwardMode.DRAFT_EXTEND_V2))
        runner.model_runner.model.forward.assert_not_called()

    def test_dsa_accepts_actual_draft_v2_optional_indexer_metadata(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.ones(4, dtype=torch.int32),
            dsa_cu_seqlens_k=torch.arange(5, dtype=torch.int32),
            dsa_cu_seqlens_q=torch.arange(5, dtype=torch.int32),
            dsa_seqlens_expanded=torch.ones(4, dtype=torch.int32),
            token_to_batch_idx=None,
            indexer_k_start_end=None,
            topk_indices_offset=None,
            paged_mqa_ctx_lens_2d=torch.ones((4, 1), dtype=torch.int32),
        )
        batch = _preplanned_batch(ForwardMode.DRAFT_EXTEND_V2)
        backend.validate_preplanned_metadata_extent(batch)

        backend.forward_metadata.dsa_cache_seqlens_int32 = torch.ones(
            3, dtype=torch.int32
        )
        with self.assertRaisesRegex(RuntimeError, "dsa_rows=3"):
            backend.validate_preplanned_metadata_extent(batch)

    def test_dsa_rejects_stale_downstream_token_metadata(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.ones(4, dtype=torch.int32),
            dsa_cu_seqlens_k=torch.arange(5, dtype=torch.int32),
            dsa_cu_seqlens_q=torch.arange(5, dtype=torch.int32),
            dsa_seqlens_expanded=torch.ones(3, dtype=torch.int32),
            token_to_batch_idx=torch.arange(3, dtype=torch.int32),
            indexer_k_start_end=(
                torch.arange(3, dtype=torch.int32),
                torch.arange(3, dtype=torch.int32),
            ),
            topk_indices_offset=None,
            paged_mqa_ctx_lens_2d=torch.ones((3, 1), dtype=torch.int32),
        )

        with self.assertRaisesRegex(RuntimeError, "expanded_rows=3"):
            backend.validate_preplanned_metadata_extent(
                _preplanned_batch(ForwardMode.DRAFT_EXTEND_V2)
            )


if __name__ == "__main__":
    unittest.main()
