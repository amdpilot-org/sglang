from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)


class _DecodeMode:
    def is_extend_without_speculative(self):
        return False

    def is_decode_or_idle(self):
        return True

    def is_target_verify(self):
        return False

    def is_draft_extend_v2(self):
        return False


def _run_breakable_decode(metadata):
    backend = SimpleNamespace(get_indexer_metadata=lambda _layer_id, _fb: metadata)
    indexer = dsa_indexer.Indexer.__new__(dsa_indexer.Indexer)
    indexer.alt_stream = None
    indexer.use_dsa_indexer_fusion = True
    indexer.dsa_enable_prefill_cp = False
    indexer.index_topk = 1
    indexer.n_heads = 1
    indexer.softmax_scale = 1.0
    indexer.block_size = 128
    indexer.scale_fmt = None
    indexer._get_q_k_bf16 = lambda *args, **kwargs: (
        torch.ones((1, 1, 1), device="cuda"),
        torch.ones((1, 1, 1), device="cuda"),
        torch.ones((1, 1), device="cuda"),
    )
    indexer._get_topk_paged = (
        lambda _fb, _layer_id, _q, _weights, resolved: resolved.get_page_table_64()
    )
    forward_batch = SimpleNamespace(
        forward_mode=_DecodeMode(),
        seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
        attn_cp_metadata=None,
    )
    x = torch.ones((1, 1), device="cuda")

    with (
        patch.object(dsa_indexer, "get_attn_backend", return_value=backend),
        patch.object(dsa_indexer, "get_is_capture_mode", return_value=False),
        patch.object(
            dsa_indexer, "_broadcast_indexer_topk_from_rank0", side_effect=lambda x: x
        ),
        patch.object(
            dsa_indexer,
            "maybe_capture_indexer_topk",
            side_effect=lambda _layer_id, result: result,
        ),
        patch(
            "sglang.kernels.ops.attention.dsa.tilelang_kernel.act_quant",
            side_effect=lambda q, *_args: (
                q,
                torch.ones((1, 1), device="cuda"),
            ),
        ),
        enable_breakable_cuda_graph(),
    ):
        return indexer.forward_cuda(
            x,
            x,
            torch.zeros(1, device="cuda", dtype=torch.int64),
            forward_batch,
            0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_breakable_decode_resolves_dsa_metadata():
    expected = torch.tensor([[7]], device="cuda", dtype=torch.int32)
    metadata = SimpleNamespace(get_page_table_64=lambda: expected)

    actual = _run_breakable_decode(metadata)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_breakable_decode_preserves_backend_without_dsa_metadata():
    assert _run_breakable_decode(None) is None
