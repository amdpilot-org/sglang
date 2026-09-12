from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import dsa_backend as backend_module
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_first_target_verify_graph_init_launches_metadata_kernel():
    """Buffer creation must not defer the Triton module load to first replay."""
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.decode_cuda_graph_metadata = {}
    backend.dsa_index_kpool = 1
    backend.dsa_index_topk = 4
    backend.speculative_num_draft_tokens = 2
    backend.real_page_size = 1
    backend.dsa_decode_impl = "trtllm"
    backend.req_to_token = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    backend.set_dsa_prefill_impl = Mock()
    backend._update_kpool_metadata_replay = Mock()

    metadata = SimpleNamespace(
        cache_seqlens_int32=torch.empty(1, dtype=torch.int32),
        cu_seqlens_k=torch.empty(2, dtype=torch.int32),
        page_table_1=torch.empty((2, 16), dtype=torch.int32),
        dsa_seqlens_expanded=torch.empty(2, dtype=torch.int32),
        dsa_cache_seqlens_int32=torch.empty(2, dtype=torch.int32),
        dsa_cu_seqlens_k=torch.empty(3, dtype=torch.int32),
        real_page_table=None,
        paged_mqa_ctx_lens_2d=None,
        page_size=1,
    )
    metadata.real_page_table = metadata.page_table_1

    def build(*args, **kwargs):
        backend.decode_cuda_graph_metadata[1] = metadata

    backend._build_forward_metadata_cuda_graph = Mock(side_effect=build)

    with (
        patch.object(backend_module, "_is_hip", True),
        patch.object(backend_module, "is_cuda", return_value=False),
        patch.object(backend_module, "fused_dsa_target_verify_metadata") as fused,
    ):
        backend._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=None,
        )

    backend._build_forward_metadata_cuda_graph.assert_called_once()
    fused.assert_called_once()
    assert backend.forward_metadata is metadata


def test_existing_graph_metadata_does_not_rebuild_buffers():
    """The warmup fix must preserve the established replay path."""
    backend = object.__new__(DeepseekSparseAttnBackend)
    backend.decode_cuda_graph_metadata = {1: object()}
    backend._build_forward_metadata_cuda_graph = Mock()
    backend.set_dsa_prefill_impl = Mock(side_effect=RuntimeError("replay reached"))

    try:
        backend._apply_cuda_graph_metadata(
            bs=1,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([3]),
            seq_lens_cpu=torch.tensor([3]),
            forward_mode=ForwardMode.TARGET_VERIFY,
            spec_info=None,
        )
    except RuntimeError as exc:
        assert str(exc) == "replay reached"
    else:
        raise AssertionError("existing metadata did not enter the replay path")

    backend._build_forward_metadata_cuda_graph.assert_not_called()
