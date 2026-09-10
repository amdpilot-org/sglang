import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import minimax_sparse_backend as backend_module
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _make_runner():
    sparse_config = {
        "sparse_index_dim": 16,
        "sparse_attention_freq": [0, 1],
        "sparse_disable_index_value": [0, 1],
        "sparse_score_type": "max",
        "sparse_block_size": 16,
        "sparse_init_block": 1,
        "sparse_local_block": 2,
        "sparse_topk_blocks": 4,
        "sparse_num_index_heads": 1,
    }
    kv_pool = MiniMaxSparseKVPool(
        size=8,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=32,
        idx_head_dim=16,
        dense_layer_ids=[0],
        sparse_layer_ids=[1],
        disable_value_sparse_layer_ids=[1],
        device="cpu",
        start_layer=0,
        end_layer=2,
    )
    req_pool = ReqToTokenPool(
        size=1,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
    )
    return SimpleNamespace(
        token_to_kv_pool=kv_pool,
        req_to_token_pool=req_pool,
        model_config=SimpleNamespace(
            context_len=16,
            hf_config=SimpleNamespace(sparse_attention_config=sparse_config),
            num_attention_heads=2,
        ),
        server_args=SimpleNamespace(
            kv_cache_dtype="auto",
            attention_backend="trtllm_mha",
        ),
    )


class TestMiniMaxSparseSpecDecodeBoundary(unittest.TestCase):
    def test_gpu_speculative_decode_fails_early(self):
        spec = SimpleNamespace(
            speculative_algorithm="EAGLE3",
            speculative_num_draft_tokens=4,
        )
        with mock.patch.object(backend_module, "get_spec", return_value=spec):
            with self.assertRaisesRegex(
                NotImplementedError,
                "MiniMax-M3 sparse attention on GPU supports ordinary "
                "decode and extend",
            ):
                backend_module.MiniMaxSparseAttnBackend(_make_runner())

    def test_ordinary_backend_construction_remains_supported(self):
        spec = SimpleNamespace(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
        )
        with mock.patch.object(backend_module, "get_spec", return_value=spec):
            backend = backend_module.MiniMaxSparseAttnBackend(_make_runner())
        self.assertIsNone(backend.speculative_num_draft_tokens)


if __name__ == "__main__":
    unittest.main()
