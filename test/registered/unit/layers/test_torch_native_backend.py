from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_sliding_window_mask_is_bidirectional_for_encoder_attention():
    mask = TorchNativeAttnBackend._make_sliding_window_mask(
        q_len=5,
        kv_len=5,
        sliding_window_size=1,
        device=torch.device("cpu"),
        causal=False,
    )

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
            [False, False, False, True, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_sliding_window_mask_preserves_causal_semantics_with_offset():
    mask = TorchNativeAttnBackend._make_sliding_window_mask(
        q_len=2,
        kv_len=5,
        sliding_window_size=2,
        device=torch.device("cpu"),
        query_offset=3,
    )

    expected = torch.tensor(
        [
            [False, True, True, True, False],
            [False, False, True, True, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_bidirectional_window_zero_only_attends_to_same_position():
    mask = TorchNativeAttnBackend._make_sliding_window_mask(
        q_len=4,
        kv_len=4,
        sliding_window_size=0,
        device=torch.device("cpu"),
        causal=False,
    )

    torch.testing.assert_close(mask, torch.eye(4, dtype=torch.bool))


def test_encoder_forward_extend_passes_sliding_window():
    backend = TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)
    backend.swa_out_cache_loc = None
    backend.req_to_token_pool = SimpleNamespace(req_to_token=torch.empty(0))
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=Mock(return_value=torch.empty(0)),
        get_value_buffer=Mock(return_value=torch.empty(0)),
    )
    backend._run_sdpa_forward_extend = Mock()

    layer = SimpleNamespace(
        qk_head_dim=2,
        v_head_dim=2,
        tp_q_head_num=1,
        tp_k_head_num=1,
        is_cross_attention=False,
        attn_type=AttentionType.ENCODER_ONLY,
        scaling=1.0,
        sliding_window_size=128,
        layer_id=0,
    )
    forward_batch = SimpleNamespace(
        out_cache_loc=None,
        req_pool_indices=torch.empty(0, dtype=torch.int64),
        seq_lens=torch.empty(0, dtype=torch.int64),
        extend_prefix_lens=torch.empty(0, dtype=torch.int64),
        extend_seq_lens=torch.empty(0, dtype=torch.int64),
        encoder_lens=None,
    )

    backend.forward_extend(
        torch.empty((1, 2)), None, None, layer, forward_batch, save_kv_cache=False
    )

    assert backend._run_sdpa_forward_extend.call_args.kwargs["causal"] is False
    assert (
        backend._run_sdpa_forward_extend.call_args.kwargs["sliding_window_size"] == 128
    )
