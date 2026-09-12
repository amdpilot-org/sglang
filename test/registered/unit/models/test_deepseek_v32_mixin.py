from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.models.deepseek_common import v32_mixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _Attention(v32_mixin.DeepseekV32AttentionMixin):
    pass


class _Model(v32_mixin.DeepseekV32ModelMixin):
    pass


def _init_attention(config, *, layer_id=2, is_nextn=False, skip_rope=False):
    attention = _Attention()
    attention.init_dsa_indexer(
        config=config,
        hidden_size=64,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        max_position_embeddings=1024,
        rope_theta=10000,
        rope_scaling={"factor": 2},
        quant_config=None,
        layer_id=layer_id,
        prefix="model.layers.2.self_attn",
        alt_stream=None,
        skip_rope=skip_rope,
        is_nextn=is_nextn,
    )
    return attention


def test_non_dsa_attention_has_no_indexer():
    config = SimpleNamespace()
    with patch.object(v32_mixin, "is_deepseek_dsa", return_value=False):
        attention = _init_attention(config)

    assert attention.use_dsa is False
    assert attention.skip_topk is None
    assert attention.next_skip_topk is None
    assert attention.indexer is None


@pytest.mark.parametrize("is_nextn", [False, True])
def test_skip_topk_layer_only_constructs_indexer_for_nextn(is_nextn):
    config = SimpleNamespace(indexer_rope_interleave=True)
    with (
        patch.object(v32_mixin, "is_deepseek_dsa", return_value=True),
        patch.object(v32_mixin, "dsa_layer_skips_topk", return_value=True),
        patch.object(v32_mixin, "get_dsa_index_kpool", return_value=1),
        patch.object(v32_mixin, "get_dsa_index_n_heads", return_value=4),
        patch.object(v32_mixin, "get_dsa_index_head_dim", return_value=32),
        patch.object(v32_mixin, "get_dsa_index_topk", return_value=16),
        patch.object(v32_mixin, "Indexer") as indexer,
    ):
        attention = _init_attention(config, is_nextn=is_nextn)

    assert attention.skip_topk is True
    assert attention.next_skip_topk is True
    assert indexer.call_count == int(is_nextn)


def test_kpool_indexer_receives_v32_configuration_and_skip_rope():
    config = SimpleNamespace(indexer_rope_interleave=True)
    with (
        patch.object(v32_mixin, "is_deepseek_dsa", return_value=True),
        patch.object(v32_mixin, "dsa_layer_skips_topk", return_value=False),
        patch.object(v32_mixin, "get_dsa_index_kpool", return_value=2),
        patch.object(v32_mixin, "get_dsa_index_n_heads", return_value=4),
        patch.object(v32_mixin, "get_dsa_index_head_dim", return_value=32),
        patch.object(v32_mixin, "get_dsa_index_topk", return_value=2048),
        patch.object(v32_mixin, "IndexerKPool") as indexer_kpool,
    ):
        attention = _init_attention(config, skip_rope=True)

    kwargs = indexer_kpool.call_args.kwargs
    assert attention.indexer is indexer_kpool.return_value
    assert kwargs["index_n_heads"] == 4
    assert kwargs["index_head_dim"] == 32
    assert kwargs["index_topk"] == 2048
    assert kwargs["is_neox_style"] is False
    assert kwargs["skip_rope"] is True
    assert kwargs["prefix"].endswith("self_attn.indexer")


@pytest.mark.parametrize(
    ("use_dsa", "use_mha", "expected"),
    [(False, False, False), (True, True, False), (True, False, True)],
)
def test_dsa_topk_routing_respects_attention_mode(use_dsa, use_mha, expected):
    model = _Model()
    model.use_dsa = use_dsa
    backend = SimpleNamespace(primary=SimpleNamespace(use_mha=use_mha))
    with patch(
        "sglang.srt.model_executor.forward_context.get_attn_backend",
        return_value=backend,
    ):
        assert model._dsa_forward_uses_topk() is expected


def test_pipeline_topk_handoff_is_limited_to_skip_boundary():
    model = _Model()
    model.use_dsa = True
    model.start_layer = 3
    model.end_layer = 7
    model.config = SimpleNamespace(num_hidden_layers=8)

    with patch.object(
        v32_mixin, "dsa_layer_skips_topk", side_effect=lambda _, layer: layer == 3
    ):
        assert model.dsa_stage_requires_input_topk(True)
        assert not model.dsa_stage_must_forward_topk(True)

    with patch.object(
        v32_mixin, "dsa_layer_skips_topk", side_effect=lambda _, layer: layer == 7
    ):
        assert not model.dsa_stage_requires_input_topk(True)
        assert model.dsa_stage_must_forward_topk(True)

    model.config.num_hidden_layers = 7
    with patch.object(v32_mixin, "dsa_layer_skips_topk", return_value=True):
        assert not model.dsa_stage_must_forward_topk(True)


def test_empty_pipeline_topk_preserves_device_and_shape():
    model = _Model()
    model.config = SimpleNamespace()
    hidden_states = torch.zeros((2, 4))
    with patch.object(v32_mixin, "get_dsa_index_topk", return_value=17):
        topk = model.empty_dsa_topk(hidden_states)

    assert topk.shape == (0, 17)
    assert topk.dtype == torch.int32
    assert topk.device == hidden_states.device
