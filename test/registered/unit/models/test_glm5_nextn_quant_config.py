from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.srt.models.deepseek_nextn import (
    DeepseekModelNextN,
    DeepseekV3ForCausalLMNextN,
)
from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)


class _ModelOptFp4Config:
    @staticmethod
    def get_name():
        return "modelopt_fp4"


def _config(ignore=None):
    return SimpleNamespace(
        num_hidden_layers=45,
        quantization_config={"ignore": ignore or []},
    )


def test_glm5_builds_quantized_nextn_layer_for_modelopt_fp4_checkpoint():
    quant_config = _ModelOptFp4Config()
    model = object.__new__(Glm5NextForConditionalGenerationNextN)

    resolved = model._resolve_nextn_quant_config(_config(), quant_config)

    config = SimpleNamespace(
        vocab_size=128,
        hidden_size=64,
        rms_norm_eps=1e-5,
        qk_rope_head_dim=0,
    )

    class _Decoder(nn.Module):
        def __init__(self, quant_config):
            super().__init__()
            self.received_quant_config = quant_config

    with (
        patch(
            "sglang.srt.models.deepseek_nextn.VocabParallelEmbedding",
            side_effect=lambda *args, **kwargs: nn.Identity(),
        ),
        patch(
            "sglang.srt.models.deepseek_nextn.RMSNorm",
            side_effect=lambda *args, **kwargs: nn.Identity(),
        ),
        patch(
            "sglang.srt.models.deepseek_nextn.DeepseekV2DecoderLayer",
            side_effect=lambda *args, **kwargs: _Decoder(kwargs["quant_config"]),
        ),
        patch(
            "sglang.srt.models.deepseek_nextn.get_embedding_tp_kwargs",
            return_value={},
        ),
        patch("sglang.srt.models.deepseek_nextn._is_cuda", False),
        patch("sglang.srt.models.deepseek_nextn._is_npu", False),
    ):
        nextn = DeepseekModelNextN(config, resolved)

    assert nextn.decoder.received_quant_config is quant_config


def test_glm5_drops_modelopt_fp4_for_checkpoint_declared_bf16_nextn_layer():
    quant_config = _ModelOptFp4Config()
    config = _config(ignore=["model.layers.45.*"])
    model = object.__new__(Glm5NextForConditionalGenerationNextN)

    resolved = model._resolve_nextn_quant_config(config, quant_config)

    assert resolved is None


def test_deepseek_retains_legacy_unquantized_modelopt_fp4_nextn_behavior():
    quant_config = _ModelOptFp4Config()
    model = object.__new__(DeepseekV3ForCausalLMNextN)

    resolved = model._resolve_nextn_quant_config(_config(), quant_config)

    assert resolved is None
