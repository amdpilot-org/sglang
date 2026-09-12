from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.srt.models.deepseek_nextn import DeepseekModelNextN
from sglang.srt.models.glm5_next_nextn import Glm5NextForConditionalGenerationNextN


class Quant:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class DictLike:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return self.value


def config(qconfig):
    return SimpleNamespace(num_hidden_layers=45, quantization_config=qconfig)


glm = object.__new__(Glm5NextForConditionalGenerationNextN)
fp4 = Quant("modelopt_fp4")
assert glm._resolve_nextn_quant_config(config({}), fp4) is fp4
assert glm._resolve_nextn_quant_config(config({"ignore": ["unrelated.*"]}), fp4) is fp4
assert glm._resolve_nextn_quant_config(
    config(DictLike({"ignore": ["model.layers.45.*"]})), fp4
) is None
assert glm._resolve_nextn_quant_config(config({}), None) is None

# A non-ModelOpt configuration must still follow the inherited resolver.
other = Quant("fp8")
assert glm._resolve_nextn_quant_config(config({}), other) is other

# Exercise the actual constructor boundary independently from the submitted test.
decoder_seen = []


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        decoder_seen.append(kwargs["quant_config"])


minimal = SimpleNamespace(
    vocab_size=16, hidden_size=16, rms_norm_eps=1e-5, qk_rope_head_dim=0
)
with (
    patch("sglang.srt.models.deepseek_nextn.VocabParallelEmbedding", return_value=nn.Identity()),
    patch("sglang.srt.models.deepseek_nextn.RMSNorm", return_value=nn.Identity()),
    patch("sglang.srt.models.deepseek_nextn.DeepseekV2DecoderLayer", Decoder),
    patch("sglang.srt.models.deepseek_nextn.get_embedding_tp_kwargs", return_value={}),
    patch("sglang.srt.models.deepseek_nextn._is_cuda", False),
    patch("sglang.srt.models.deepseek_nextn._is_npu", False),
):
    DeepseekModelNextN(minimal, fp4)

assert decoder_seen == [fp4], decoder_seen
print("all adversarial resolver and constructor boundaries passed")
