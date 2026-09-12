from dataclasses import replace

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
)
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSAMockModelRunner,
    TinyDSAModelConfig,
    make_dsa_dense_fallback_cases,
)


class RecordingTranslator:
    def __init__(self):
        self.calls = []

    def translate_dcp_read_ids(self, ids):
        self.calls.append(ids.clone())
        return ids + 1000


case = replace(make_dsa_dense_fallback_cases("dsa")[0], page_size=1)
config = TinyDSAModelConfig(
    num_heads=4,
    head_dim=576,
    hidden_size=2304,
    context_len=256,
    num_kv_heads=1,
    qk_nope_head_dim=512,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
)
runner = DSAMockModelRunner(
    case=case,
    model_config=config,
    dtype=torch.bfloat16,
    device="cuda",
    max_context_len=256,
    head_dim=576,
    dsa_prefill_backend="tilelang",
    dsa_decode_backend="tilelang",
    fp8_kv_cache=True,
)
try:
    translator = RecordingTranslator()
    runner.kv_index_translator = translator
    extend = DeepseekSparseAttnBackend(runner)
    decode = DeepseekSparseAttnMultiStepBackend(
        runner, topk=1, speculative_num_steps=4
    )
    assert extend.kv_index_translator is translator
    assert len(decode.attn_backends) == 3
    assert all(child.kv_index_translator is translator for child in decode.attn_backends)
    prefix = torch.tensor([0, 5, 63], device="cuda", dtype=torch.int64)
    translated = extend.kv_index_translator.translate_dcp_read_ids(prefix)
    torch.testing.assert_close(translated, prefix + 1000)
    assert len(translator.calls) == 1
    print("PASS: non-identity translator preserved across extend and 3 decode children")
finally:
    runner._server_args_override.restore()
