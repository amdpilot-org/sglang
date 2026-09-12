import unittest
from dataclasses import replace

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DeepseekSparseAttnMultiStepBackend,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (
    DSAMockModelRunner,
    TinyDSAModelConfig,
    make_dsa_dense_fallback_cases,
)
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small")


@unittest.skipUnless(is_hip(), "ROCm-only DSA FP8 regression")
class TestEagleDsaKVIndexTranslator(CustomTestCase):
    def setUp(self):
        super().setUp()
        # The legacy ROCm DSA pool used by this focused fixture requires
        # page_size=1. The production failure is independent of page size: it
        # happens before the pool read, while translating its prefix indices.
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
        self.runner = DSAMockModelRunner(
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

    def tearDown(self):
        self.runner._server_args_override.restore()
        super().tearDown()

    def test_late_created_draft_extend_backend_uses_runner_translator(self):
        backend = DeepseekSparseAttnBackend(self.runner)

        self.assertIs(backend.kv_index_translator, self.runner.kv_index_translator)
        prefix_ids = torch.tensor([1, 7, 31], device="cuda", dtype=torch.int64)
        translated = backend.kv_index_translator.translate_dcp_read_ids(prefix_ids)
        torch.testing.assert_close(translated, prefix_ids)

    def test_late_created_multistep_children_use_runner_translator(self):
        backend = DeepseekSparseAttnMultiStepBackend(
            self.runner, topk=1, speculative_num_steps=3
        )

        self.assertEqual(len(backend.attn_backends), 2)
        for child in backend.attn_backends:
            self.assertIs(
                child.kv_index_translator, self.runner.kv_index_translator
            )


if __name__ == "__main__":
    unittest.main()
