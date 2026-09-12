from types import SimpleNamespace
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class TestFP4DequantWorkspaceMetadata(unittest.TestCase):
    def setUp(self):
        self.pool = SimpleNamespace(
            get_raw_kv_buffer=lambda layer_id: (None, None, None, None),
            get_dequant_workspace=lambda: (None, None),
        )

    def call_prepare(self, **overrides):
        metadata = {
            "req_pool_indices_cpu": [0],
            "extend_prefix_lens_cpu": [0],
            "extend_seq_lens_cpu": [0],
        }
        metadata.update(overrides)
        return MHATokenToKVPool._prepare_dequant_extend_workspace(
            self.pool,
            layer_id=0,
            global_layer_id=0,
            req_to_token=torch.empty((1, 0), dtype=torch.int32),
            page_size=64,
            **metadata,
        )

    def test_missing_speculative_cpu_metadata_has_actionable_error(self):
        for field in (
            "req_pool_indices_cpu",
            "extend_prefix_lens_cpu",
            "extend_seq_lens_cpu",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    RuntimeError, "requires CPU request, prefix-length, and extend-length"
                ):
                    self.call_prepare(**{field: None})

    def test_empty_cpu_metadata_is_valid(self):
        result = self.call_prepare(
            req_pool_indices_cpu=[],
            extend_prefix_lens_cpu=[],
            extend_seq_lens_cpu=[],
        )
        self.assertEqual(result, (None, None))


if __name__ == "__main__":
    unittest.main()
