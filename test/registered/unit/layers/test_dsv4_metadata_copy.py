import unittest

import torch

from sglang.srt.layers.attention.deepseek_v4_backend import (
    DSV4RawDecodeMetadata,
    DSV4RawVerifyMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDSV4MetadataCopy(unittest.TestCase):
    def _check_copy(self, metadata_cls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        req_backing = torch.arange(16, dtype=torch.int32, device=device)
        seq_backing = torch.arange(16, dtype=torch.int32, device=device)
        out_cache_loc = torch.arange(8, dtype=torch.int64, device=device)
        expected_req = req_backing[1:9].clone()
        expected_seq = seq_backing[1:9].clone()
        chosen = metadata_cls(req_backing[:8], seq_backing[:8], out_cache_loc)
        other = metadata_cls(req_backing[1:9], seq_backing[1:9], out_cache_loc)

        chosen.copy_(other)

        torch.testing.assert_close(chosen.req_pool_indices, expected_req)
        torch.testing.assert_close(chosen.seq_lens, expected_seq)
        torch.testing.assert_close(chosen.out_cache_loc, out_cache_loc)

    def test_raw_verify_copy_accepts_overlapping_views(self):
        self._check_copy(DSV4RawVerifyMetadata)

    def test_raw_decode_copy_accepts_overlapping_views(self):
        self._check_copy(DSV4RawDecodeMetadata)


if __name__ == "__main__":
    unittest.main()
