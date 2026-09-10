import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci


register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestMaskedSetKVBufferLayout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        torch.manual_seed(31568)
        cls.num_tokens = 17
        cls.head_num = 4
        cls.head_dim = 32
        cls.pool = MHATokenToKVPool(
            size=64,
            page_size=1,
            dtype=torch.float32,
            head_num=cls.head_num,
            head_dim=cls.head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        cls.layer = SimpleNamespace(layer_id=0)
        cls.locations = torch.arange(
            cls.num_tokens, device="cuda", dtype=torch.int64
        ).flip(0)
        cls.mask = (torch.arange(cls.num_tokens, device="cuda") % 2 == 0).to(
            torch.int32
        )

    def _fill_sentinel(self):
        self.pool.k_buffer[0].fill_(-7)
        self.pool.v_buffer[0].fill_(-7)

    def _assert_sentinel(self):
        self.assertTrue(torch.all(self.pool.k_buffer[0] == -7))
        self.assertTrue(torch.all(self.pool.v_buffer[0] == -7))

    def test_noncontiguous_batch_and_head_strides_are_supported(self):
        key = torch.randn(
            (self.num_tokens, self.head_num, self.head_dim + 2),
            device="cuda",
            dtype=torch.float32,
        )[:, :, : self.head_dim]
        value = torch.randn(
            (self.num_tokens, self.head_num, self.head_dim + 2),
            device="cuda",
            dtype=torch.float32,
        )[:, :, : self.head_dim]
        self._fill_sentinel()

        self.pool.set_kv_buffer(
            self.layer,
            self.locations,
            key,
            value,
            dcp_kv_mask=self.mask,
        )

        selected = self.mask.bool()
        expected_key = torch.full_like(self.pool.k_buffer[0], -7)
        expected_value = torch.full_like(self.pool.v_buffer[0], -7)
        expected_key[self.locations[selected]] = key[selected]
        expected_value[self.locations[selected]] = value[selected]
        self.assertTrue(torch.equal(self.pool.k_buffer[0], expected_key))
        self.assertTrue(torch.equal(self.pool.v_buffer[0], expected_value))

    def test_noncontiguous_final_dimension_fails_clearly(self):
        key = torch.randn(
            (self.num_tokens, self.head_num, self.head_dim + 2),
            device="cuda",
            dtype=torch.float32,
        ).permute(0, 2, 1)
        value = torch.randn(
            (self.num_tokens, self.head_num, self.head_dim + 2),
            device="cuda",
            dtype=torch.float32,
        ).permute(0, 2, 1)
        self._fill_sentinel()

        with self.assertRaisesRegex(ValueError, "unit stride on the final D"):
            self.pool.set_kv_buffer(
                self.layer,
                self.locations,
                key,
                value,
                dcp_kv_mask=self.mask,
            )
        self._assert_sentinel()

    def test_packed_2d_representation_fails_clearly(self):
        key = torch.randn(
            (self.num_tokens, self.head_num, self.head_dim),
            device="cuda",
            dtype=torch.float32,
        ).reshape(self.num_tokens, self.head_num * self.head_dim)
        value = torch.randn_like(key)
        self._fill_sentinel()

        with self.assertRaisesRegex(ValueError, "unpacked \\[N, H, D\\]"):
            self.pool.set_kv_buffer(
                self.layer,
                self.locations,
                key,
                value,
                dcp_kv_mask=self.mask,
            )
        self._assert_sentinel()


if __name__ == "__main__":
    unittest.main()
