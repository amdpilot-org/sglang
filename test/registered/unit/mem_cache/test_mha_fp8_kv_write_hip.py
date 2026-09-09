"""ROCm MHA FP8 KV-cache write tests."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=15, stage="stage-b", runner_config="1-gpu-small-amd")

_RUNNABLE = torch.cuda.is_available() and bool(torch.version.hip) and is_fp8_fnuz()


def _build_pool(head_dim: int = 128, v_head_dim: int = 128) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=31,
        page_size=1,
        dtype=torch.float8_e4m3fnuz,
        head_num=2,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def _reference(
    pool: MHATokenToKVPool,
    k: torch.Tensor,
    v: torch.Tensor,
    loc: torch.Tensor,
    k_scale=1.0,
    v_scale=1.0,
):
    k_expected = torch.zeros(pool.k_buffer[0].shape, dtype=torch.float32)
    v_expected = torch.zeros(pool.v_buffer[0].shape, dtype=torch.float32)
    loc_cpu = loc.cpu()
    valid_loc = loc_cpu != 0
    k_expected[loc_cpu[valid_loc]] = (
        k.cpu().float()[valid_loc] / float(k_scale)
    )
    v_expected[loc_cpu[valid_loc]] = (
        v.cpu().float()[valid_loc] / float(v_scale)
    )
    return (
        k_expected.clamp(torch.finfo(pool.dtype).min, torch.finfo(pool.dtype).max)
        .to(pool.dtype)
        .cuda()
        .view(torch.uint8),
        v_expected.clamp(torch.finfo(pool.dtype).min, torch.finfo(pool.dtype).max)
        .to(pool.dtype)
        .cuda()
        .view(torch.uint8),
    )


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx94x e4m3fnuz")
class TestMHAFP8KVWriteHIP(unittest.TestCase):
    def _run_write(
        self,
        pool,
        k,
        v,
        loc,
        k_scale=1.0,
        v_scale=1.0,
    ):
        pool.k_buffer[0].zero_()
        pool.v_buffer[0].zero_()
        k_before = k.clone()
        v_before = v.clone()
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0),
            loc,
            k,
            v,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        torch.cuda.synchronize()
        expected_k, expected_v = _reference(
            pool, k, v, loc, k_scale, v_scale
        )
        self.assertTrue(torch.equal(k, k_before))
        self.assertTrue(torch.equal(v, v_before))
        self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))

    def test_scalar_and_device_scales(self):
        pool = _build_pool()
        loc = torch.tensor([0, 2, 5], device="cuda")
        k = torch.randn(3, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        self._run_write(pool, k, v, loc)
        self._run_write(pool, k, v, loc, 1.25, 1.75)
        self._run_write(
            pool,
            k,
            v,
            loc,
            torch.tensor(1.25, device="cuda"),
            torch.tensor(1.75, device="cuda"),
        )

    def test_asymmetric_rows(self):
        pool = _build_pool(head_dim=192, v_head_dim=128)
        loc = torch.tensor([3, 7], device="cuda")
        k = torch.randn(2, 2, 192, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
        self._run_write(pool, k, v, loc, 1.5, 2.5)

    def test_e4m3fnuz_range(self):
        pool = _build_pool(head_dim=8, v_head_dim=8)
        loc = torch.tensor([1], device="cuda")
        values = torch.tensor(
            [0.0, 224.0, 240.0, 248.0, -240.0, -248.0, 300.0, -300.0],
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = values.view(1, 1, 8).expand(1, 2, 8).contiguous()
        self._run_write(pool, k, k, loc)

    def test_noncontiguous_fallback(self):
        pool = _build_pool()
        loc = torch.tensor([2, 4], device="cuda")
        k_base = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
        v_base = torch.randn_like(k_base)
        k = k_base[:, ::2, :]
        v = v_base[:, ::2, :]
        self.assertFalse(k.is_contiguous())
        self._run_write(pool, k, v, loc, 1.25, 1.75)

    def test_alignment_fallback(self):
        pool = _build_pool(head_dim=7, v_head_dim=7)
        loc = torch.tensor([2], device="cuda")
        k = torch.randn(1, 1, 7, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        self._run_write(pool, k, v, loc, 1.25, 1.75)

    def test_untouched_slots(self):
        pool = _build_pool()
        pool.k_buffer[0].random_(0, 256)
        pool.v_buffer[0].random_(0, 256)
        k_before = pool.k_buffer[0].clone()
        v_before = pool.v_buffer[0].clone()
        loc = torch.tensor([2, 5], device="cuda")
        k = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=0), loc, k, v, k_scale=1.25, v_scale=1.75
        )
        torch.cuda.synchronize()
        expected_k, expected_v = _reference(pool, k, v, loc, 1.25, 1.75)
        self.assertTrue(torch.equal(pool.k_buffer[0][2], expected_k[2]))
        self.assertTrue(torch.equal(pool.v_buffer[0][5], expected_v[5]))
        untouched = torch.ones(
            pool.k_buffer[0].shape[0], dtype=torch.bool, device="cuda"
        )
        untouched[loc] = False
        self.assertTrue(torch.equal(pool.k_buffer[0][untouched], k_before[untouched]))
        self.assertTrue(torch.equal(pool.v_buffer[0][untouched], v_before[untouched]))

    def test_cuda_graph_changes_slots_and_device_scales(self):
        pool = _build_pool()
        loc = torch.tensor([1, 2], device="cuda")
        k = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        k_scale = torch.tensor(1.0, device="cuda")
        v_scale = torch.tensor(1.0, device="cuda")
        layer = SimpleNamespace(layer_id=0)

        pool.set_kv_buffer(layer, loc, k, v, k_scale, v_scale)
        pool.k_buffer[0].zero_()
        pool.v_buffer[0].zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_kv_buffer(layer, loc, k, v, k_scale, v_scale)

        loc.copy_(torch.tensor([3, 5], device="cuda"))
        k_scale.fill_(1.25)
        v_scale.fill_(1.75)
        graph.replay()
        torch.cuda.synchronize()
        expected_k, expected_v = _reference(pool, k, v, loc, k_scale, v_scale)
        self.assertTrue(torch.equal(pool.k_buffer[0], expected_k))
        self.assertTrue(torch.equal(pool.v_buffer[0], expected_v))


if __name__ == "__main__":
    unittest.main()
