import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.flash_mla_sm120 import _split_kv_pages_to_64
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=4, suite="stage-b-test-1-gpu-small-amd-mi35x")


_SRC_PAGE_STRIDE = 149_760
_DST_PAGE_STRIDE = 37_440
_SRC_LANES_PER_PAGE = _SRC_PAGE_STRIDE // 8
_DST_LANES_PER_PAGE = _DST_PAGE_STRIDE // 8
_RATIO = 4
_PROBE_PROGRAMS = 458_868


@triton.jit
def _offset_probe_kernel(
    out_ptr,
    num_programs,
    SRC_LANES_PER_PAGE: tl.constexpr,
    DST_LANES_PER_PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    WIDEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDEN:
        pid = pid.to(tl.int64)
    if pid < num_programs - 4:
        return

    page_idx = pid // RATIO
    sub = pid % RATIO
    src_offset = page_idx * SRC_LANES_PER_PAGE
    dst_offset = (page_idx * RATIO + sub) * DST_LANES_PER_PAGE
    slot = pid - (num_programs - 4)
    tl.store(out_ptr + slot * 2, src_offset)
    tl.store(out_ptr + slot * 2 + 1, dst_offset)


def _wrap_int32(value):
    value &= 0xFFFFFFFF
    return value - 0x1_0000_0000 if value >= 0x8000_0000 else value


class TestPageSplitOffsets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA or HIP required")
        cls.device = torch.device("cuda")

    def test_int64_offsets_match_independent_reference(self):
        out = torch.empty((4, 2), dtype=torch.int64, device=self.device)
        _offset_probe_kernel[(_PROBE_PROGRAMS,)](
            out,
            _PROBE_PROGRAMS,
            _SRC_LANES_PER_PAGE,
            _DST_LANES_PER_PAGE,
            _RATIO,
            True,
            num_warps=1,
        )
        torch.cuda.synchronize()

        expected = []
        for pid in range(_PROBE_PROGRAMS - 4, _PROBE_PROGRAMS):
            page_idx = pid // _RATIO
            sub = pid % _RATIO
            expected.append(
                [
                    page_idx * _SRC_LANES_PER_PAGE,
                    (page_idx * _RATIO + sub) * _DST_LANES_PER_PAGE,
                ]
            )
        self.assertEqual(out.tolist(), expected)

    def test_int32_offsets_wrap_at_lane_boundary(self):
        out = torch.empty((4, 2), dtype=torch.int64, device=self.device)
        _offset_probe_kernel[(_PROBE_PROGRAMS,)](
            out,
            _PROBE_PROGRAMS,
            _SRC_LANES_PER_PAGE,
            _DST_LANES_PER_PAGE,
            _RATIO,
            False,
            num_warps=1,
        )
        torch.cuda.synchronize()

        expected = []
        for pid in range(_PROBE_PROGRAMS - 4, _PROBE_PROGRAMS):
            page_idx = pid // _RATIO
            sub = pid % _RATIO
            expected.append(
                [
                    _wrap_int32(page_idx * _SRC_LANES_PER_PAGE),
                    _wrap_int32((page_idx * _RATIO + sub) * _DST_LANES_PER_PAGE),
                ]
            )
        self.assertEqual(out.tolist(), expected)

    def test_small_page_split_matches_byte_reference(self):
        torch.manual_seed(34025)
        source = torch.randint(
            0, 256, (2, _SRC_PAGE_STRIDE), dtype=torch.uint8, device=self.device
        )
        output_view = _split_kv_pages_to_64(source, 256)
        output = output_view.as_strided((8, _DST_PAGE_STRIDE), (_DST_PAGE_STRIDE, 1))

        expected = torch.empty_like(output)
        for page in range(2):
            for sub in range(4):
                destination = page * 4 + sub
                expected[destination, :36_864] = source[
                    page, sub * 36_864 : (sub + 1) * 36_864
                ]
                expected[destination, 36_864 : 37_376] = source[
                    page, 147_456 + sub * 512 : 147_456 + (sub + 1) * 512
                ]

        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(output[:, :37_376].cpu(), expected[:, :37_376].cpu())
        )


if __name__ == "__main__":
    unittest.main()
