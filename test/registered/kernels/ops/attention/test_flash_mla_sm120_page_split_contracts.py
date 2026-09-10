import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_sm120 import _split_kv_pages_to_64
from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd-mi35x")


_SRC_PAGE_STRIDE = 149_760
_DST_PAGE_STRIDE = 37_440
_DATA_PER_SUBPAGE = 36_864
_SCALE_PER_SUBPAGE = 512
_SRC_SCALE_OFFSET = 147_456
_RATIO = 4
_SENTINEL = 0xA5


class TestPageSplitContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA or HIP required")
        cls.device = torch.device("cuda")

    def _install_destination(self, num_pages):
        destination = torch.full(
            (num_pages * _RATIO, _DST_PAGE_STRIDE),
            _SENTINEL,
            dtype=torch.uint8,
            device=self.device,
        )
        buffers = get_resources().buffers
        key = f"flash_mla_sm120_split:{destination.device}"
        missing = object()
        old = buffers.get(key, missing)

        def restore():
            if old is missing:
                buffers.pop(key, None)
            else:
                buffers[key] = old

        self.addCleanup(restore)
        buffers[key] = destination
        return destination

    def _make_source(self, dtype):
        torch.manual_seed(34025)
        if dtype in (torch.float16, torch.bfloat16):
            values = torch.randn(
                (2, _SRC_PAGE_STRIDE // 2), dtype=dtype, device=self.device
            )
            source = values.view(torch.uint8).view(dtype).as_strided(
                (2, 256, 1, 292),
                (_SRC_PAGE_STRIDE // 2, 292, 292, 1),
            )
        else:
            values = (
                torch.randn((2, _SRC_PAGE_STRIDE), dtype=torch.bfloat16, device=self.device)
                .clamp(-2.0, 2.0)
                .to(dtype)
            )
            source = values.view(torch.uint8).view(dtype).as_strided(
                (2, 256, 1, 584),
                (_SRC_PAGE_STRIDE, 584, 584, 1),
            )
        return source, values.view(torch.uint8)

    def _expected_destination(self, source_bytes):
        expected = torch.full(
            (8, _DST_PAGE_STRIDE),
            _SENTINEL,
            dtype=torch.uint8,
            device=source_bytes.device,
        )
        for page in range(2):
            for subpage in range(_RATIO):
                destination = page * _RATIO + subpage
                expected[destination, :_DATA_PER_SUBPAGE] = source_bytes[
                    page,
                    subpage * _DATA_PER_SUBPAGE : (subpage + 1) * _DATA_PER_SUBPAGE,
                ]
                scale_offset = _SRC_SCALE_OFFSET + subpage * _SCALE_PER_SUBPAGE
                expected[
                    destination,
                    _DATA_PER_SUBPAGE : _DATA_PER_SUBPAGE + _SCALE_PER_SUBPAGE,
                ] = source_bytes[
                    page,
                    scale_offset : scale_offset + _SCALE_PER_SUBPAGE,
                ]
        return expected

    def test_documented_dtype_views_match_independent_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                source, source_bytes = self._make_source(dtype)
                destination = self._install_destination(2)
                output = _split_kv_pages_to_64(source.view(torch.uint8), 256)
                torch.cuda.synchronize()

                actual = destination.cpu()
                expected = self._expected_destination(source_bytes).cpu()
                torch.testing.assert_close(
                    actual[:, : _DATA_PER_SUBPAGE + _SCALE_PER_SUBPAGE],
                    expected[:, : _DATA_PER_SUBPAGE + _SCALE_PER_SUBPAGE],
                    atol=0,
                    rtol=0,
                )
                self.assertTrue(
                    torch.equal(actual.view(dtype), expected.view(dtype))
                )
                self.assertTrue(
                    bool(
                        (
                            actual[:, _DATA_PER_SUBPAGE + _SCALE_PER_SUBPAGE :]
                            == _SENTINEL
                        ).all()
                    )
                )
                self.assertEqual(output.data_ptr(), destination.data_ptr())

    def test_non_uint8_inputs_fail_clearly(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                source, _ = self._make_source(dtype)
                with self.assertRaisesRegex(
                    AssertionError, "expects a uint8 byte view"
                ):
                    _split_kv_pages_to_64(source, 256)

    def test_page_size_64_aliases_input(self):
        source = torch.empty(
            (3, 64, 1, 584), dtype=torch.uint8, device=self.device
        )
        self.assertIs(_split_kv_pages_to_64(source, 64), source)

    def test_cuda_graph_reuses_static_destination_address(self):
        source = torch.randint(
            0,
            256,
            (2, _SRC_PAGE_STRIDE),
            dtype=torch.uint8,
            device=self.device,
        )
        destination = self._install_destination(2)
        output = _split_kv_pages_to_64(source, 256)
        torch.cuda.synchronize()
        address = output.data_ptr()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            _split_kv_pages_to_64(source, 256)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = _split_kv_pages_to_64(source, 256)
        self.assertEqual(captured.data_ptr(), address)
        self.assertEqual(destination.data_ptr(), address)

        destination.fill_(0x5A)
        source.copy_(
            torch.randint(
                0,
                256,
                (2, _SRC_PAGE_STRIDE),
                dtype=torch.uint8,
                device=self.device,
            )
        )
        graph.replay()
        torch.cuda.synchronize()
        self.assertFalse(bool((destination == 0x5A).all()))


if __name__ == "__main__":
    unittest.main()
