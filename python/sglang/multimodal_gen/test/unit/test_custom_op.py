import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.layers import custom_op


class NativeOnlyOp(custom_op.CustomOp):
    def forward_native(self, value, *, increment=1):
        return value + increment


class XpuOverrideOp(NativeOnlyOp):
    def forward_xpu(self, value, *, increment=1):
        return value + increment + 100


class NoImplementationOp(custom_op.CustomOp):
    pass


class TestCustomOpXpuDispatch(unittest.TestCase):
    def setUp(self):
        self.platform_patches = [
            patch.object(custom_op, "_is_cuda", False),
            patch.object(custom_op.current_platform, "is_hip", return_value=False),
            patch.object(custom_op.current_platform, "is_npu", return_value=False),
            patch.object(custom_op.current_platform, "is_xpu", return_value=True),
            patch.object(custom_op.current_platform, "is_musa", return_value=False),
        ]
        for platform_patch in self.platform_patches:
            platform_patch.start()
            self.addCleanup(platform_patch.stop)

    def test_native_only_op_constructs_and_runs_on_xpu(self):
        op = NativeOnlyOp()

        self.assertEqual(op(4, increment=3), 7)
        self.assertEqual(op._forward_method, op.forward_xpu)

    def test_explicit_xpu_override_is_preserved(self):
        op = XpuOverrideOp()

        self.assertEqual(op(4, increment=3), 107)
        self.assertEqual(op._forward_method, op.forward_xpu)

    def test_missing_native_implementation_fails_on_execution(self):
        op = NoImplementationOp()

        with self.assertRaises(NotImplementedError):
            op(4)


if __name__ == "__main__":
    unittest.main()
