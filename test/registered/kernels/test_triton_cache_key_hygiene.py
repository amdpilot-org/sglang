import ast
import unittest
from pathlib import Path

import torch

from sglang.srt.mem_cache.memory_pool import masked_set_kv_buffer_kernel
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

REPO_ROOT = Path(__file__).resolve().parents[3]
XPU_CHUNK_DELTA_H = (
    REPO_ROOT / "python/sglang/srt/hardware_backend/xpu/kernels/fla/chunk_delta_h.py"
)


class TestTritonCacheKeyHygiene(unittest.TestCase):
    def test_masked_set_write_count_is_not_specialized(self):
        params = {param.name: param for param in masked_set_kv_buffer_kernel.params}
        self.assertNotIn("N", params)

    def test_xpu_chunk_kernel_has_no_unused_nt_bucket(self):
        tree = ast.parse(XPU_CHUNK_DELTA_H.read_text(encoding="utf-8"))
        kernel = next(
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_k_loop"
        )
        self.assertNotIn("NT_BUCKET", {arg.arg for arg in kernel.args.args})

        autotune = next(
            decorator
            for decorator in kernel.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "autotune"
        )
        key_keyword = next(kw for kw in autotune.keywords if kw.arg == "key")
        self.assertNotIn("NT_BUCKET", ast.literal_eval(key_keyword.value))
        launch_keywords = {
            keyword.arg
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        }
        self.assertNotIn("NT_BUCKET", launch_keywords)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_masked_set_matches_torch_for_boundary_shapes(self):
        device = torch.device("cuda")
        cases = (
            (1, 1, 1, [True]),
            (3, 2, 65, [True, False, True]),
            (65, 3, 47, [i % 3 == 0 for i in range(65)]),
        )
        for n, h, d, mask_values in cases:
            with self.subTest(n=n, h=h, d=d):
                k = torch.randn(n, h, d, device=device)
                v = torch.randn(n, h, d, device=device)
                loc = torch.randperm(n, device=device, dtype=torch.int64)
                mask = torch.tensor(mask_values, device=device, dtype=torch.bool)
                k_buffer = torch.full((n, h, d), float("nan"), device=device)
                v_buffer = torch.full((n, h, d), float("nan"), device=device)

                masked_set_kv_buffer_kernel[(n,)](
                    k,
                    v,
                    k_buffer,
                    v_buffer,
                    loc,
                    mask,
                    h,
                    d,
                    128,
                    k.stride(0),
                    k.stride(1),
                    v.stride(0),
                    v.stride(1),
                )

                expected_k = torch.full_like(k_buffer, float("nan"))
                expected_v = torch.full_like(v_buffer, float("nan"))
                expected_k[loc[mask]] = k[mask]
                expected_v[loc[mask]] = v[mask]
                torch.testing.assert_close(k_buffer, expected_k, equal_nan=True)
                torch.testing.assert_close(v_buffer, expected_v, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
