import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=45, suite="stage-b-test-cpu-intel")


class TestDwdpOptionalCudaImport(unittest.TestCase):
    def _run_import(self, setup: str, assertions: str) -> subprocess.CompletedProcess:
        script = textwrap.dedent(
            f"""
            {setup}
            from sglang.srt.layers.moe.dwdp import DwdpManager
            import sglang.srt.layers.moe.dwdp.page_pool as page_pool
            import sglang.srt.layers.moe.dwdp.transport as transport
            {assertions}
            """
        )
        return subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONPATH": "python"},
            cwd=os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            ),
            capture_output=True,
            text=True,
        )

    def test_public_import_without_cuda_bindings(self):
        result = self._run_import(
            """
            import builtins
            original_import = builtins.__import__
            def import_without_cuda(name, globals=None, locals=None, fromlist=(), level=0):
                importer = (globals or {}).get("__name__", "")
                if (name == "cuda" or name.startswith("cuda.")) and (
                    importer.startswith("sglang.srt.layers.moe.dwdp")
                    or importer == "sglang.srt.utils.cuda_vmm_utils"
                ):
                    raise ModuleNotFoundError("No module named 'cuda'", name="cuda")
                return original_import(name, globals, locals, fromlist, level)
            builtins.__import__ = import_without_cuda
            """,
            "assert page_pool.cuda is None; assert transport.cuda is None",
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_import_preserves_available_cuda_driver(self):
        result = self._run_import(
            """
            import sys
            import types
            import torch
            cuda_package = types.ModuleType("cuda")
            bindings = types.ModuleType("cuda.bindings")
            driver = types.ModuleType("cuda.bindings.driver")
            driver.__file__ = "<fake-cuda-driver>"
            driver.CUmemAllocationGranularity_flags = types.SimpleNamespace(
                CU_MEM_ALLOC_GRANULARITY_RECOMMENDED=object()
            )
            driver.__getattr__ = lambda name: type(name, (), {})
            cuda_package.bindings = bindings
            bindings.driver = driver
            sys.modules.update({
                "cuda": cuda_package,
                "cuda.bindings": bindings,
                "cuda.bindings.driver": driver,
            })
            """,
            "assert page_pool.cuda is driver; assert transport.cuda is driver",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
