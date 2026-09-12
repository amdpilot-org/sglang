"""Structural regression for the SM10x TGV two-CTA exit barrier.

This intentionally avoids importing the NVIDIA-only CuTe DSL module so the
kernel-lifetime invariant can also be checked on non-NVIDIA review hosts.
"""

import ast
from pathlib import Path


def check_exit_barrier(source: str) -> None:
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TgvGemmCuteExtKernel"
    )
    call = next(
        node
        for node in kernel.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "kernel"
    )

    dispatch_lines = [
        node.lineno
        for node in ast.walk(call)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"dma_a_warp", "dma_b_warp", "mma_warp", "epilog_warp"}
    ]
    candidates = []
    for node in ast.walk(call):
        if not isinstance(node, ast.If):
            continue
        condition = ast.unparse(node.test)
        calls = [
            ast.unparse(item.func)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        ]
        if condition == "cutlass.const_expr(self.use_2cta)" and calls[-2:] == [
            "cute.arch.cluster_arrive_relaxed",
            "cute.arch.cluster_wait",
        ]:
            candidates.append(node)

    assert dispatch_lines, "TGV warp dispatch calls were not found"
    assert candidates, "missing two-CTA cluster arrive/wait exit barrier"
    assert max(node.lineno for node in candidates) > max(dispatch_lines), (
        "two-CTA cluster barrier must follow every warp dispatch branch"
    )


def test_checked_in_kernel_has_gated_trailing_barrier() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py"
    check_exit_barrier(source.read_text())


def test_boundary_rejects_missing_wait() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py"
    text = source.read_text()
    marker = "            cute.arch.cluster_wait()\n\n    # "
    assert marker in text
    broken = text.replace(marker, "\n    # ", 1)
    try:
        check_exit_barrier(broken)
    except AssertionError as error:
        assert "missing two-CTA" in str(error)
    else:
        raise AssertionError("regression checker accepted an incomplete barrier")


def test_boundary_rejects_barrier_before_dispatch() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py"
    text = source.read_text()
    block_start = text.index(
        "        if cutlass.const_expr(self.use_2cta):\n"
        "            # Cluster-wide exit barrier"
    )
    block_end = text.index("\n\n    # ", block_start)
    block = text[block_start:block_end]
    broken = text[:block_start] + text[block_end:]
    insertion = broken.index("        # ---- Warp dispatch")
    broken = broken[:insertion] + block + "\n\n" + broken[insertion:]
    try:
        check_exit_barrier(broken)
    except AssertionError as error:
        assert "must follow" in str(error)
    else:
        raise AssertionError("regression checker accepted an early barrier")
