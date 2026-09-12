#!/usr/bin/env python3
"""Static contract checks for the exact custom-all-reduce review revisions."""

import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
utils = (
    root
    / "python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py"
).read_text()
legacy = (
    root / "python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh"
).read_text()
jit = (
    root / "python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh"
).read_text()

tree = ast.parse(utils)
guard = next(
    (
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SingleStreamGuard"
    ),
    None,
)
capture_bypass = False
if guard:
    method = next(
        n
        for n in guard.body
        if isinstance(n, ast.FunctionDef) and n.name == "maybe_serialize"
    )
    capture_bypass = any(
        isinstance(n, ast.If)
        and "is_current_stream_capturing" in ast.unparse(n.test)
        and any(isinstance(x, ast.Return) for x in n.body)
        for n in method.body
    )

checks = {
    "host_guard_present": guard is not None,
    "capture_explicitly_bypasses_guard": capture_bypass,
    "jit_poll_has_unbounded_while_true": "while (true)" in jit or "while(true)" in jit,
    "legacy_poll_has_unbounded_while": "while (" in legacy or "while(" in legacy,
}
for key, value in checks.items():
    print(f"{key}={value}")

# Full resolution requires graph replay safety and bounded device failure.
full = checks["host_guard_present"] and not any(
    (
        checks["capture_explicitly_bypasses_guard"],
        checks["jit_poll_has_unbounded_while_true"],
        checks["legacy_poll_has_unbounded_while"],
    )
)
print(f"fully_resolves_original={full}")
sys.exit(0 if full else 1)
