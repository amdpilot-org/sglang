#!/usr/bin/env python3
"""Check that graph capture cannot select or directly launch custom AR."""

import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = {
    "guard": root
    / "python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py",
    "legacy": root
    / "python/sglang/srt/distributed/device_communicators/custom_all_reduce.py",
    "v2": root
    / "python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py",
}


def method(path: pathlib.Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


guard_source = ast.unparse(
    method(paths["guard"], "SingleStreamGuard", "maybe_serialize")
)
legacy_source = ast.unparse(
    method(paths["legacy"], "CustomAllreduce", "should_custom_ar")
)
v2_source = ast.unparse(method(paths["v2"], "CustomAllReduceV2", "should_custom_ar"))

checks = {
    "direct_capture_rejected": "is_current_stream_capturing" in guard_source
    and "raise RuntimeError" in guard_source,
    "legacy_capture_not_selected": "is_current_stream_capturing" in legacy_source,
    "v2_capture_not_selected": "is_current_stream_capturing" in v2_source,
}
for name, passed in checks.items():
    print(f"{name}={passed}")
print(f"graph_capture_contract_safe={all(checks.values())}")
raise SystemExit(0 if all(checks.values()) else 1)
