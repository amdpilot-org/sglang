#!/usr/bin/env python3
"""Check that the per-communicator lock covers each native launch."""

import ast
import sys
from pathlib import Path


def function(tree, class_name, function_name):
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )


root = Path(sys.argv[1]).resolve()
utils = ast.parse(
    (
        root
        / "python/sglang/srt/distributed/device_communicators/custom_all_reduce_utils.py"
    ).read_text()
)
legacy = ast.parse(
    (
        root / "python/sglang/srt/distributed/device_communicators/custom_all_reduce.py"
    ).read_text()
)
v2 = ast.parse(
    (
        root
        / "python/sglang/srt/distributed/device_communicators/custom_all_reduce_v2.py"
    ).read_text()
)

serialize = function(utils, "SingleStreamGuard", "serialize")
assert any(isinstance(node, ast.With) for node in ast.walk(serialize)), (
    "SingleStreamGuard.serialize must hold a lock while yielding to the launch"
)

for tree, cls, method in (
    (legacy, "CustomAllreduce", "_all_reduce_impl"),
    (v2, "CustomAllReduceV2", "custom_all_reduce"),
):
    launch = function(tree, cls, method)
    guarded = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "serialize"
            for item in node.items
        )
    ]
    assert guarded, f"{cls}.{method} must enqueue under stream_guard.serialize()"

print("host-thread serialization covers legacy and V2 native launches")
