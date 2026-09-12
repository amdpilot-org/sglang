"""Regression tests for synchronization-free Mamba slot donation."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SOURCES = (
    _REPO_ROOT / "python/sglang/srt/mem_cache/mamba_radix_cache.py",
    _REPO_ROOT / "python/sglang/srt/mem_cache/memory_pool.py",
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _debug_flag_assignment(tree: ast.Module) -> ast.Assign:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_MAMBA_DEBUG_ASSERTS"
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one _MAMBA_DEBUG_ASSERTS assignment, found {len(matches)}"
        )
    return matches[0]


class TestMambaDebugAsserts(CustomTestCase):
    def test_flag_is_off_by_default_and_only_one_enables_it(self):
        """Production defaults and non-canonical values must stay sync-free."""
        for source in _SOURCES:
            assignment = _debug_flag_assignment(_parse(source))
            code = compile(
                ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                str(source),
                "exec",
            )
            for value, expected in ((None, False), ("0", False), ("true", False), ("1", True)):
                env = {} if value is None else {"SGLANG_MAMBA_DEBUG_ASSERTS": value}
                namespace = {"os": os}
                with patch.dict(os.environ, env, clear=True):
                    exec(code, namespace)
                self.assertIs(namespace["_MAMBA_DEBUG_ASSERTS"], expected)

    def test_scalar_reads_are_nested_under_debug_gate(self):
        """Neither reported ``Tensor.item`` may execute on the default path."""
        expected = {
            "mamba_radix_cache.py": "src_active",
            "memory_pool.py": "mamba_value_donated",
        }
        for source in _SOURCES:
            tree = _parse(source)
            guarded_item_receivers = []
            unguarded_item_receivers = []

            class Visitor(ast.NodeVisitor):
                def __init__(self):
                    self.debug_gate_depth = 0

                def visit_If(self, node: ast.If):
                    is_debug_gate = (
                        isinstance(node.test, ast.Name)
                        and node.test.id == "_MAMBA_DEBUG_ASSERTS"
                    )
                    self.debug_gate_depth += is_debug_gate
                    for child in node.body:
                        self.visit(child)
                    self.debug_gate_depth -= is_debug_gate
                    for child in node.orelse:
                        self.visit(child)

                def visit_Call(self, node: ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "item"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == expected[source.name]
                    ):
                        target = (
                            guarded_item_receivers
                            if self.debug_gate_depth
                            else unguarded_item_receivers
                        )
                        target.append(node.func.value.id)
                    self.generic_visit(node)

            Visitor().visit(tree)
            self.assertEqual(guarded_item_receivers, [expected[source.name]])
            self.assertEqual(unguarded_item_receivers, [])


if __name__ == "__main__":
    unittest.main()
