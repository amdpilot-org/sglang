"""Exercise MLXAttentionWrapper.__call__ routing without requiring Metal/MLX."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def load_call(path: Path):
    tree = ast.parse(path.read_text())
    wrapper = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLXAttentionWrapper"
    )
    method = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    method.decorator_list = []
    method.returns = None
    for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"get_context": lambda: SimpleNamespace(aot=SimpleNamespace(rope=None))}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["__call__"]


class WrapperProbe:
    _window_size = None

    def _batched_decode(self, x, ctx):
        return "hand-rolled"

    def _delegated_decode(self, x, ctx, *args, **kwargs):
        return "delegated", args, kwargs


call = load_call(Path(sys.argv[1]))
result = call(WrapperProbe(), "x", None, None, ("shared-k", "shared-v"))
assert result[0] == "delegated", result
assert result[1] == (("shared-k", "shared-v"),), result
print("PASS: shared-KV argument reaches delegated decode and preserves arity")
