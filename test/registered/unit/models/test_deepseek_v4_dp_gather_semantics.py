"""Regression coverage for DeepSeek-V4 DP-attention gather semantics."""

import ast
import os
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_MODELS_DIR = Path(
    os.environ.get(
        "SGLANG_DSV4_GATHER_TEST_MODELS_DIR",
        Path(__file__).parents[4] / "python" / "sglang" / "srt" / "models",
    )
)


def _function_calls(path: Path, function_name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    assert functions, f"expected {function_name} in {path}"
    return [
        node
        for function in functions
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]


def _named_calls(calls: list[ast.Call], name: str) -> list[ast.Call]:
    return [
        call
        for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == name
    ]


def _expression(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _same_expression(actual: ast.expr, expected: str) -> bool:
    return ast.dump(actual, include_attributes=False) == ast.dump(
        _expression(expected), include_attributes=False
    )


def test_post_attention_moe_gathers_replicated_hidden_states():
    calls = _function_calls(_MODELS_DIR / "deepseek_v4.py", "_run_moe_ffn_dp_sync")

    replicate_calls = _named_calls(calls, "dp_gather_replicate")
    assert len(replicate_calls) == 1
    assert len(replicate_calls[0].args) >= 2
    assert _same_expression(replicate_calls[0].args[1], "local_hidden_states")
    assert not _named_calls(calls, "dp_gather_partial")


def test_main_and_nextn_clone_local_token_ids_before_gather():
    for filename in ("deepseek_v4.py", "deepseek_v4_nextn.py"):
        calls = _function_calls(_MODELS_DIR / filename, "forward")
        matching = [
            call
            for call in _named_calls(calls, "dp_gather_replicate")
            if len(call.args) >= 2
            and _same_expression(call.args[0], "input_ids_global")
        ]
        assert len(matching) == 1, filename
        assert _same_expression(
            matching[0].args[1], "input_ids[:, None].clone()"
        ), filename


def test_tbo_gather_keeps_partial_semantics():
    calls = _function_calls(_MODELS_DIR / "deepseek_v4.py", "op_gather_a")

    assert len(_named_calls(calls, "dp_gather_partial")) == 1
    assert not _named_calls(calls, "dp_gather_replicate")
