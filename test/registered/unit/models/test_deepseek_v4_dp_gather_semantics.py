"""Regression coverage for DeepSeek-V4 DP-attention gather semantics.

Post-attention hidden states and token IDs are replicated across attention-TP
ranks.  They must use the replicate gather; treating them as partial values
adds the replicas and scales them by ``attn_tp_size``.  The TBO pre-MoE input,
on the other hand, is intentionally partial and must retain its partial gather.
"""

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
    return [call for call in calls if isinstance(call.func, ast.Name) and call.func.id == name]


def test_post_attention_moe_gathers_replicated_hidden_states():
    calls = _function_calls(_MODELS_DIR / "deepseek_v4.py", "_run_moe_ffn_dp_sync")

    replicate_calls = _named_calls(calls, "dp_gather_replicate")
    assert len(replicate_calls) == 1
    assert isinstance(replicate_calls[0].args[1], ast.Name)
    assert replicate_calls[0].args[1].id == "local_hidden_states"


def test_main_and_nextn_clone_replicated_token_ids_before_gather():
    for filename in ("deepseek_v4.py", "deepseek_v4_nextn.py"):
        calls = _function_calls(_MODELS_DIR / filename, "forward")
        matching = [
            call
            for call in _named_calls(calls, "dp_gather_replicate")
            if isinstance(call.args[0], ast.Name)
            and call.args[0].id == "input_ids_global"
        ]
        assert len(matching) == 1, filename
        local_arg = matching[0].args[1]
        assert isinstance(local_arg, ast.Call), filename
        assert isinstance(local_arg.func, ast.Attribute), filename
        assert local_arg.func.attr == "clone", filename


def test_tbo_gather_keeps_partial_semantics():
    calls = _function_calls(_MODELS_DIR / "deepseek_v4.py", "op_gather_a")

    assert len(_named_calls(calls, "dp_gather_partial")) == 1
    assert not _named_calls(calls, "dp_gather_replicate")
