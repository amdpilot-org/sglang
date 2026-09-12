import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.utils import GenerationBatchResult


def _find_reachable_host_conversions(source_text, entrypoint="run_eagle_verify"):
    """Find forbidden conversions in an entrypoint and its module-local callees."""
    tree = ast.parse(source_text)
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [entrypoint]
    visited = set()
    forbidden = []

    while pending:
        function_name = pending.pop()
        if function_name in visited or function_name not in functions:
            continue
        visited.add(function_name)
        function = functions[function_name]
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr in {
                "cpu",
                "tolist",
                "numpy",
            }:
                forbidden.append(f"{function_name}:{node.lineno}:{node.func.attr}")
            elif isinstance(node.func, ast.Name) and node.func.id in functions:
                pending.append(node.func.id)

    return sorted(forbidden)


def test_eagle_verify_call_graph_has_no_host_tensor_conversion():
    """Keep EAGLE verification and its local helpers free of host conversions."""
    speculative_dir = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
    )
    source = speculative_dir / "eagle_worker_common.py"
    assert _find_reachable_host_conversions(source.read_text()) == []


def test_eagle_verify_guard_follows_local_helpers():
    source = """
def sync_helper(tensor):
    return tensor.tolist()

def run_eagle_verify(tensor):
    return sync_helper(tensor)
"""
    assert _find_reachable_host_conversions(source) == ["sync_helper:3:tolist"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("token_ids", "accept_lens"),
    [
        ([], []),
        ([101, 102, 103, 201, 202, 203], [3, 2]),
    ],
)
def test_eagle_result_uses_async_pinned_d2h(monkeypatch, token_ids, accept_lens):
    device = torch.device("cuda")
    copy_calls = []
    original_copy = torch.Tensor.copy_

    def record_copy(self, src, *args, **kwargs):
        copy_calls.append((self, src, args, kwargs))
        return original_copy(self, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    result = GenerationBatchResult(
        logits_output=SimpleNamespace(
            hidden_states=None,
            auxiliary_device_output=None,
            sampling_mask_output=None,
        ),
        next_token_ids=torch.tensor(token_ids, dtype=torch.int64, device=device),
        accept_lens=torch.tensor(accept_lens, dtype=torch.int32, device=device),
    )
    result.copy_done = torch.cuda.Event()

    result.copy_to_cpu(return_logprob=False)
    result.copy_done.synchronize()

    assert len(copy_calls) == 2
    assert all(dst.device.type == "cpu" for dst, _, _, _ in copy_calls)
    assert all(src.device.type == "cuda" for _, src, _, _ in copy_calls)
    assert all(args == () for _, _, args, _ in copy_calls)
    assert all(kwargs == {"non_blocking": True} for _, _, _, kwargs in copy_calls)
    assert result.next_token_ids.device.type == "cpu"
    assert result.accept_lens.device.type == "cpu"
    # PyTorch does not mark zero-sized allocations pinned on every backend.
    if token_ids:
        assert result.next_token_ids.is_pinned()
    if accept_lens:
        assert result.accept_lens.is_pinned()
    assert result.next_token_ids.tolist() == token_ids
    assert result.accept_lens.tolist() == accept_lens
