import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.utils import GenerationBatchResult


def test_eagle_verify_hot_path_has_no_host_tensor_conversion():
    """Keep EAGLE verification free of synchronous device-to-host conversions."""
    speculative_dir = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "sglang"
        / "srt"
        / "speculative"
    )
    source = speculative_dir / "eagle_worker_common.py"
    tree = ast.parse(source.read_text())
    verify = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_eagle_verify"
    )
    forbidden = []
    for node in ast.walk(verify):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"cpu", "tolist", "numpy"}:
                forbidden.append(f"{source.name}:{node.lineno}:{node.func.attr}")

    assert forbidden == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    ("token_ids", "accept_lens"),
    [
        ([], []),
        ([101, 102, 103, 201, 202, 203], [3, 2]),
    ],
)
def test_eagle_result_uses_async_pinned_d2h(token_ids, accept_lens):
    device = torch.device("cuda")
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

    assert result.next_token_ids.device.type == "cpu"
    assert result.accept_lens.device.type == "cpu"
    # PyTorch does not mark zero-sized allocations pinned on every backend.
    if token_ids:
        assert result.next_token_ids.is_pinned()
    if accept_lens:
        assert result.accept_lens.is_pinned()
    assert result.next_token_ids.tolist() == token_ids
    assert result.accept_lens.tolist() == accept_lens
