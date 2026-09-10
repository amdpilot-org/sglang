import os
import pytest
import torch

from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


EXPERTS = 8
TOPK = 2
HIDDEN_SIZE = 64
INTERMEDIATE_SIZE = 64
TOKEN_COUNTS = (1, 4, 16)
GUARD_SIZE = 32
SENTINEL = -12345.0


def _reference(hidden_states, w1, w2, topk_weights, topk_ids):
    token_count = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    expanded = hidden_states.repeat_interleave(topk_ids.shape[1], dim=0)
    expert_outputs = torch.empty(
        token_count * topk,
        HIDDEN_SIZE,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    for expert_index in range(EXPERTS):
        mask = (topk_ids == expert_index).reshape(-1)
        if not bool(mask.any()):
            continue
        gate_up = expanded[mask] @ w1[expert_index].t()
        gate, up = gate_up.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate) * up
        expert_outputs[mask] = activated @ w2[expert_index].t()
    return (
        expert_outputs.view(token_count, topk, HIDDEN_SIZE)
        * topk_weights.to(expert_outputs.dtype).unsqueeze(-1)
    ).sum(dim=1)


def _make_batch(hidden_states, dtype):
    logits = torch.randn(
        hidden_states.shape[0], EXPERTS, device=hidden_states.device, dtype=dtype
    )
    probabilities = torch.softmax(logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(probabilities, TOPK, dim=-1)
    topk_weights = topk_weights.to(dtype)
    return StandardTopKOutput(topk_weights, topk_ids.to(torch.int32), logits)


@pytest.fixture(autouse=True)
def publish_runtime_config():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    if not model_parallel_is_initialized():
        if not torch.distributed.is_initialized():
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=(
                    f"file:///tmp/sglang-fused-moe-reuse-{os.getpid()}"
                ),
                backend="gloo",
            )
        initialize_model_parallel(tensor_model_parallel_size=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fused_moe_fresh_and_reused_outputs_match_reference():
    torch.manual_seed(32312)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    w1 = torch.randn(
        EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype
    ).mul_(0.02)
    w2 = torch.randn(
        EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype
    ).mul_(0.02)

    guarded_storage = torch.full(
        (GUARD_SIZE + max(TOKEN_COUNTS) * HIDDEN_SIZE + GUARD_SIZE,),
        SENTINEL,
        device=device,
        dtype=dtype,
    )
    before_guard = guarded_storage[:GUARD_SIZE].clone()
    after_guard = guarded_storage[-GUARD_SIZE:].clone()
    max_token_count = max(TOKEN_COUNTS)
    reused_view = guarded_storage[
        GUARD_SIZE : GUARD_SIZE + max_token_count * HIDDEN_SIZE
    ].view(max_token_count, HIDDEN_SIZE)
    reused_address = reused_view.data_ptr()

    for token_count in TOKEN_COUNTS:
        hidden_states = torch.randn(
            token_count, HIDDEN_SIZE, device=device, dtype=dtype
        ).mul_(0.05)
        topk_output = _make_batch(hidden_states, dtype)
        topk_weights, topk_ids, _ = topk_output
        reference = _reference(hidden_states, w1, w2, topk_weights, topk_ids)
        input_sentinel = hidden_states.clone()

        fresh_output = fused_moe(
            hidden_states,
            w1,
            w2,
            topk_output,
            MoeRunnerConfig(inplace=False),
        )
        assert fresh_output.data_ptr() != hidden_states.data_ptr()
        assert fresh_output.dtype == dtype
        assert fresh_output.shape == hidden_states.shape
        assert torch.equal(hidden_states, input_sentinel)
        torch.testing.assert_close(
            fresh_output.float(), reference.float(), rtol=2e-2, atol=2e-2
        )

        reused_output = reused_view[:token_count]
        reused_output.copy_(hidden_states)
        fused_moe(
            reused_output,
            w1,
            w2,
            topk_output,
            MoeRunnerConfig(inplace=True),
        )
        assert reused_output.data_ptr() == reused_address
        assert reused_output.dtype == dtype
        assert reused_output.shape == hidden_states.shape
        torch.testing.assert_close(
            reused_output.float(), reference.float(), rtol=2e-2, atol=2e-2
        )
        assert torch.equal(guarded_storage[:GUARD_SIZE], before_guard)
        assert torch.equal(guarded_storage[-GUARD_SIZE:], after_guard)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fused_moe_rejects_unsupported_variants():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_states = torch.randn(1, HIDDEN_SIZE, device=device, dtype=dtype)
    w1 = torch.randn(
        EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype
    )
    w2 = torch.randn(
        EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype
    )
    topk_output = _make_batch(hidden_states, dtype)

    with pytest.raises(AssertionError):
        fused_moe(
            hidden_states.double(),
            w1.double(),
            w2.double(),
            topk_output,
            MoeRunnerConfig(inplace=False),
        )

    with pytest.raises(AssertionError, match="no combine \\+ inplace"):
        fused_moe(
            hidden_states,
            w1,
            w2,
            topk_output,
            MoeRunnerConfig(inplace=True, no_combine=True),
        )
