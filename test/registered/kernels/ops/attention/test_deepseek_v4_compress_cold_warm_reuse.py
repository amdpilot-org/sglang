from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    CompressorPrefillPlan,
    compress_forward,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.deepseek_v4.common import (
    make_paged_context,
    to_seq_extend,
)

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=25, suite="nightly-amd-kernel-1-gpu", nightly=True)

HEAD_DIM = 512
LAST_DIM = HEAD_DIM * 2
SENTINEL = -12345.0
SHAPE_SEQUENCE = (
    ((128, 128),),
    ((128, 128), (128, 64)),
    ((128, 128), (128, 64), (256, 128)),
)


def _make_case(
    seq_extend_pairs: tuple[tuple[int, int], ...], dtype: torch.dtype
) -> tuple:
    seq_lens, extend_lens, num_q_tokens = to_seq_extend(list(seq_extend_pairs))
    context = make_paged_context(
        bs=len(seq_extend_pairs),
        compress_ratio=128,
        head_dim=HEAD_DIM,
        swa_page_size=256,
        ring_size=256,
        num_reqs_capacity=8,
    )
    device = torch.device(get_device())
    plan = CompressorPrefillPlan.generate(
        compress_ratio=128,
        req_pool_indices=context.req_pool_indices,
        seq_lens=seq_lens.to(device),
        extend_lens=extend_lens.to(device),
        req_to_token=context.req_to_token,
        full_to_state=context.full_to_swa,
        swa_page_size=context.swa_page_size,
        ring_size=context.ring_size,
        num_q_tokens=num_q_tokens,
        use_cuda_graph=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    kv_score_input = torch.randn(
        (num_q_tokens, LAST_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    ape = torch.randn(
        (128, HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=dtype)
    buffer_storage = torch.full(
        (context.num_pages + 2, 128, LAST_DIM),
        SENTINEL,
        device=device,
        dtype=dtype,
    )
    buffer_storage[1:-1].copy_(
        torch.randn(
            (context.num_pages, 128, LAST_DIM),
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
    )
    output_storage = torch.full(
        (num_q_tokens + 2, HEAD_DIM), SENTINEL, device=device, dtype=dtype
    )
    return (
        plan,
        kv_score_input,
        ape,
        buffer_storage,
        output_storage,
    )


def _reference(
    pre_write_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: CompressorPrefillPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    final_buffer = pre_write_buffer.clone()
    write_words = plan.plan_w.view(torch.int32).view(-1, 2)
    valid_writes = write_words[:, 0] != -1
    if valid_writes.any():
        ragged_ids = (write_words[valid_writes, 0] & 0xFFFF).long()
        write_locs = write_words[valid_writes, 1].long()
        final_buffer.view(-1, LAST_DIM)[write_locs] = kv_score_input[ragged_ids]

    plan_words = plan.plan_c.view(torch.int32).view(-1, 4)
    valid_plans = plan_words[:, 0] != -1
    reference_output = torch.full(
        (plan.plan_c.shape[0], HEAD_DIM),
        SENTINEL,
        device=kv_score_input.device,
        dtype=kv_score_input.dtype,
    )
    for row in torch.nonzero(valid_plans).flatten().tolist():
        ragged_id = int(plan_words[row, 1] & 0xFFFF)
        buffer_len = (int(plan_words[row, 1]) >> 16) & 0xFFFF
        read_page = int(plan_words[row, 3])
        source = torch.empty(
            (128, LAST_DIM),
            device=kv_score_input.device,
            dtype=kv_score_input.dtype,
        )
        if buffer_len:
            source[:buffer_len] = pre_write_buffer[read_page, :buffer_len]
        source[buffer_len:] = kv_score_input[
            ragged_id - 127 + buffer_len : ragged_id + 1
        ]
        kv = source[:, :HEAD_DIM].float()
        score = source[:, HEAD_DIM:].float() + ape.float()
        reference_output[row] = (score.softmax(dim=0) * kv).sum(dim=0).to(
            kv_score_input.dtype
        )
    return reference_output, final_buffer


def _assert_no_alias(tensors: tuple[torch.Tensor, ...]) -> None:
    ranges = [
        (tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
        for tensor in tensors
    ]
    for left, first in enumerate(ranges):
        for second in ranges[left + 1 :]:
            assert first[1] <= second[0] or second[1] <= first[0]


def _assert_case(
    output: torch.Tensor,
    reference_output: torch.Tensor,
    buffer: torch.Tensor,
    reference_buffer: torch.Tensor,
    output_storage: torch.Tensor,
    buffer_storage: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    tolerance = 2.0e-5 if dtype == torch.float32 else 5.0e-2
    torch.testing.assert_close(output, reference_output, atol=tolerance, rtol=tolerance)
    assert torch.equal(buffer, reference_buffer)
    assert output_storage[0].eq(SENTINEL).all()
    assert output_storage[-1].eq(SENTINEL).all()
    assert buffer_storage[0].eq(SENTINEL).all()
    assert buffer_storage[-1].eq(SENTINEL).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA/HIP device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_c128_prefill_compiled_path_reuse_across_shapes(dtype: torch.dtype) -> None:
    for seq_extend_pairs in SHAPE_SEQUENCE:
        (
            plan,
            kv_score_input,
            ape,
            buffer_storage,
            output_storage,
        ) = _make_case(seq_extend_pairs, dtype)
        buffer = buffer_storage[1:-1]
        output = output_storage[1:-1]
        _assert_no_alias(
            (kv_score_input, ape, buffer, output, plan.plan_c, plan.plan_w)
        )
        pre_write_buffer = buffer.clone()

        def run() -> None:
            compress_forward(
                kv_score_buffer=buffer,
                kv_score_input=kv_score_input,
                ape=ape,
                plan=plan,
                head_dim=HEAD_DIM,
                compress_ratio=128,
                out=output,
            )

        run()
        torch.cuda.synchronize()
        reference_output, reference_buffer = _reference(
            pre_write_buffer, kv_score_input, ape, plan
        )
        _assert_case(
            output,
            reference_output,
            buffer,
            reference_buffer,
            output_storage,
            buffer_storage,
            dtype,
        )

        addresses = tuple(
            tensor.data_ptr()
            for tensor in (
                kv_score_input,
                ape,
                buffer,
                output,
                plan.plan_c,
                plan.plan_w,
            )
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        assert addresses == tuple(
            tensor.data_ptr()
            for tensor in (
                kv_score_input,
                ape,
                buffer,
                output,
                plan.plan_c,
                plan.plan_w,
            )
        )
        output.fill_(SENTINEL)
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        _assert_case(
            output,
            reference_output,
            buffer,
            reference_buffer,
            output_storage,
            buffer_storage,
            dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA/HIP device")
def test_c128_prefill_rejects_complex_dtype_without_coercion() -> None:
    (
        plan,
        kv_score_input,
        ape,
        buffer_storage,
        output_storage,
    ) = _make_case(SHAPE_SEQUENCE[0], torch.float32)
    complex_input = torch.complex(kv_score_input, kv_score_input)
    complex_ape = torch.complex(ape, ape)
    complex_buffer = torch.complex(buffer_storage[1:-1], buffer_storage[1:-1])
    complex_output = torch.complex(output_storage[1:-1], output_storage[1:-1])
    with pytest.raises(KeyError, match="torch.complex64"):
        compress_forward(
            kv_score_buffer=complex_buffer,
            kv_score_input=complex_input,
            ape=complex_ape,
            plan=plan,
            head_dim=HEAD_DIM,
            compress_ratio=128,
            out=complex_output,
        )
