import time

import pytest
import torch

from sglang.kernels.ops.attention.rotary_triton import _triton_mrope_forward_fused
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")
register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
DTYPE = torch.bfloat16
HEAD_DIM = 256
ROTARY_DIM = 64
MAX_POSITIONS = 512
BASE = 10_000_000
MAX_TOKENS = 64
NUM_Q_HEADS = 3
NUM_KV_HEADS = 1
GUARD_ELEMENTS = 8
POSITION_SENTINEL = 511
Q_SENTINEL = 123.0
K_SENTINEL = 124.0
TOKEN_SEQUENCE = (1, 17, 33, 64)


@pytest.fixture(autouse=True)
def server_args():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def _guarded_buffer(rows, columns, sentinel):
    flat = torch.full(
        (GUARD_ELEMENTS + rows * columns + GUARD_ELEMENTS,),
        sentinel,
        device=DEVICE,
        dtype=torch.float32 if rows == 3 else DTYPE,
    )
    buffer = flat[GUARD_ELEMENTS : GUARD_ELEMENTS + rows * columns].view(rows, columns)
    return flat, buffer


def _build_rope():
    return MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITIONS,
        base=BASE,
        is_neox_style=True,
        dtype=DTYPE,
        mrope_section=[11, 11, 10],
        mrope_interleaved=True,
    ).to(DEVICE)


def _reference(rope, positions, values, num_tokens):
    half_rotary = ROTARY_DIM // 2
    values = values.view(num_tokens, -1, HEAD_DIM)
    cache = rope.cos_sin_cache[positions].float()
    axes = torch.zeros(half_rotary, device=DEVICE, dtype=torch.int64)
    for axis, section in enumerate(rope.mrope_section[1:], start=1):
        lanes = torch.arange(axis, min(3 * section, half_rotary), 3, device=DEVICE)
        axes[lanes] = axis
    axis_index = axes.view(1, half_rotary, 1).expand(num_tokens, half_rotary, 1)
    cos = torch.gather(cache[..., :half_rotary].permute(1, 2, 0), 2, axis_index)
    sin = torch.gather(cache[..., half_rotary:].permute(1, 2, 0), 2, axis_index)
    cos = cos.squeeze(-1).unsqueeze(1)
    sin = sin.squeeze(-1).unsqueeze(1)
    first = values[..., :half_rotary].float()
    second = values[..., half_rotary:ROTARY_DIM].float()
    output = values.clone().float()
    output[..., :half_rotary] = first * cos - second * sin
    output[..., half_rotary:ROTARY_DIM] = second * cos + first * sin
    return output.reshape(num_tokens, -1).to(values.dtype)


def _cache_count():
    return sum(
        len(device_cache[0])
        for device_cache in _triton_mrope_forward_fused.device_caches.values()
    )


def _fill_positions(position_buffer, num_tokens):
    position_buffer.fill_(POSITION_SENTINEL)
    position_buffer[:, :num_tokens] = torch.stack(
        [
            torch.arange(num_tokens, device=DEVICE) % 7,
            torch.arange(num_tokens, device=DEVICE) % 5 + 3,
            torch.arange(num_tokens, device=DEVICE) % 3 + 11,
        ]
    )


def _check_storage(flat, buffer, num_tokens, sentinel):
    rows, columns = buffer.shape
    torch.testing.assert_close(
        flat[:GUARD_ELEMENTS],
        torch.full_like(flat[:GUARD_ELEMENTS], sentinel),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        flat[GUARD_ELEMENTS + rows * columns :],
        torch.full_like(
            flat[GUARD_ELEMENTS + rows * columns :], sentinel
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        buffer[num_tokens:],
        torch.full_like(buffer[num_tokens:], sentinel),
        atol=0,
        rtol=0,
    )


def _check_position_storage(flat, buffer, num_tokens):
    torch.testing.assert_close(
        flat[:GUARD_ELEMENTS],
        torch.full_like(flat[:GUARD_ELEMENTS], POSITION_SENTINEL),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        flat[GUARD_ELEMENTS + buffer.numel() :],
        torch.full_like(
            flat[GUARD_ELEMENTS + buffer.numel() :], POSITION_SENTINEL
        ),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        buffer[:, num_tokens:],
        torch.full_like(buffer[:, num_tokens:], POSITION_SENTINEL),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or HIP GPU")
def test_triton_mrope_reuses_compiled_path_across_shape_sequence():
    torch.manual_seed(35345)
    rope = _build_rope()
    position_flat, position_buffer = _guarded_buffer(3, MAX_TOKENS, POSITION_SENTINEL)
    position_flat = position_flat.to(torch.int64)
    position_buffer = position_flat[
        GUARD_ELEMENTS : GUARD_ELEMENTS + 3 * MAX_TOKENS
    ].view(3, MAX_TOKENS)
    q_flat, q_buffer = _guarded_buffer(
        MAX_TOKENS, NUM_Q_HEADS * HEAD_DIM, Q_SENTINEL
    )
    k_flat, k_buffer = _guarded_buffer(
        MAX_TOKENS, NUM_KV_HEADS * HEAD_DIM, K_SENTINEL
    )

    static_addresses = {
        "positions": position_buffer.data_ptr(),
        "query": q_buffer.data_ptr(),
        "key": k_buffer.data_ptr(),
    }
    initial_cache_count = _cache_count()
    cache_count_after_first = None

    for pass_name in ("first_sequence", "warm_sequence"):
        for case_index, num_tokens in enumerate(TOKEN_SEQUENCE):
            torch.manual_seed(35345 + case_index)
            _fill_positions(position_buffer, num_tokens)
            positions = position_buffer[:, :num_tokens]
            q_buffer.fill_(Q_SENTINEL)
            k_buffer.fill_(K_SENTINEL)
            q_buffer[:num_tokens].copy_(
                torch.randn(
                    num_tokens,
                    NUM_Q_HEADS * HEAD_DIM,
                    device=DEVICE,
                    dtype=DTYPE,
                )
            )
            k_buffer[:num_tokens].copy_(
                torch.randn(
                    num_tokens,
                    NUM_KV_HEADS * HEAD_DIM,
                    device=DEVICE,
                    dtype=DTYPE,
                )
            )
            query = q_buffer[:num_tokens]
            key = k_buffer[:num_tokens]
            query_reference = query.clone()
            key_reference = key.clone()

            started = time.perf_counter()
            query_output, key_output = rope.forward_cuda(positions, query, key)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            assert elapsed >= 0.0

            assert query_output is query
            assert key_output is key
            assert query.dtype is DTYPE
            assert key.dtype is DTYPE
            assert rope.cos_sin_cache.dtype is DTYPE
            assert positions.data_ptr() == static_addresses["positions"]
            assert query.data_ptr() == static_addresses["query"]
            assert key.data_ptr() == static_addresses["key"]

            torch.testing.assert_close(
                query_output.float(),
                _reference(rope, positions, query_reference, num_tokens).float(),
                atol=2e-2,
                rtol=2e-2,
            )
            torch.testing.assert_close(
                key_output.float(),
                _reference(rope, positions, key_reference, num_tokens).float(),
                atol=2e-2,
                rtol=2e-2,
            )
            _check_storage(q_flat, q_buffer, num_tokens, Q_SENTINEL)
            _check_storage(k_flat, k_buffer, num_tokens, K_SENTINEL)
            _check_position_storage(position_flat, position_buffer, num_tokens)

            current_cache_count = _cache_count()
            if pass_name == "first_sequence" and case_index == 0:
                cache_count_after_first = current_cache_count
            assert current_cache_count == cache_count_after_first

    assert cache_count_after_first >= initial_cache_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or HIP GPU")
def test_rejects_unsupported_position_rank():
    rope = _build_rope()
    positions = torch.zeros(1, 3, MAX_TOKENS, device=DEVICE, dtype=torch.int64)
    query = torch.zeros(1, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
    key = torch.zeros(1, NUM_KV_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
    with pytest.raises(AssertionError):
        rope.forward_cuda(positions, query, key)
