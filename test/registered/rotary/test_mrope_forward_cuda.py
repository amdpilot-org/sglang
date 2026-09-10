import pytest
import torch

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
TOKENS = 33
NUM_Q_HEADS = 4
NUM_KV_HEADS = 2


@pytest.fixture(autouse=True)
def server_args():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


def _build_rope(mrope_section, mrope_interleaved):
    return MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=MAX_POSITIONS,
        base=BASE,
        is_neox_style=True,
        dtype=DTYPE,
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    ).to(DEVICE)


def _positions():
    return torch.stack(
        [
            torch.arange(TOKENS, device=DEVICE) % 7,
            torch.arange(TOKENS, device=DEVICE) % 5 + 3,
            torch.arange(TOKENS, device=DEVICE) % 3 + 11,
        ]
    ).contiguous()


def _reference(rope, positions, values):
    half = ROTARY_DIM // 2
    values = values.view(TOKENS, -1, HEAD_DIM)
    cache = rope.cos_sin_cache[positions].float()
    if positions.ndim == 1:
        cos = cache[..., :half]
        sin = cache[..., half:]
    else:
        axis_index = rope.axis_map.view(1, half, 1).expand(TOKENS, half, 1)
        cos = torch.gather(
            cache[..., :half].permute(1, 2, 0), 2, axis_index
        ).squeeze(-1)
        sin = torch.gather(
            cache[..., half:].permute(1, 2, 0), 2, axis_index
        ).squeeze(-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    first = values[..., :half].float()
    second = values[..., half:ROTARY_DIM].float()
    output = values.clone().float()
    output[..., :half] = first * cos - second * sin
    output[..., half:ROTARY_DIM] = second * cos + first * sin
    return output.reshape(TOKENS, -1).to(values.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA or HIP GPU")
@pytest.mark.parametrize(
    "mrope_section,mrope_interleaved",
    [
        ([11, 11, 10], True),
        ([12, 10, 10], False),
    ],
)
def test_mrope_forward_cuda_matches_independent_reference(
    mrope_section,
    mrope_interleaved,
):
    torch.manual_seed(0)
    rope = _build_rope(mrope_section, mrope_interleaved)
    positions = _positions()
    query = torch.randn(
        TOKENS, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
    )
    key = torch.randn(TOKENS, NUM_KV_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)

    query_1d, key_1d = rope.forward_cuda(
        positions[0].contiguous(), query.clone(), key.clone()
    )
    torch.testing.assert_close(
        query_1d.float(),
        _reference(rope, positions[0], query).float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        key_1d.float(),
        _reference(rope, positions[0], key).float(),
        atol=2e-2,
        rtol=2e-2,
    )

    assert not torch.equal(positions[0], positions[1])
    assert not torch.equal(positions[1], positions[2])
    query_3d, key_3d = rope.forward_cuda(positions, query.clone(), key.clone())
    torch.testing.assert_close(
        query_3d.float(),
        _reference(rope, positions, query).float(),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        key_3d.float(),
        _reference(rope, positions, key).float(),
        atol=2e-2,
        rtol=2e-2,
    )
