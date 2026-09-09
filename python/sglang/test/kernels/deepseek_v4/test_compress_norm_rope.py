import pytest
import torch

from sglang.kernels.ops.attention.dsv4.compress import (
    _jit_compress_norm_rope_module,
)
from sglang.kernels.jit.utils.common import is_hip_runtime


HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
GROUP_COUNT = NOPE_DIM // 64
SENTINEL = 0xA5


def _require_gfx942():
    if not is_hip_runtime():
        pytest.skip("This regression targets the ROCm software conversion path.")
    if "gfx942" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("This regression targets MI300X gfx942.")


def _gamma():
    nope_pattern = torch.tensor(
        [64, 128, 160, 224, -64, -128, -160, -224], dtype=torch.float32
    )
    rope_pattern = torch.tensor([1, -2, 0.5, -0.75], dtype=torch.float32).repeat(16)
    gamma_reference = torch.empty(HEAD_DIM, dtype=torch.float32)
    for group_index in range(GROUP_COUNT):
        gamma_reference[group_index * 64 : (group_index + 1) * 64] = (
            nope_pattern * (2.0 ** -group_index)
        ).repeat(8)
    gamma_reference[NOPE_DIM:] = rope_pattern
    return gamma_reference.to(torch.bfloat16), gamma_reference


def _expected_nope_bytes(sign):
    nope_pattern = torch.tensor(
        [64, 128, 160, 224, -64, -128, -160, -224], dtype=torch.float32
    )
    if sign == -1:
        nope_pattern = -nope_pattern
    group_bytes = nope_pattern.to(torch.float8_e4m3fnuz).view(torch.uint8).repeat(8)
    return group_bytes.repeat(GROUP_COUNT)


def _expected_cache(page_size, out_loc, gamma_reference):
    page_bytes = ((584 * page_size + 575) // 576) * 576
    expected = torch.full((3, page_bytes), SENTINEL, dtype=torch.uint8)
    positive_nope = _expected_nope_bytes(1)
    negative_nope = _expected_nope_bytes(-1)
    positive_rope = gamma_reference[NOPE_DIM:].to(torch.bfloat16).view(torch.uint8)
    negative_rope = (-gamma_reference[NOPE_DIM:]).to(torch.bfloat16).view(torch.uint8)

    for row_index, sign in enumerate((1, -1)):
        location = int(out_loc[row_index].item())
        page = location // page_size
        slot = location % page_size
        value_start = slot * 576
        scale_start = page_size * 576 + slot * 8
        expected[page, value_start : value_start + NOPE_DIM] = (
            positive_nope if sign == 1 else negative_nope
        )
        expected[page, value_start + NOPE_DIM : value_start + 576] = (
            positive_rope if sign == 1 else negative_rope
        )
        expected[page, scale_start : scale_start + 7] = torch.tensor(
            [127, 126, 125, 124, 123, 122, 121], dtype=torch.uint8
        )
    return expected


@pytest.mark.parametrize(("page_size", "ratio"), [(64, 4), (2, 128)])
def test_compress_norm_rope_flashmla_full_cache_gfx942(page_size, ratio):
    _require_gfx942()
    module = _jit_compress_norm_rope_module(
        dtype=torch.bfloat16,
        head_dim=HEAD_DIM,
        rope_dim=ROPE_DIM,
        page_size=page_size,
        bf16_store=False,
    )

    input_tensor = torch.ones((3, HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    input_tensor[1] = -1
    gamma, gamma_reference = _gamma()
    gamma = gamma.cuda()
    freqs_cis = torch.zeros((ratio + 1, ROPE_DIM), dtype=torch.float32, device="cuda")
    freqs_cis[:, 0::2] = 1
    out_loc = torch.tensor(
        [page_size - 1, page_size + 1, 2 * page_size],
        dtype=torch.int64,
        device="cuda",
    )
    plan_int32 = torch.zeros((3, 4), dtype=torch.int32, device="cuda")
    plan_int32[:, 0] = torch.tensor(
        [ratio, 2 * ratio, ratio + 1], dtype=torch.int32, device="cuda"
    )
    plan = plan_int32.view(torch.uint8)
    page_bytes = ((584 * page_size + 575) // 576) * 576
    cache = torch.full((3, page_bytes), SENTINEL, dtype=torch.uint8, device="cuda")

    module.forward(
        input_tensor,
        plan,
        gamma,
        0.0,
        freqs_cis,
        out_loc,
        cache,
        True,
        ratio,
    )
    torch.cuda.synchronize()

    expected = _expected_cache(page_size, out_loc.cpu(), gamma_reference.cpu()).cuda()
    mismatch_count = int((cache != expected).sum().item())
    assert mismatch_count == 0, (
        f"FlashMLA cache mismatch for page_size={page_size}, ratio={ratio}: "
        f"{mismatch_count} bytes"
    )

    for row_index, sign in enumerate((1, -1)):
        location = int(out_loc[row_index].item())
        page = location // page_size
        slot = location % page_size
        value_start = slot * 576
        scale_start = page_size * 576 + slot * 8
        for group_index in range(GROUP_COUNT):
            actual_bytes = cache[
                page,
                value_start + group_index * 64 : value_start + (group_index + 1) * 64,
            ].view(torch.float8_e4m3fnuz)
            decoded = actual_bytes.to(torch.float32) * (2.0 ** -group_index)
            reference = sign * gamma_reference[
                group_index * 64 : (group_index + 1) * 64
            ]
            assert torch.equal(decoded, reference.cuda())
        assert torch.equal(
            cache[page, scale_start : scale_start + 7],
            torch.tensor([127, 126, 125, 124, 123, 122, 121], dtype=torch.uint8, device="cuda"),
        )
