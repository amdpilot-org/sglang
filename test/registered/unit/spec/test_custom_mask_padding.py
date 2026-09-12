import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _make_eagle(custom_mask):
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    spec_input = EagleVerifyInput.create_idle_input(
        topk=2,
        spec_steps=2,
        num_verify_tokens=6,
        device=custom_mask.device,
    )
    spec_input.custom_mask = custom_mask
    return spec_input


def _make_dflash(custom_mask):
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    return DFlashVerifyInput(
        draft_token=torch.empty(0, dtype=torch.long, device=custom_mask.device),
        positions=torch.empty(0, dtype=torch.int64, device=custom_mask.device),
        draft_token_num=3,
        custom_mask=custom_mask,
    )


def _generate(spec_input, batch_size, seq_len=5):
    device = spec_input.custom_mask.device
    req_pool_indices = torch.arange(batch_size, device=device)
    paged_kernel_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    req_to_token = torch.zeros(
        (batch_size, seq_len + 16), dtype=torch.int32, device=device
    )
    mask_numel = (
        int(paged_kernel_lens.sum().item()) * spec_input.draft_token_num
        + spec_input.draft_token_num**2 * batch_size
    )
    *_, custom_mask = spec_input.generate_attn_arg_prefill(
        req_pool_indices,
        paged_kernel_lens,
        int(paged_kernel_lens.sum().item()),
        req_to_token,
    )
    return mask_numel, custom_mask


@pytest.mark.parametrize("make_spec_input", [_make_eagle, _make_dflash])
def test_custom_mask_padding_does_not_mutate_retained_mask(make_spec_input):
    original = torch.tensor([False, True, False], dtype=torch.bool, device="cuda")
    spec_input = make_spec_input(original)

    mask_numel, padded = _generate(spec_input, batch_size=2)

    assert padded.numel() == mask_numel
    torch.testing.assert_close(padded[: original.numel()], original)
    assert padded[original.numel() :].all()
    assert spec_input.custom_mask is original


@pytest.mark.parametrize("make_spec_input", [_make_eagle, _make_dflash])
def test_custom_mask_shrink_returns_exact_view(make_spec_input):
    original = torch.arange(200, device="cuda").remainder(3).bool()
    spec_input = make_spec_input(original)

    mask_numel, shrunk = _generate(spec_input, batch_size=1)

    assert shrunk.numel() == mask_numel
    torch.testing.assert_close(shrunk, original[:mask_numel])
    assert shrunk.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert spec_input.custom_mask is original


def test_dflash_none_custom_mask_is_unchanged():
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    spec_input = DFlashVerifyInput(
        draft_token=torch.empty(0, dtype=torch.long, device="cuda"),
        positions=torch.empty(0, dtype=torch.int64, device="cuda"),
        draft_token_num=3,
        custom_mask=None,
    )
    req_pool_indices = torch.arange(1, device="cuda")
    paged_kernel_lens = torch.tensor([5], dtype=torch.int32, device="cuda")
    req_to_token = torch.zeros((1, 21), dtype=torch.int32, device="cuda")

    *_, custom_mask = spec_input.generate_attn_arg_prefill(
        req_pool_indices, paged_kernel_lens, 5, req_to_token
    )

    assert custom_mask is None
