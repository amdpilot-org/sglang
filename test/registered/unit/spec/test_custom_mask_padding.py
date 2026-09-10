import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _make_eagle(device):
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    return EagleVerifyInput.create_idle_input(
        topk=2,
        spec_steps=3,
        num_verify_tokens=8,
        device=device,
    )


def _make_dflash(device):
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    return DFlashVerifyInput(
        draft_token=torch.empty(0, dtype=torch.long, device=device),
        positions=torch.empty(0, dtype=torch.int64, device=device),
        draft_token_num=8,
        custom_mask=_initial_mask(device),
    )


def _initial_mask(device):
    return torch.tensor(
        [index % 2 == 0 for index in range(64)],
        dtype=torch.bool,
        device=device,
    )


def _call(spec_input, batch_size, device):
    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    paged_kernel_lens = torch.full(
        (batch_size,), 10, dtype=torch.int32, device=device
    )
    paged_kernel_lens_sum = 10 * batch_size
    req_to_token = torch.arange(
        32 * 20, dtype=torch.int32, device=device
    ).reshape(32, 20)
    kv_indices, cum_kv_seq_len, qo_indptr, mask = spec_input.generate_attn_arg_prefill(
        req_pool_indices,
        paged_kernel_lens,
        paged_kernel_lens_sum,
        req_to_token,
    )
    expected_kv_indices = torch.cat(
        [req_to_token[index, :18] for index in range(batch_size)]
    )
    expected_cum_kv_seq_len = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=device
    )
    expected_cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens + 8, dim=0)
    expected_qo_indptr = torch.arange(
        0, (batch_size + 1) * 8, 8, dtype=torch.int32, device=device
    )
    return (
        kv_indices,
        expected_kv_indices,
        cum_kv_seq_len,
        expected_cum_kv_seq_len,
        qo_indptr,
        expected_qo_indptr,
        mask,
    )


def _expected_mask(reference, mask_numel, device):
    if reference.numel() < mask_numel:
        return torch.cat(
            [
                reference,
                torch.full(
                    (mask_numel - reference.numel(),),
                    True,
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )
    return reference[:mask_numel].clone()


def _run_alternating_trees(spec_input, device):
    reference = _initial_mask(device)
    peak_mask_numel = 0
    for batch_size in [1, 8, 1, 16, 1, 32, 1]:
        (
            kv_indices,
            expected_kv_indices,
            cum_kv_seq_len,
            expected_cum_kv_seq_len,
            qo_indptr,
            expected_qo_indptr,
            mask,
        ) = _call(spec_input, batch_size, device)
        mask_numel = 144 * batch_size
        expected_mask = _expected_mask(reference, mask_numel, device)

        assert mask.numel() == mask_numel
        assert torch.equal(mask, expected_mask)
        assert torch.equal(kv_indices, expected_kv_indices)
        assert torch.equal(cum_kv_seq_len, expected_cum_kv_seq_len)
        assert torch.equal(qo_indptr, expected_qo_indptr)

        reference = expected_mask
        peak_mask_numel = max(peak_mask_numel, mask_numel)

    assert spec_input.custom_mask.numel() <= peak_mask_numel


def test_eagle_preserves_mask_content_while_bounded():
    device = "cuda"
    spec_input = _make_eagle(device)
    spec_input.custom_mask = _initial_mask(device)
    _run_alternating_trees(spec_input, device)


def test_dflash_preserves_mask_content_while_bounded():
    device = "cuda"
    spec_input = _make_dflash(device)
    _run_alternating_trees(spec_input, device)
