import argparse
import json
from pathlib import Path

import torch


def _make_eagle(draft_token_num, device):
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    spec_input = EagleVerifyInput.create_idle_input(
        topk=4,
        spec_steps=3,
        num_verify_tokens=draft_token_num,
        device=device,
    )
    return spec_input


def _make_dflash(draft_token_num, device, custom_mask):
    from sglang.srt.speculative.dflash_info import DFlashVerifyInput

    return DFlashVerifyInput(
        draft_token=torch.empty((0,), dtype=torch.long, device=device),
        positions=torch.empty((0,), dtype=torch.int64, device=device),
        draft_token_num=draft_token_num,
        custom_mask=custom_mask,
    )


def _expected_metadata(req_to_token, req_pool_indices, paged_kernel_lens, draft_token_num):
    batch_size = len(req_pool_indices)
    qo_indptr = torch.arange(
        0,
        (batch_size + 1) * draft_token_num,
        step=draft_token_num,
        dtype=torch.int32,
        device=req_to_token.device,
    )
    cum_kv_seq_len = torch.zeros((batch_size + 1,), dtype=torch.int32, device=req_to_token.device)
    cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens + draft_token_num, dim=0)
    kv_indices = torch.cat(
        [
            req_to_token[
                int(req_pool_indices[index].item()),
                : int(paged_kernel_lens[index].item()) + draft_token_num,
            ]
            for index in range(batch_size)
        ],
        dim=0,
    ).to(torch.int32)
    return qo_indptr, cum_kv_seq_len, kv_indices


def _run_case(spec_input, batch_size, draft_token_num, device):
    req_pool_indices = torch.arange(batch_size, device=device)
    paged_kernel_lens = torch.full((batch_size,), 10, dtype=torch.int32, device=device)
    paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
    req_to_token = torch.arange(
        batch_size * 64, dtype=torch.int32, device=device
    ).reshape(batch_size, 64)
    outputs = spec_input.generate_attn_arg_prefill(
        req_pool_indices,
        paged_kernel_lens,
        paged_kernel_lens_sum,
        req_to_token,
    )
    expected_qo_indptr, expected_cum_kv_seq_len, expected_kv_indices = _expected_metadata(
        req_to_token, req_pool_indices, paged_kernel_lens, draft_token_num
    )
    expected_mask_numel = (
        paged_kernel_lens_sum * draft_token_num
        + (draft_token_num**2) * batch_size
    )
    return {
        "outputs": outputs,
        "expected_qo_indptr": expected_qo_indptr,
        "expected_cum_kv_seq_len": expected_cum_kv_seq_len,
        "expected_kv_indices": expected_kv_indices,
        "expected_mask_numel": expected_mask_numel,
    }


def _record_case(spec_input, batch_size, draft_token_num, device, expected_mask):
    result = _run_case(spec_input, batch_size, draft_token_num, device)
    kv_indices, cum_kv_seq_len, qo_indptr, mask = result["outputs"]
    expected_mask_numel = result["expected_mask_numel"]
    mask_match = torch.equal(mask[:expected_mask_numel], expected_mask[:expected_mask_numel])
    return {
        "batch_size": batch_size,
        "draft_token_num": draft_token_num,
        "expected_mask_numel": expected_mask_numel,
        "returned_mask_numel": int(mask.numel()),
        "storage_mask_numel": int(spec_input.custom_mask.numel()),
        "mask_exact": int(mask.numel()) == expected_mask_numel,
        "mask_content_match": bool(mask_match),
        "qo_indptr_match": bool(torch.equal(qo_indptr, result["expected_qo_indptr"])),
        "cum_kv_seq_len_match": bool(
            torch.equal(cum_kv_seq_len, result["expected_cum_kv_seq_len"])
        ),
        "kv_indices_match": bool(torch.equal(kv_indices, result["expected_kv_indices"])),
    }


def _make_pattern_mask(draft_token_num, batch_size, device):
    mask_numel = draft_token_num * batch_size * (10 + draft_token_num)
    return (torch.arange(mask_numel, device=device) % 3 == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    draft_token_nums = [4, 8, 16]
    batch_sizes = [2, 8, 4, 16, 1, 32, 2]
    results = {"label": args.label, "device": device, "cases": []}

    for draft_token_num in draft_token_nums:
        eagle = _make_eagle(draft_token_num, device)
        dflash = _make_dflash(
            draft_token_num,
            device,
            torch.full((0,), True, dtype=torch.bool, device=device),
        )
        for batch_size in batch_sizes:
            results["cases"].append(
                _record_case(
                    eagle,
                    batch_size,
                    draft_token_num,
                    device,
                    torch.ones(
                        (
                            draft_token_num
                            * (10 * batch_size + draft_token_num * batch_size)
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )
            results["cases"].append(
                _record_case(
                    dflash,
                    batch_size,
                    draft_token_num,
                    device,
                    torch.ones(
                        (
                            draft_token_num
                            * (10 * batch_size + draft_token_num * batch_size)
                        ),
                        dtype=torch.bool,
                        device=device,
                    ),
                )
            )

        shrink_batch_sizes = [32, 16, 4, 2, 1]
        max_mask = _make_pattern_mask(draft_token_num, 32, device)
        eagle = _make_eagle(draft_token_num, device)
        eagle.custom_mask = max_mask.clone()
        dflash = _make_dflash(draft_token_num, device, max_mask.clone())
        for batch_size in shrink_batch_sizes:
            expected_mask = _make_pattern_mask(draft_token_num, batch_size, device)
            results["cases"].append(
                _record_case(
                    eagle,
                    batch_size,
                    draft_token_num,
                    device,
                    expected_mask,
                )
            )
            results["cases"].append(
                _record_case(
                    dflash,
                    batch_size,
                    draft_token_num,
                    device,
                    expected_mask,
                )
            )

    results["all_cases_pass"] = all(
        case["mask_exact"]
        and case["mask_content_match"]
        and case["qo_indptr_match"]
        and case["cum_kv_seq_len_match"]
        and case["kv_indices_match"]
        for case in results["cases"]
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
