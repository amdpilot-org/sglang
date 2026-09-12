"""Direct GPU regression for the DSV4 c128 prefill-plan ragged-id bound."""

from __future__ import annotations

import argparse

import torch

from sglang.test.kernels.deepseek_v4.common import make_paged_context


SHAPES = {
    "uniform_4x96": [4] * 96,
    "ragged_4x72_3x24": [4] * 72 + [3] * 24,
    "ragged_3x104_2x24": [3] * 104 + [2] * 24,
}


def maximum_ragged_id(plan) -> int:
    plan_w_ids = plan.plan_w.view(torch.uint32).reshape(-1, 2)[:, 0].to(torch.int64)
    # PlanC's ragged_id is the low uint16 of its second uint32 word. Cast before
    # masking because PyTorch/ROCm does not implement uint32 bitwise_and.
    plan_c_ids = (
        plan.plan_c.view(torch.uint32)
        .reshape(-1, 4)[:, 1]
        .to(torch.int64)
        .bitwise_and(0xFFFF)
    )
    maxima = []
    for values, invalid in ((plan_w_ids, 0xFFFFFFFF), (plan_c_ids, 0xFFFF)):
        valid = values[values != invalid]
        if valid.numel():
            maxima.append(int(valid.max()))
    return max(maxima, default=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    total_bad = 0
    for name, extend_values in SHAPES.items():
        batch_size = len(extend_values)
        context = make_paged_context(
            bs=batch_size,
            compress_ratio=128,
            ring_size=128,
            num_reqs_capacity=batch_size,
            max_tokens_per_req=1024,
            num_swa_pages_per_req=4,
        )
        extend_lens = torch.tensor(extend_values, dtype=torch.int64, device="cuda")
        seq_lens = extend_lens + 512
        real_rows = sum(extend_values)
        bad = 0
        max_observed = -1
        for _ in range(args.iterations):
            plan = context.make_prefill_plan(
                seq_lens, extend_lens, num_q_tokens=real_rows
            )
            observed = maximum_ragged_id(plan)
            max_observed = max(max_observed, observed)
            bad += observed >= real_rows
        torch.cuda.synchronize()
        total_bad += bad
        print(
            f"{name} rows={real_rows} bad={bad} calls={args.iterations} "
            f"max_observed={max_observed} bound={real_rows - 1}"
        )

    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)} total_bad={total_bad}"
    )
    if total_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
