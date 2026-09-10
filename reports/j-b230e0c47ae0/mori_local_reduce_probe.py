import json
import os
import torch
from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

assert torch.cuda.is_available()
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
assert props.major == 9 and props.minor == 4, (props.major, props.minor)
assert props.warp_size == 64, props.warp_size

capacities = [32, 64, 128, 256]
topk = 8
model_dim = 128
num_experts = 8
local_experts = 2
sentinel = -7777.0
generator = torch.Generator(device=device)
generator.manual_seed(27194)
results = []
for capacity in capacities:
    token_num = capacity
    expert_mask = torch.zeros(num_experts, dtype=torch.int32, device=device)
    expert_mask[:local_experts] = 1
    topk_ids = torch.randint(0, num_experts, (token_num, topk), device=device, dtype=torch.int32, generator=generator)
    topk_ids[:, 0] = torch.arange(token_num, device=device, dtype=torch.int32) % local_experts

    guard_elements = 128
    flat = torch.full((guard_elements + token_num * topk * model_dim + guard_elements,), sentinel, dtype=torch.bfloat16, device=device)
    target = flat[guard_elements:guard_elements + token_num * topk * model_dim]
    token_index = torch.arange(token_num, device=device).view(token_num, 1, 1)
    slot_index = torch.arange(topk, device=device).view(1, topk, 1)
    target.view(token_num, topk, model_dim).copy_((token_index + slot_index + 1).to(torch.bfloat16))
    left_guard_before = flat[:guard_elements].view(torch.uint16).to(torch.int64).sum().item()
    right_guard_before = flat[-guard_elements:].view(torch.uint16).to(torch.int64).sum().item()

    target.view(token_num, topk, model_dim)[expert_mask[topk_ids] == 0] = sentinel
    target_hash_before = target.view(torch.uint16).to(torch.int64).sum().item()
    out = torch.full((token_num, model_dim), sentinel, dtype=torch.bfloat16, device=device)
    _run_moe_reduction(
        target,
        out,
        token_num,
        topk,
        model_dim,
        expert_mask=expert_mask,
        topk_ids=topk_ids,
    )
    torch.cuda.synchronize()

    target_cpu = target.view(token_num, topk, model_dim).cpu()
    ids_cpu = topk_ids.cpu()
    mask_cpu = expert_mask.cpu()
    valid = mask_cpu[ids_cpu].bool()
    target_numpy = target_cpu.float().numpy()
    valid_numpy = valid.numpy()
    reference_numpy = (target_numpy * valid_numpy[:, :, None]).sum(axis=1)
    ref = torch.from_numpy(reference_numpy)
    actual = out.cpu()
    diff = (actual.float() - ref).abs()
    invalid_slots = ~valid
    invalid_target = target_cpu[invalid_slots]
    left_guard_after = flat[:guard_elements].view(torch.uint16).to(torch.int64).sum().item()
    right_guard_after = flat[-guard_elements:].view(torch.uint16).to(torch.int64).sum().item()
    result = {
        "capacity": capacity,
        "token_num": token_num,
        "topk": topk,
        "model_dim": model_dim,
        "valid_slots": int(valid.sum().item()),
        "invalid_slots": int(invalid_slots.sum().item()),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "exact_bf16": bool(torch.equal(actual, ref.to(torch.bfloat16))),
        "all_finite": bool(torch.isfinite(actual).all().item()),
        "target_hash_before": target_hash_before,
        "target_hash_after": target.view(torch.uint16).to(torch.int64).sum().item(),
        "left_guard_before": left_guard_before,
        "left_guard_after": left_guard_after,
        "right_guard_before": right_guard_before,
        "right_guard_after": right_guard_after,
        "invalid_slot_sentinel_min": float(invalid_target.min().item()) if invalid_target.numel() else None,
        "invalid_slot_sentinel_max": float(invalid_target.max().item()) if invalid_target.numel() else None,
        "out_first_row": [float(x) for x in actual[0, :8].float()],
        "ref_first_row": [float(x) for x in ref[0, :8]],
    }
    results.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)

summary = {
    "gpu": {"name": props.name, "major": props.major, "minor": props.minor, "warp_size": props.warp_size},
    "aiter_source": "/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py",
    "reduction_helper": "_run_moe_reduction",
    "capacities": capacities,
    "all_exact": all(x["exact_bf16"] for x in results),
    "all_guards_unchanged": all(x["left_guard_before"] == x["left_guard_after"] and x["right_guard_before"] == x["right_guard_after"] for x in results),
    "all_invalid_slots_sentinel": all(x["invalid_slot_sentinel_min"] == torch.tensor(sentinel, dtype=torch.bfloat16).item() and x["invalid_slot_sentinel_max"] == torch.tensor(sentinel, dtype=torch.bfloat16).item() for x in results),
    "results": results,
}
outpath = os.environ.get("MORI_REDUCE_RESULTS", "/tmp/sglang-cache-JOB_ID/mori_reduce_results.json")
with open(outpath, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps({k: v for k, v in summary.items() if k != "results"}, sort_keys=True))
