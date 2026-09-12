import random

import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1


def run_case(num_experts, block_e, seed):
    rng = random.Random(seed)
    padded_counts = [rng.randrange(0, 9) * block_e for _ in range(num_experts)]
    valid_counts = [rng.randrange(0, count + 1) if count else 0 for count in padded_counts]
    padded = torch.tensor(padded_counts, dtype=torch.int32, device="cuda")
    valid = torch.tensor(valid_counts, dtype=torch.int32, device="cuda")
    starts = torch.full_like(padded, -31337)
    guard = 2 * block_e
    raw = torch.full((sum(padded_counts) + 2 * guard,), -424242, dtype=torch.int32, device="cuda")
    output = raw[guard:-guard]
    handle = _fwd_kernel_ep_scatter_1[(num_experts,)](
        padded,
        valid,
        starts,
        output,
        num_experts=num_experts,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        num_warps=8,
    )
    torch.cuda.synchronize()
    expected_starts = (torch.cumsum(padded, 0) - padded).to(torch.int32)
    expected_output = torch.cat([
        torch.cat((torch.full((v,), i, dtype=torch.int32), torch.full((p-v,), -1, dtype=torch.int32)))
        for i, (p, v) in enumerate(zip(padded_counts, valid_counts))
    ]).to("cuda")
    torch.testing.assert_close(starts, expected_starts, rtol=0, atol=0)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    assert torch.all(raw[:guard] == -424242) and torch.all(raw[-guard:] == -424242)
    assert "tt.load %cur_expert_start" not in handle.asm["ttir"]
    return handle


last = None
cases = [(2, 128), (7, 64), (24, 32), (33, 16), (64, 128), (129, 32)]
for repetition in range(20):
    for case_number, (num_experts, block_e) in enumerate(cases):
        last = run_case(num_experts, block_e, repetition * 100 + case_number)
print(f"passed {20 * len(cases)} guarded randomized launches on {torch.cuda.get_device_name(0)}")
print("architecture", torch.cuda.get_device_properties(0).gcnArchName)
print("ttir reload marker", "tt.load %cur_expert_start" in last.asm["ttir"])
open("/job/review-evidence/candidate_kernel.ttir", "w").write(last.asm["ttir"])
open("/job/review-evidence/candidate_kernel.amdgcn", "w").write(last.asm["amdgcn"])
