"""Issue #33693 shape-path reproduction against the checked-out implementation."""

from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe


class ReachedKernelPreparation(RuntimeError):
    pass


def reaches_kernel_preparation(*args, **kwargs):
    raise ReachedKernelPreparation


def check(label, hidden_k, weight_k, *, use_fp8=False, use_int4=False, expect_pass=False):
    device = "cuda"
    hidden = torch.empty((1, hidden_k), dtype=torch.bfloat16, device=device)
    # Only the dimensions inspected before kernel preparation matter here.
    w1 = torch.empty((1, 2, weight_k), dtype=torch.int8, device=device)
    w2 = torch.empty((1, hidden_k, 1), dtype=torch.int8, device=device)
    topk_weights = torch.ones((1, 1), dtype=torch.float32, device=device)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32, device=device)

    outcome = None
    with patch.object(fused_moe, "_prepare_fused_moe_run", reaches_kernel_preparation):
        try:
            fused_moe.fused_experts_impl(
                hidden,
                w1,
                w2,
                topk_weights,
                topk_ids,
                use_fp8_w8a8=use_fp8,
                use_int4_w4a16=use_int4,
            )
        except AssertionError as exc:
            outcome = f"assertion: {exc}"
        except ReachedKernelPreparation:
            outcome = "accepted: reached kernel preparation"

    print(f"{label}: {outcome}")
    if expect_pass:
        assert outcome == "accepted: reached kernel preparation"
    else:
        assert outcome == "assertion: Hidden size mismatch"


print(
    "device:",
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_properties(0).gcnArchName,
)

# Published DSV4-Flash-0731 shape: hidden=4096, packed FP4 physical K=2048.
check("legacy_fp8_route", 4096, 2048, use_fp8=True, expect_pass=False)
check("current_fp4_route", 4096, 2048, use_int4=True, expect_pass=True)

# Independent boundaries: ordinary FP8 K is uncompressed; malformed FP4 is rejected.
check("ordinary_fp8_route", 4096, 4096, use_fp8=True, expect_pass=True)
check("malformed_fp4_route", 4096, 2047, use_int4=True, expect_pass=False)
