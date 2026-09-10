import json
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from aiter import biased_grouped_topk
from aiter.tuned_gemm import tgemm
from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def error_metrics(actual, reference):
    actual_float = actual.float()
    reference_float = reference.float()
    absolute = (actual_float - reference_float).abs()
    relative = absolute / reference_float.abs().clamp_min(1e-6)
    return {
        "max_abs_error": absolute.max().item(),
        "max_rel_error": relative.max().item(),
        "mean_abs_error": absolute.mean().item(),
    }


def grouped_topk_reference(gating_output, correction_bias, topk, num_expert_group, topk_group):
    scores = gating_output.float().sigmoid()
    scores_for_choice = scores + correction_bias.float().unsqueeze(0)
    grouped_scores = scores_for_choice.view(
        scores.shape[0], num_expert_group, -1
    )
    group_scores = grouped_scores.topk(2, dim=-1).values.sum(dim=-1)
    group_indices = group_scores.topk(topk_group, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_indices, 1)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand_as(grouped_scores)
        .reshape(scores_for_choice.shape)
    )
    masked_scores = scores_for_choice.masked_fill(~expert_mask, float("-inf"))
    topk_ids = masked_scores.topk(topk, dim=-1).indices
    topk_weights = scores.gather(1, topk_ids)
    return topk_weights, topk_ids


def run_aiter_topk(gating_output, correction_bias, topk, num_expert_group, topk_group):
    topk_weights = torch.empty(
        (gating_output.shape[0], topk),
        dtype=torch.float32,
        device=gating_output.device,
    )
    topk_ids = torch.empty(
        (gating_output.shape[0], topk),
        dtype=torch.int32,
        device=gating_output.device,
    )
    biased_grouped_topk(
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        need_renorm=False,
    )
    return topk_weights, topk_ids


def main():
    torch.manual_seed(34857)
    device = torch.device("cuda:0")
    num_tokens = 1
    hidden_size = 7168
    num_experts = 256

    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=device, dtype=torch.float32
    )
    hidden_states[:, 0] = 0.01
    hidden_states_bf16 = hidden_states.to(torch.bfloat16)
    router_weight = (
        torch.randn((num_experts, hidden_size), device=device, dtype=torch.float32)
        * 0.001
    )
    router_weight[1] = router_weight[0]
    router_weight[:, 0] = 0.001
    router_weight[1, 0] = 0.00101
    router_weight[2:] = 0.0
    router_weight_bf16 = router_weight.to(torch.bfloat16)

    reference_logits = torch.matmul(
        hidden_states_bf16.float(), router_weight_bf16.t().float()
    )
    helper_logits = aiter_dsv3_router_gemm(hidden_states_bf16, router_weight_bf16)
    direct_bf16_logits = tgemm.mm(
        hidden_states_bf16, router_weight_bf16.detach(), otype=torch.bfloat16
    )
    direct_fp32_logits = tgemm.mm(
        hidden_states_bf16, router_weight_bf16.detach(), otype=torch.float32
    )

    reference_argmax = reference_logits.argmax(dim=-1)
    helper_argmax = helper_logits.argmax(dim=-1)
    reference_top8 = reference_logits.topk(8, dim=-1).indices
    helper_top8 = helper_logits.topk(8, dim=-1).indices
    winner_logit = reference_logits[0, reference_argmax.item()]
    winner_bf16 = winner_logit.to(torch.bfloat16)
    next_bf16 = torch.nextafter(
        winner_bf16,
        torch.tensor(float("inf"), device=device, dtype=torch.bfloat16),
    )
    bf16_ulp = (next_bf16.float() - winner_bf16.float()).item()
    near_tie_difference = (
        reference_logits[0, 1] - reference_logits[0, 0]
    ).item()

    router_result = {
        "shapes": {
            "hidden_states": list(hidden_states_bf16.shape),
            "router_weight": list(router_weight_bf16.shape),
        },
        "input_dtypes": {
            "hidden_states": str(hidden_states_bf16.dtype),
            "router_weight": str(router_weight_bf16.dtype),
        },
        "helper_output_dtype": str(helper_logits.dtype),
        "direct_bf16_output_dtype": str(direct_bf16_logits.dtype),
        "direct_fp32_output_dtype": str(direct_fp32_logits.dtype),
        "helper_vs_reference": error_metrics(helper_logits, reference_logits),
        "direct_bf16_vs_reference": error_metrics(
            direct_bf16_logits, reference_logits
        ),
        "direct_fp32_vs_reference": error_metrics(
            direct_fp32_logits, reference_logits
        ),
        "direct_fp32_equals_direct_bf16": bool(
            torch.equal(direct_fp32_logits, direct_bf16_logits)
        ),
        "near_tie": {
            "reference_logit_expert_0": reference_logits[0, 0].item(),
            "reference_logit_expert_1": reference_logits[0, 1].item(),
            "reference_difference": near_tie_difference,
            "bf16_ulp_at_reference_winner": bf16_ulp,
            "difference_is_below_bf16_ulp": bool(
                abs(near_tie_difference) < bf16_ulp
            ),
            "reference_argmax": reference_argmax.tolist(),
            "actual_argmax": helper_argmax.tolist(),
            "argmax_matches_reference": bool(
                torch.equal(reference_argmax, helper_argmax)
            ),
            "reference_top8": reference_top8.tolist(),
            "actual_top8": helper_top8.tolist(),
            "top8_set_matches_reference": bool(
                torch.equal(
                    reference_top8.sort(dim=-1).values,
                    helper_top8.sort(dim=-1).values,
                )
            ),
        },
    }

    gating_bf16 = torch.zeros(
        (num_tokens, num_experts), device=device, dtype=torch.bfloat16
    )
    gating_fp32 = gating_bf16.to(torch.float32)
    correction_bias_fp32 = torch.zeros(
        (num_experts,), device=device, dtype=torch.float32
    )
    correction_bias_fp32[:9] = 0.1 + torch.arange(
        9, device=device, dtype=torch.float32
    ) * 1e-5
    correction_bias_bf16 = correction_bias_fp32.to(torch.bfloat16)

    topk = 8
    num_expert_group = 8
    topk_group = 4
    reference_weights, reference_ids = grouped_topk_reference(
        gating_fp32,
        correction_bias_fp32,
        topk,
        num_expert_group,
        topk_group,
    )
    current_weights, current_ids = run_aiter_topk(
        gating_bf16,
        correction_bias_bf16,
        topk,
        num_expert_group,
        topk_group,
    )
    candidate_weights, candidate_ids = run_aiter_topk(
        gating_fp32,
        correction_bias_fp32,
        topk,
        num_expert_group,
        topk_group,
    )

    def sorted_ids(ids):
        return ids.sort(dim=-1).values.tolist()

    bias_result = {
        "shapes": {
            "gating_output": list(gating_bf16.shape),
            "correction_bias": list(correction_bias_fp32.shape),
        },
        "topk": topk,
        "num_expert_group": num_expert_group,
        "topk_group": topk_group,
        "near_tie_bias_values_fp32": correction_bias_fp32[:9].tolist(),
        "near_tie_bias_values_after_bf16_cast": correction_bias_bf16[
            :9
        ].float().tolist(),
        "reference_input_dtypes": {
            "gating_output": str(gating_fp32.dtype),
            "correction_bias": str(correction_bias_fp32.dtype),
        },
        "current_input_dtypes": {
            "gating_output": str(gating_bf16.dtype),
            "correction_bias": str(correction_bias_bf16.dtype),
            "bias_cast_occurs_before_add": True,
        },
        "candidate_input_dtypes": {
            "gating_output": str(gating_fp32.dtype),
            "correction_bias": str(correction_bias_fp32.dtype),
            "bias_cast_occurs_before_add": False,
        },
        "reference_top8": sorted_ids(reference_ids),
        "current_top8": sorted_ids(current_ids),
        "candidate_top8": sorted_ids(candidate_ids),
        "current_top8_set_matches_reference": bool(
            torch.equal(
                reference_ids.sort(dim=-1).values,
                current_ids.sort(dim=-1).values,
            )
        ),
        "candidate_top8_set_matches_reference": bool(
            torch.equal(
                reference_ids.sort(dim=-1).values,
                candidate_ids.sort(dim=-1).values,
            )
        ),
        "reference_top8_weights": reference_weights.tolist(),
        "current_top8_weights": current_weights.tolist(),
        "candidate_top8_weights": candidate_weights.tolist(),
    }

    device_properties = torch.cuda.get_device_properties(0)
    result = {
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "gpu_name": device_properties.name,
        "gpu_gcn_arch_name": device_properties.gcnArchName,
        "operations": {
            "router_gemm_python": "aiter_dsv3_router_gemm -> tgemm.mm",
            "router_gemm_tuned_dispatcher": "aiter.tuned_gemm.tgemm.mm",
            "expert_bias_topk": "aiter.biased_grouped_topk",
            "independent_reference": "torch.matmul in fp32 and torch.topk in fp32",
        },
        "source_paths": {
            "router_gemm": "python/sglang/srt/layers/rocm_linear_utils.py",
            "router_gate": "python/sglang/srt/models/deepseek_v2.py",
            "expert_topk": "python/sglang/srt/layers/moe/topk.py",
            "aiter_topk": "/sgl-workspace/aiter/aiter/ops/topk.py",
            "aiter_tuned_gemm": "/sgl-workspace/aiter/aiter/tuned_gemm.py",
        },
        "native_paths": {
            "aiter_core": "/sgl-workspace/aiter/aiter/jit/module_aiter_core.so",
            "aiter_moe_asm": "/sgl-workspace/aiter/aiter/jit/module_moe_asm.so",
        },
        "router_gemm": router_result,
        "expert_correction_bias": bias_result,
    }

    output_path = Path(
        "/job/sglang/reports/j-8de2cc83c930/rocm_router_precision_results.json"
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
