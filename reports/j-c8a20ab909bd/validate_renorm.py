from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace

import torch


ABS_TOLERANCE = 2e-6
SEED = 20260909


def torch_top_k_renorm_reference(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    probs_fp32 = probs.float().contiguous()
    top_ks = top_ks.to(device=probs_fp32.device, dtype=torch.int64).reshape(-1)
    sorted_probs = torch.sort(probs_fp32, dim=-1, descending=True).values
    cutoff = (top_ks - 1).clamp_(min=0, max=probs_fp32.shape[-1] - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1)
    kept = probs_fp32 >= pivots.unsqueeze(1)
    result = probs_fp32 * kept
    return result / result.sum(dim=-1, keepdim=True)


def torch_top_p_renorm_reference(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    probs_fp32 = probs.float().contiguous()
    top_ps = top_ps.to(device=probs_fp32.device, dtype=torch.float32).reshape(-1)
    sorted_probs = torch.sort(probs_fp32, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(1)
    cutoff.clamp_(max=probs_fp32.shape[-1] - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1)
    kept = probs_fp32 >= pivots.unsqueeze(1)
    result = probs_fp32 * kept
    return result / result.sum(dim=-1, keepdim=True)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    difference = (actual - expected).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "row_sum_max_abs": float((actual.sum(-1) - expected.sum(-1)).abs().max()),
    }


def call_and_measure(function, probs, values, reference) -> dict:
    try:
        actual = function(probs, values)
        expected = reference(probs, values)
        if actual.dtype != expected.dtype:
            expected = expected.to(dtype=actual.dtype)
        result = metrics(actual, expected)
        result["passed"] = (
            result["max_abs"] <= ABS_TOLERANCE
            and result["row_sum_max_abs"] <= ABS_TOLERANCE
        )
        return result
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()

    import sglang
    import sglang.srt.speculative.dflash_utils as dflash
    import sglang.srt.utils as srt_utils
    import triton

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    dispatch_k = getattr(dflash, "top_k_renorm_prob", None)
    dispatch_p = getattr(dflash, "top_p_renorm_prob", None)
    selected_k = getattr(dflash, "_dflash_top_k_renorm_prob", None)
    selected_p = getattr(dflash, "_dflash_top_p_renorm_prob", None)

    report = {
        "label": args.label,
        "commit": args.commit,
        "seed": SEED,
        "abs_tolerance": ABS_TOLERANCE,
        "python": sys.executable,
        "sglang_module": sglang.__file__,
        "dflash_module": dflash.__file__,
        "triton_module": triton.__file__,
        "torch_version": torch.__version__,
        "torch_hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(device),
        "gcn_arch_name": torch.cuda.get_device_capability(device),
        "platform": {
            "is_cuda": getattr(srt_utils, "is_cuda", lambda: False)(),
            "is_hip": getattr(srt_utils, "is_hip", lambda: False)(),
            "is_musa": getattr(srt_utils, "is_musa", lambda: False)(),
            "is_npu": getattr(srt_utils, "is_npu", lambda: False)(),
        },
        "dispatch": {
            "top_k_renorm_prob": None if dispatch_k is None else repr(dispatch_k),
            "top_p_renorm_prob": None if dispatch_p is None else repr(dispatch_p),
            "selected_top_k_helper": None if selected_k is None else repr(selected_k),
            "selected_top_p_helper": None if selected_p is None else repr(selected_p),
            "sampling_verify_available": dflash.is_dflash_sampling_verify_available(),
        },
        "cases": {},
    }

    shapes = [
        (4, 1024, torch.float32),
        (3, 4097, torch.bfloat16),
        (2, 2048, torch.float16),
    ]
    for batch_size, vocab_size, dtype in shapes:
        logits = torch.randn(
            (batch_size, vocab_size), device=device, dtype=torch.float32
        )
        probs = torch.softmax(logits, dim=-1).to(dtype)
        top_ks = torch.tensor(
            [1, 5, 17, min(64, vocab_size)][:batch_size],
            device=device,
            dtype=torch.int32,
        )
        top_ps = torch.tensor(
            [0.25, 0.5, 0.8, 1.0][:batch_size],
            device=device,
            dtype=torch.float32,
        )
        case_name = f"{dtype}_b{batch_size}_v{vocab_size}"
        case = {
            "direct_top_k": (
                call_and_measure(dispatch_k, probs, top_ks, torch_top_k_renorm_reference)
                if callable(dispatch_k)
                else {"passed": False, "error": "dispatch is not callable"}
            ),
            "direct_top_p": (
                call_and_measure(dispatch_p, probs, top_ps, torch_top_p_renorm_reference)
                if callable(dispatch_p)
                else {"passed": False, "error": "dispatch is not callable"}
            ),
        }

        if selected_k is not None:
            case["selected_top_k"] = call_and_measure(
                selected_k, probs, top_ks, torch_top_k_renorm_reference
            )
        if selected_p is not None:
            case["selected_top_p"] = call_and_measure(
                selected_p, probs, top_ps, torch_top_p_renorm_reference
            )

        builder = getattr(dflash, "build_dflash_verify_target_probs", None)
        if builder is not None:
            draft_token_num = 2
            expanded_logits = torch.randn(
                (batch_size * draft_token_num, vocab_size),
                device=device,
                dtype=torch.float32,
            )
            sampling_info = SimpleNamespace(
                temperatures=torch.ones((batch_size, 1), device=device),
                top_ks=top_ks,
                top_ps=top_ps,
                need_top_k_sampling=True,
                need_top_p_sampling=True,
            )
            try:
                actual = builder(
                    next_token_logits=expanded_logits,
                    sampling_info=sampling_info,
                    draft_token_num=draft_token_num,
                    bs=batch_size,
                    use_sparse_topk=False,
                ).reshape(-1, vocab_size)
                scaled_logits = expanded_logits / sampling_info.temperatures.repeat_interleave(
                    draft_token_num, dim=0
                )
                base_probs = torch.softmax(scaled_logits, dim=-1)
                repeated_top_ks = top_ks.repeat_interleave(draft_token_num, dim=0)
                repeated_top_ps = top_ps.repeat_interleave(draft_token_num, dim=0)
                expected = torch_top_p_renorm_reference(
                    torch_top_k_renorm_reference(base_probs, repeated_top_ks),
                    repeated_top_ps,
                )
                case["build_verify_target_probs"] = metrics(actual, expected)
                case["build_verify_target_probs"]["passed"] = (
                    case["build_verify_target_probs"]["max_abs"] <= ABS_TOLERANCE
                    and case["build_verify_target_probs"]["row_sum_max_abs"]
                    <= ABS_TOLERANCE
                )
            except Exception as exc:
                case["build_verify_target_probs"] = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        report["cases"][case_name] = case

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
