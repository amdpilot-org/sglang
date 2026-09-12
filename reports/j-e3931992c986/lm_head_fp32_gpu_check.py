"""Single-GPU numerical and timing evidence for FP32 LM-head output.

This is deliberately not described as a distributed TP run.  The shard
reassembly check only exercises the arithmetic that a rank-concatenating
all-gather must preserve.
"""

import json
import statistics
import time

import torch


def timed(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {"median_us": statistics.median(samples), "min_us": min(samples)}


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(33627)
    device = "cuda"
    k, vocab = 4096, 8192
    weight = torch.randn(vocab, k, device=device, dtype=torch.bfloat16)
    cases = []
    for tokens in (1, 8, 32, 128):
        hidden = torch.randn(tokens, k, device=device, dtype=torch.bfloat16)
        reference = hidden.float() @ weight.float().T
        direct = torch.mm(hidden, weight.T, out_dtype=torch.float32)
        post_cast = (hidden @ weight.T).float()
        shard_direct = torch.cat(
            [
                torch.mm(hidden, shard.T, out_dtype=torch.float32)
                for shard in weight.tensor_split(4, dim=0)
            ],
            dim=-1,
        )
        cases.append(
            {
                "tokens": tokens,
                "direct_max_abs_vs_reference": (direct - reference).abs().max().item(),
                "post_cast_max_abs_vs_reference": (post_cast - reference)
                .abs()
                .max()
                .item(),
                "direct_top1_mismatches": (direct.argmax(-1) != reference.argmax(-1))
                .sum()
                .item(),
                "post_cast_top1_mismatches": (
                    post_cast.argmax(-1) != reference.argmax(-1)
                )
                .sum()
                .item(),
                "tp4_shard_reassembly_exact": torch.equal(shard_direct, direct),
                "tp4_shard_max_abs_vs_full_gemm": (shard_direct - direct)
                .abs()
                .max()
                .item(),
                "tp4_shard_top1_mismatches_vs_full_gemm": (
                    shard_direct.argmax(-1) != direct.argmax(-1)
                )
                .sum()
                .item(),
                "direct_timing": timed(
                    lambda: torch.mm(hidden, weight.T, out_dtype=torch.float32)
                ),
                "post_cast_timing": timed(lambda: (hidden @ weight.T).float()),
            }
        )
        del hidden, reference, direct, post_cast, shard_direct

    hidden = torch.ones((1, 64), dtype=torch.bfloat16, device=device)
    ranking_weight = torch.ones((2, 64), dtype=torch.bfloat16, device=device)
    ranking_weight[1, 0] = 1.0078125
    ranking_ref = hidden.float() @ ranking_weight.float().T
    ranking_direct = torch.mm(hidden, ranking_weight.T, out_dtype=torch.float32)
    ranking_post = (hidden @ ranking_weight.T).float()

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "gpu": torch.cuda.get_device_name(),
                "seed": 33627,
                "shape": {"k": k, "vocab": vocab},
                "cases": cases,
                "constructed_ranking_case": {
                    "reference_logits": ranking_ref.tolist(),
                    "direct_logits": ranking_direct.tolist(),
                    "post_cast_logits": ranking_post.tolist(),
                    "reference_top1": ranking_ref.argmax(-1).item(),
                    "direct_top1": ranking_direct.argmax(-1).item(),
                    "post_cast_top1": ranking_post.argmax(-1).item(),
                },
                "limitations": [
                    "one assigned GPU only; no live TP4 collective",
                    "8192x4096 synthetic BF16 head, not a loaded model",
                    "event timings exclude a distributed logits all-gather",
                ],
                "wall_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
