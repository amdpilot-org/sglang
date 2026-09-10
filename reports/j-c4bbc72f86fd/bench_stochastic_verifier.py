import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from sgl_kernel import tree_speculative_sampling_target_only


def build_binary_tree(batch_size, num_draft_tokens, vocab_size, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = torch.randint(
        0,
        vocab_size,
        (batch_size, num_draft_tokens),
        dtype=torch.int64,
        generator=generator,
    )
    retrive_index = (
        torch.arange(batch_size, dtype=torch.int64).unsqueeze(1)
        * num_draft_tokens
        + torch.arange(num_draft_tokens, dtype=torch.int64).unsqueeze(0)
    )
    retrive_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64
    )
    retrive_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int64
    )
    for node_index in range(num_draft_tokens):
        left_child = 2 * node_index + 1
        if left_child < num_draft_tokens:
            retrive_next_token[:, node_index] = left_child
        if node_index % 2 == 1 and node_index + 1 < num_draft_tokens:
            retrive_next_sibling[:, node_index] = node_index + 1

    target_probs = torch.zeros(
        (batch_size, num_draft_tokens, vocab_size),
        dtype=torch.float32,
    )
    target_probs.scatter_(
        2,
        candidates.unsqueeze(2),
        torch.ones(
            (batch_size, num_draft_tokens, 1),
            dtype=torch.float32,
        ),
    )
    draft_probs = torch.zeros_like(target_probs)
    uniform_samples = torch.full(
        (batch_size, num_draft_tokens), 0.5, dtype=torch.float32
    )
    uniform_samples_for_final_sampling = torch.full(
        (batch_size, 1), 0.25, dtype=torch.float32
    )
    num_spec_steps = int(math.floor(math.log2(num_draft_tokens))) + 1
    return {
        "candidates": candidates,
        "retrive_index": retrive_index,
        "retrive_next_token": retrive_next_token,
        "retrive_next_sibling": retrive_next_sibling,
        "target_probs": target_probs,
        "draft_probs": draft_probs,
        "uniform_samples": uniform_samples,
        "uniform_samples_for_final_sampling": uniform_samples_for_final_sampling,
        "num_spec_steps": num_spec_steps,
    }


def reference_tree_speculative_sampling(
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    num_spec_steps,
):
    batch_size, num_draft_tokens = candidates.shape
    vocab_size = target_probs.shape[-1]

    candidates_cpu = candidates.cpu()
    retrive_index_cpu = retrive_index.cpu()
    retrive_next_token_cpu = retrive_next_token.cpu()
    retrive_next_sibling_cpu = retrive_next_sibling.cpu()
    uniform_cpu = uniform_samples.cpu()
    final_uniform_cpu = uniform_samples_for_final_sampling.cpu()
    target_cpu = target_probs.cpu().clone()
    draft_cpu = draft_probs.cpu().clone()

    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32
    )
    accept_index = torch.full(
        (batch_size, num_spec_steps), -1, dtype=torch.int32
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32)
    effective_threshold_acc = max(float(threshold_acc), 1e-9)

    for batch_index in range(batch_size):
        current_row = 0
        coin = float(uniform_cpu[batch_index, 0])
        probability_accumulator = 0.0
        last_accepted_index = int(retrive_index_cpu[batch_index, 0])
        accept_index[batch_index, 0] = last_accepted_index
        accepted_count = 0
        current_index = 0

        for _ in range(1, num_spec_steps):
            current_index = int(retrive_next_token_cpu[batch_index, current_index])
            while current_index != -1:
                draft_token = int(candidates_cpu[batch_index, current_index])
                target_probability = float(
                    target_cpu[batch_index, current_row, draft_token]
                )
                probability_accumulator += target_probability

                if (
                    coin <= probability_accumulator / effective_threshold_acc
                    or target_probability >= threshold_single
                ):
                    current_row = current_index
                    coin = float(uniform_cpu[batch_index, current_index])
                    predicts[last_accepted_index] = draft_token
                    accepted_count += 1
                    accept_index[batch_index, accepted_count] = int(
                        retrive_index_cpu[batch_index, current_index]
                    )
                    last_accepted_index = int(
                        retrive_index_cpu[batch_index, current_index]
                    )
                    break

                draft_cpu[batch_index, current_row, draft_token] = target_probability
                current_index = int(
                    retrive_next_sibling_cpu[batch_index, current_index]
                )

            if current_index == -1:
                break

        accept_token_num[batch_index] = accepted_count
        final_coin = float(final_uniform_cpu[batch_index])

        if accepted_count == num_spec_steps - 1:
            residual = target_cpu[batch_index, current_row].clone()
        else:
            residual = (
                target_cpu[batch_index, current_row]
                - draft_cpu[batch_index, current_row]
            ).clamp_min(0)

        residual_mass = float(residual.sum())
        if residual_mass > 0:
            inclusive_cdf = torch.cumsum(residual, dim=0)
            crossings = torch.nonzero(inclusive_cdf > final_coin * residual_mass)
            sampled_token = (
                int(crossings[0, 0]) if crossings.numel() else vocab_size - 1
            )
        else:
            sampled_token = vocab_size - 1

        predicts[last_accepted_index] = sampled_token

    return predicts, accept_index, accept_token_num, draft_cpu


def run_kernel(case, deterministic):
    device = case["candidates"].device
    batch_size, num_draft_tokens = case["candidates"].shape
    predicts = torch.full(
        (batch_size * num_draft_tokens,), -1, dtype=torch.int32, device=device
    )
    accept_index = torch.full(
        (batch_size, case["num_spec_steps"]), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int32, device=device)
    tree_speculative_sampling_target_only(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        candidates=case["candidates"],
        retrive_index=case["retrive_index"],
        retrive_next_token=case["retrive_next_token"],
        retrive_next_sibling=case["retrive_next_sibling"],
        uniform_samples=case["uniform_samples"],
        uniform_samples_for_final_sampling=case["uniform_samples_for_final_sampling"],
        target_probs=case["target_probs"],
        draft_probs=case["draft_probs"],
        threshold_single=1.0,
        threshold_acc=1.0,
        deterministic=deterministic,
    )
    return predicts, accept_index, accept_token_num, case["draft_probs"]


def benchmark_case(case, deterministic, warmup_calls, timed_calls, repeats):
    outputs = run_kernel(case, deterministic)
    torch.cuda.synchronize()
    for _ in range(warmup_calls):
        run_kernel(case, deterministic)
    torch.cuda.synchronize()
    timings = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(timed_calls):
            run_kernel(case, deterministic)
        end_event.record()
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event) / timed_calls)
    return outputs, timings


def tensor_bytes(*tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--timed-calls", type=int, default=20)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP GPU is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one assigned GPU")

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    cases = [
        {"name": "small", "batch": 2, "tree": 8, "vocab": 1024, "deterministic": True},
        {"name": "medium", "batch": 16, "tree": 16, "vocab": 4096, "deterministic": True},
        {"name": "large", "batch": 64, "tree": 32, "vocab": 8192, "deterministic": True},
        {"name": "medium_nondeterministic", "batch": 16, "tree": 16, "vocab": 4096, "deterministic": False},
    ]

    results = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "architecture": torch.cuda.get_device_capability(0),
            "device_count": torch.cuda.device_count(),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "timing": {
            "method": "CUDA/HIP events around bounded loops; mean of timed_calls per repeat",
            "warmup_calls": args.warmup_calls,
            "timed_calls": args.timed_calls,
            "timing_repeats": args.timing_repeats,
        },
        "cases": [],
    }

    for case_index, case in enumerate(cases):
        seed = 0o30344 + case_index
        cpu_case = build_binary_tree(
            case["batch"], case["tree"], case["vocab"], seed
        )
        gpu_case = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in cpu_case.items()
        }
        reference_outputs = reference_tree_speculative_sampling(
            gpu_case["candidates"].cpu(),
            gpu_case["retrive_index"].cpu(),
            gpu_case["retrive_next_token"].cpu(),
            gpu_case["retrive_next_sibling"].cpu(),
            gpu_case["uniform_samples"].cpu(),
            gpu_case["uniform_samples_for_final_sampling"].cpu(),
            gpu_case["target_probs"].cpu(),
            gpu_case["draft_probs"].cpu(),
            1.0,
            1.0,
            gpu_case["num_spec_steps"],
        )

        torch.cuda.reset_peak_memory_stats()
        outputs, timings = benchmark_case(
            gpu_case,
            case["deterministic"],
            args.warmup_calls,
            args.timed_calls,
            args.timing_repeats,
        )
        torch.testing.assert_close(outputs[0].cpu(), reference_outputs[0])
        torch.testing.assert_close(outputs[1].cpu(), reference_outputs[1])
        torch.testing.assert_close(outputs[2].cpu(), reference_outputs[2])
        torch.testing.assert_close(outputs[3].cpu(), reference_outputs[3])

        mean_timing = sum(timings) / len(timings)
        variance = sum((timing - mean_timing) ** 2 for timing in timings) / len(timings)
        standard_deviation = math.sqrt(variance)
        live_bytes = tensor_bytes(
            gpu_case["candidates"],
            gpu_case["retrive_index"],
            gpu_case["retrive_next_token"],
            gpu_case["retrive_next_sibling"],
            gpu_case["target_probs"],
            gpu_case["draft_probs"],
            gpu_case["uniform_samples"],
            gpu_case["uniform_samples_for_final_sampling"],
            outputs[0],
            outputs[1],
            outputs[2],
        )
        results["cases"].append(
            {
                "name": case["name"],
                "batch_size": case["batch"],
                "num_draft_tokens": case["tree"],
                "vocab_size": case["vocab"],
                "num_spec_steps": gpu_case["num_spec_steps"],
                "deterministic": case["deterministic"],
                "seed": seed,
                "timings_ms": timings,
                "mean_ms": mean_timing,
                "stddev_ms": standard_deviation,
                "min_ms": min(timings),
                "max_ms": max(timings),
                "live_tensor_bytes": live_bytes,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "reference_match": True,
            }
        )
        del cpu_case, gpu_case, outputs, reference_outputs
        torch.cuda.empty_cache()

    results["gpu"]["rocm_smi"] = subprocess.check_output(
        ["rocm-smi", "--showproductname", "--showserial", "--showdriverversion"],
        text=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
