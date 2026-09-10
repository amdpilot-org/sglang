from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.models.dspark import (
    VanillaMarkov,
    project_through_lm_head,
    run_markov_block,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)


HIDDEN_SIZE = 512
VOCAB_SIZE = 1024
MARKOV_RANK = 64
RECURRENT_STEPS = 4
WARMUP_SEQUENCES = 5
MEASURED_SEQUENCES = 30
WEIGHT_LIMIT_BYTES = 4 * 1024**3
MEMORY_LIMIT_BYTES = 48 * 1024**3
CASES = [
    {"batch_size": 1, "gamma": 2},
    {"batch_size": 1, "gamma": 4},
    {"batch_size": 2, "gamma": 4},
    {"batch_size": 4, "gamma": 4},
    {"batch_size": 8, "gamma": 4},
    {"batch_size": 4, "gamma": 8},
]


class SyntheticLMHead(nn.Module):
    quant_method = None

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.org_vocab_size = vocab_size
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=torch.bfloat16, device="cuda")
        )


def _weight_bytes(lm_head: SyntheticLMHead, markov_head: VanillaMarkov) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            lm_head.weight,
            markov_head.markov_w1.weight,
            markov_head.markov_w2.weight,
        )
    )


def _reference_markov_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    markov_head: VanillaMarkov,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_tokens = []
    corrected_logits = []
    previous_tokens = anchor_tokens.long()
    for step_index in range(base_logits.shape[1]):
        latent_states = markov_head.markov_w1(previous_tokens)
        step_bias = markov_head.markov_w2(latent_states)
        step_logits = base_logits[:, step_index, :].float() + step_bias
        sampled_tokens.append(torch.argmax(step_logits, dim=-1))
        corrected_logits.append(step_logits.unsqueeze(1))
        previous_tokens = sampled_tokens[-1]
    return torch.stack(sampled_tokens, dim=1), torch.cat(corrected_logits, dim=1)


def _reference_accept(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    verify_num_draft_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = candidates.shape[0]
    target_predict = torch.argmax(target_logits, dim=-1).view(
        batch_size, verify_num_draft_tokens
    )
    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)
    rows = torch.arange(batch_size, device=candidates.device)
    bonus = target_predict[rows, correct_len.long()]
    return correct_len, bonus, torch.zeros_like(correct_len)


def _reference_commit(
    *,
    draft_tokens: torch.Tensor,
    correct_len: torch.Tensor,
    bonus: torch.Tensor,
    gamma: int,
) -> torch.Tensor:
    batch_size = draft_tokens.shape[0]
    output_tokens = torch.zeros(
        (batch_size, gamma + 1), dtype=torch.int64, device=draft_tokens.device
    )
    output_tokens[:, :gamma] = draft_tokens
    rows = torch.arange(batch_size, device=draft_tokens.device)
    output_tokens[rows, correct_len.long()] = bonus
    return output_tokens


def _run_sequence(
    *,
    raw_hidden: torch.Tensor,
    anchor_tokens: torch.Tensor,
    lm_head: SyntheticLMHead,
    markov_head: VanillaMarkov,
    batch_size: int,
    gamma: int,
    validate: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    working_anchor = anchor_tokens.clone()
    step_records: list[dict[str, Any]] = []
    temperatures = torch.ones(batch_size, dtype=torch.float32, device="cuda")
    greedy_mask = torch.ones(batch_size, dtype=torch.bool, device="cuda")
    exp_noise = torch.ones(
        batch_size, VOCAB_SIZE, dtype=torch.float32, device="cuda"
    )

    def sampler(step_logits: torch.Tensor, step_index: int) -> torch.Tensor:
        return SampleStepTokens.execute(
            step_logits=step_logits,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
            exp_noise=exp_noise,
        )

    for step_index in range(RECURRENT_STEPS):
        step_hidden = raw_hidden[step_index]
        selected_hidden, selected_hidden_3d = select_draft_hidden_without_anchor(
            step_hidden.reshape(-1, HIDDEN_SIZE), bs=batch_size, gamma=gamma
        )
        base_logits = project_through_lm_head(selected_hidden, lm_head).view(
            batch_size, gamma, VOCAB_SIZE
        )
        draft_tokens, corrected_logits = run_markov_block(
            markov_head,
            base_logits,
            first_prev_tokens=working_anchor,
            hidden_states=selected_hidden_3d,
            sampler=sampler,
        )
        target_logits = project_through_lm_head(
            step_hidden.reshape(-1, HIDDEN_SIZE), lm_head
        )
        candidates = torch.cat(
            (working_anchor[:, None], draft_tokens), dim=1
        )
        correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
            candidates=candidates,
            target_logits=target_logits,
            verify_num_draft_tokens=gamma + 1,
        )
        output_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=correct_len,
            bonus=bonus,
            verify_num_draft_tokens=gamma + 1,
            gamma=gamma,
        )

        if validate:
            reference_hidden = step_hidden[:, 1:, :].reshape(-1, HIDDEN_SIZE)
            torch.testing.assert_close(selected_hidden, reference_hidden)
            reference_base_logits = torch.matmul(
                reference_hidden.float(), lm_head.weight.float().T
            ).view(batch_size, gamma, VOCAB_SIZE)
            torch.testing.assert_close(
                base_logits.float(),
                reference_base_logits,
                rtol=2e-2,
                atol=2e-2,
            )
            reference_draft_tokens, reference_corrected_logits = (
                _reference_markov_block(
                    base_logits=base_logits,
                    anchor_tokens=working_anchor,
                    markov_head=markov_head,
                )
            )
            torch.testing.assert_close(
                corrected_logits,
                reference_corrected_logits,
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                draft_tokens, reference_draft_tokens, rtol=0.0, atol=0.0
            )
            reference_correct_len, reference_bonus, reference_cap_trim = (
                _reference_accept(
                    candidates=candidates,
                    target_logits=target_logits,
                    verify_num_draft_tokens=gamma + 1,
                )
            )
            torch.testing.assert_close(
                correct_len, reference_correct_len, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(bonus, reference_bonus, rtol=0.0, atol=0.0)
            torch.testing.assert_close(
                cap_trim_lens, reference_cap_trim, rtol=0.0, atol=0.0
            )
            reference_output_tokens = _reference_commit(
                draft_tokens=draft_tokens,
                correct_len=reference_correct_len,
                bonus=reference_bonus,
                gamma=gamma,
            )
            torch.testing.assert_close(
                output_tokens, reference_output_tokens, rtol=0.0, atol=0.0
            )

        rows = torch.arange(batch_size, device="cuda")
        next_anchor = output_tokens[rows, correct_len.long()]
        step_records.append(
            {
                "step": step_index,
                "anchor_dtype": str(working_anchor.dtype),
                "draft_token_dtype": str(draft_tokens.dtype),
                "correct_len_dtype": str(correct_len.dtype),
                "bonus_dtype": str(bonus.dtype),
                "output_token_dtype": str(output_tokens.dtype),
                "mean_accept_length": float(correct_len.float().mean().item()),
                "anchor_checksum": int(working_anchor.sum().item()),
                "output_checksum": int(output_tokens.sum().item()),
            }
        )
        working_anchor = next_anchor

    return working_anchor, step_records


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(math.ceil(percentile * len(ordered))) - 1),
    )
    return ordered[index]


def _git_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd="/job/sglang", text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    args = parser.parse_args()

    started = time.monotonic()
    torch.manual_seed(30734)
    generator = torch.Generator(device="cuda").manual_seed(30734)
    lm_head = SyntheticLMHead(HIDDEN_SIZE, VOCAB_SIZE)
    with torch.no_grad():
        lm_head.weight.normal_(0.0, 0.03, generator=generator)
    markov_head = VanillaMarkov(vocab_size=VOCAB_SIZE, markov_rank=MARKOV_RANK).cuda()
    with torch.no_grad():
        markov_head.markov_w1.weight.normal_(0.0, 0.05, generator=generator)
        markov_head.markov_w2.weight.normal_(0.0, 0.05, generator=generator)

    generated_weight_bytes = _weight_bytes(lm_head, markov_head)
    if generated_weight_bytes >= WEIGHT_LIMIT_BYTES:
        raise RuntimeError(
            f"generated weights exceed limit: {generated_weight_bytes} bytes"
        )

    results: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    for case_index, case in enumerate(CASES):
        batch_size = case["batch_size"]
        gamma = case["gamma"]
        raw_hidden = torch.randn(
            (
                RECURRENT_STEPS,
                batch_size,
                gamma + 1,
                HIDDEN_SIZE,
            ),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        anchor_tokens = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size,),
            dtype=torch.int64,
            device="cuda",
            generator=generator,
        )

        with torch.inference_mode():
            correctness_anchor, correctness_records = _run_sequence(
                raw_hidden=raw_hidden,
                anchor_tokens=anchor_tokens,
                lm_head=lm_head,
                markov_head=markov_head,
                batch_size=batch_size,
                gamma=gamma,
                validate=True,
            )

            for _ in range(WARMUP_SEQUENCES):
                _run_sequence(
                    raw_hidden=raw_hidden,
                    anchor_tokens=anchor_tokens,
                    lm_head=lm_head,
                    markov_head=markov_head,
                    batch_size=batch_size,
                    gamma=gamma,
                    validate=False,
                )
            torch.cuda.synchronize()

            raw_timings_ms: list[float] = []
            for _ in range(MEASURED_SEQUENCES):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _run_sequence(
                    raw_hidden=raw_hidden,
                    anchor_tokens=anchor_tokens,
                    lm_head=lm_head,
                    markov_head=markov_head,
                    batch_size=batch_size,
                    gamma=gamma,
                    validate=False,
                )
                end_event.record()
                end_event.synchronize()
                raw_timings_ms.append(start_event.elapsed_time(end_event))

        median_ms = statistics.median(raw_timings_ms)
        results.append(
            {
                "case_index": case_index,
                "dimensions": {
                    "batch_size": batch_size,
                    "gamma": gamma,
                    "verify_num_draft_tokens": gamma + 1,
                    "hidden_size": HIDDEN_SIZE,
                    "vocab_size": VOCAB_SIZE,
                    "markov_rank": MARKOV_RANK,
                    "recurrent_steps": RECURRENT_STEPS,
                },
                "correctness": {
                    "passed": True,
                    "final_anchor_checksum": int(correctness_anchor.sum().item()),
                    "step_records": correctness_records,
                },
                "timing": {
                    "warmup_sequences": WARMUP_SEQUENCES,
                    "measured_sequences": MEASURED_SEQUENCES,
                    "real_block_forwards_per_sequence": RECURRENT_STEPS,
                    "median_ms": median_ms,
                    "median_per_step_ms": median_ms / RECURRENT_STEPS,
                    "p90_ms": _percentile(raw_timings_ms, 0.90),
                    "p99_ms": _percentile(raw_timings_ms, 0.99),
                    "min_ms": min(raw_timings_ms),
                    "max_ms": max(raw_timings_ms),
                    "raw_ms": raw_timings_ms,
                },
                "memory": {
                    "live_allocated_bytes_after_case": int(
                        torch.cuda.memory_allocated()
                    ),
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                },
            }
        )
        del raw_hidden

    live_allocated_bytes = int(torch.cuda.memory_allocated())
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    if live_allocated_bytes >= MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"live allocations exceed limit: {live_allocated_bytes} bytes"
        )
    if peak_allocated_bytes >= MEMORY_LIMIT_BYTES:
        raise RuntimeError(
            f"peak allocations exceed limit: {peak_allocated_bytes} bytes"
        )

    report = {
        "label": "gfx942 reduced decode-sized block study",
        "campaign": "repo-e2e-20260909",
        "source_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "source_branch": _git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "python_path": "/opt/venv/bin/python",
        "sglang_import_path": "/job/sglang/python/sglang/__init__.py",
        "sgl_kernel_native_path": (
            "/opt/venv/lib/python3.10/site-packages/sgl_kernel/"
            "common_ops.cpython-310-x86_64-linux-gnu.so"
        ),
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "gfx": "gfx942",
        },
        "dtypes": {
            "target_hidden": "bfloat16",
            "lm_head_weight": "bfloat16",
            "markov_weights": "float32",
            "tokens_and_indices": "int64",
            "commit_lengths": "int32",
            "sampled_tokens": "int64",
        },
        "accuracy_gates": {
            "hidden_selection": "exact tensor equality",
            "projection": "bf16 primitive vs independent float32 matmul: rtol=2e-2, atol=2e-2",
            "markov_block": "independent per-step embedding/linear loop: logits rtol=1e-5, atol=1e-5; tokens exact",
            "greedy_accept": "independent leading-match/cumprod rule: exact",
            "commit": "independent scatter construction: exact",
            "recurrent_state": "anchor and output checksums checked after every step",
        },
        "timing_method": {
            "events": "CUDA events around one four-step recurrent sequence",
            "warmup_sequences_per_case": WARMUP_SEQUENCES,
            "measured_sequences_per_case": MEASURED_SEQUENCES,
            "synchronization": "end event synchronized after every measured sequence",
            "tail_metrics": ["p90_ms", "p99_ms"],
        },
        "limits": {
            "wall_limit_seconds": 7200,
            "workload_cases": len(CASES),
            "generated_weight_limit_bytes": WEIGHT_LIMIT_BYTES,
            "memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "no_model_weights_downloaded": True,
            "no_distributed_claim": True,
        },
        "model_specific_constructor_boundary": (
            "No model-specific DSpark constructor or GLM checkpoint was instantiated. "
            "The reduced study uses the generic VanillaMarkov head and a synthetic "
            "LM-head adapter; DeepSeek/GLM-specific constructors and target modules "
            "remain outside this boundary."
        ),
        "generated_weight_bytes": generated_weight_bytes,
        "live_allocated_bytes": live_allocated_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "total_real_block_forwards": len(CASES)
        * (WARMUP_SEQUENCES + MEASURED_SEQUENCES)
        * RECURRENT_STEPS,
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
