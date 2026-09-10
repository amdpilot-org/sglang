#!/usr/bin/env python3
"""Bounded gfx942 probe for KDA varlen metadata and chunk_kda_fwd.

The parent process launches each trial in a child process with a timeout. A
child prints a stage marker before every potentially blocking operation, so a
timeout identifies the last reached stage. The numerical reference is an
independent sequential PyTorch implementation; it does not call the Triton KDA
kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

CHUNK_SIZE = 64
LOWER_BOUND = -5.0
TRIALS_PER_CASE = 10
TRIAL_TIMEOUT_SECONDS = 45.0


def _emit(stage: str, **payload: object) -> None:
    record = {"stage": stage, "monotonic_ns": time.monotonic_ns(), **payload}
    print(json.dumps(record, sort_keys=True), flush=True)


def _reference_chunk_indices(sequence_lengths: list[int]) -> list[list[int]]:
    return [
        [sequence_index, chunk_index]
        for sequence_index, length in enumerate(sequence_lengths)
        for chunk_index in range(math.ceil(length / CHUNK_SIZE))
    ]


def _run_child(case: str, seed: int) -> int:
    import torch
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
    from sglang.kernels.ops.attention.fla.kda import chunk_kda

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is unavailable")
    device_arch = torch.cuda.get_device_properties(0).gcnArchName
    if "gfx942" not in device_arch:
        raise RuntimeError(
            f"expected gfx942/MI300X, got {device_arch}"
        )

    sequence_lengths = [1] if case == "single_request" else [1, 65]
    request_count = len(sequence_lengths)
    total_tokens = sum(sequence_lengths)
    heads = 1
    key_dim = 32
    value_dim = 32
    state_slots = request_count + 1

    torch.manual_seed(seed)
    device = "cuda"
    cumulative_lengths = torch.tensor(
        [0] + [sum(sequence_lengths[: i + 1]) for i in range(request_count)],
        device=device,
        dtype=torch.int32,
    )
    state_indices = torch.full((state_slots,), -1, device=device, dtype=torch.int32)
    state_indices[:request_count] = torch.arange(request_count, device=device)
    sentinel_before = state_indices[-1].clone()

    query = torch.randn(
        1, total_tokens, heads, key_dim, device=device, dtype=torch.bfloat16
    ) * 0.3
    key = torch.randn_like(query) * 0.3
    value = torch.randn(
        1, total_tokens, heads, value_dim, device=device, dtype=torch.bfloat16
    ) * 0.3
    raw_gate = torch.randn(
        1, total_tokens, heads, key_dim, device=device, dtype=torch.bfloat16
    ) * 0.3
    beta = (
        torch.rand(1, total_tokens, heads, device=device) * 0.8 + 0.1
    ).to(torch.bfloat16)
    a_log = torch.randn(heads, device=device, dtype=torch.float32) * 0.2
    dt_bias = torch.randn(heads * key_dim, device=device, dtype=torch.float32) * 0.1
    initial_state = torch.randn(
        state_slots, heads, value_dim, key_dim, device=device, dtype=torch.float32
    ) * 0.05
    initial_state_copy = initial_state.clone()

    _emit(
        "device_ready",
        case=case,
        seed=seed,
        device_name=torch.cuda.get_device_name(0),
        device_arch=device_arch,
        capability=list(torch.cuda.get_device_capability(0)),
        stream=str(torch.cuda.current_stream()),
        sequence_lengths=sequence_lengths,
        state_indices=state_indices.detach().cpu().tolist(),
    )

    _emit("before_prepare_chunk_indices")
    stream_before = torch.cuda.Event(enable_timing=True)
    stream_after = torch.cuda.Event(enable_timing=True)
    stream_before.record()
    actual_indices = prepare_chunk_indices(cumulative_lengths, CHUNK_SIZE)
    stream_after.record()
    torch.cuda.synchronize()
    index_elapsed_ms = stream_before.elapsed_time(stream_after)
    _emit(
        "after_prepare_chunk_indices",
        elapsed_ms=index_elapsed_ms,
        shape=list(actual_indices.shape),
        values=actual_indices.detach().cpu().tolist(),
    )

    expected_indices = _reference_chunk_indices(sequence_lengths)
    expected_indices_tensor = torch.tensor(
        expected_indices, device=device, dtype=actual_indices.dtype
    )
    if not torch.equal(actual_indices, expected_indices_tensor):
        raise AssertionError(
            f"index map mismatch: actual={actual_indices.cpu().tolist()} "
            f"expected={expected_indices}"
        )

    _emit("before_chunk_kda_fwd")
    stream_before.record()
    output = chunk_kda(
        query.clone(),
        key.clone(),
        value.clone(),
        raw_gate.clone(),
        beta.clone(),
        scale=key_dim**-0.5,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cumulative_lengths,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
    )
    stream_after.record()
    torch.cuda.synchronize()
    kernel_elapsed_ms = stream_before.elapsed_time(stream_after)
    _emit(
        "after_chunk_kda_fwd",
        elapsed_ms=kernel_elapsed_ms,
        output_shape=list(output.shape),
        output_dtype=str(output.dtype),
    )

    normalized_query = torch.nn.functional.normalize(query.float(), dim=-1)
    normalized_key = torch.nn.functional.normalize(key.float(), dim=-1)
    value_float = value.float()
    beta_float = beta.float()
    reference_state = initial_state_copy[state_indices[:request_count]].float().clone()
    reference_output = torch.empty_like(value_float)
    token_offset = 0
    for sequence_index, sequence_length in enumerate(sequence_lengths):
        for token_index in range(sequence_length):
            raw = (
                raw_gate[0, token_offset + token_index].float()
                + dt_bias.view(heads, key_dim)
            )
            gate = LOWER_BOUND * torch.sigmoid(torch.exp(a_log)[:, None] * raw)
            decay = torch.exp(gate)
            previous_state = reference_state[sequence_index].clone()
            decayed_projection = torch.einsum(
                "hk,hvk->hv",
                normalized_key[0, token_offset + token_index],
                decay * previous_state,
            )
            correction = beta_float[0, token_offset + token_index] * (
                value_float[0, token_offset + token_index] - decayed_projection
            )
            reference_state[sequence_index] = decay * previous_state + torch.einsum(
                "hk,hv->hvk",
                normalized_key[0, token_offset + token_index],
                correction,
            )
            reference_output[0, token_offset + token_index] = torch.einsum(
                "hk,hvk->hv",
                normalized_query[0, token_offset + token_index],
                reference_state[sequence_index],
            ) * (key_dim**-0.5)
        token_offset += sequence_length

    output_delta = output.float() - reference_output
    state_delta = (
        initial_state[state_indices[:request_count]].float() - reference_state
    )
    output_max_abs = output_delta.abs().max().item()
    output_relative = (
        output_delta.norm() / reference_output.norm().clamp_min(1e-12)
    ).item()
    state_max_abs = state_delta.abs().max().item()
    if output_max_abs > 0.01 or output_relative > 0.02 or state_max_abs > 0.01:
        raise AssertionError(
            f"numerical gate failed: output_max_abs={output_max_abs:.8g}, "
            f"output_relative={output_relative:.8g}, state_max_abs={state_max_abs:.8g}"
        )
    if not torch.equal(state_indices[-1], sentinel_before):
        raise AssertionError("the -1 padding sentinel was modified")
    if not torch.equal(initial_state[-1], initial_state_copy[-1]):
        raise AssertionError("the sentinel state-pool row was modified")

    _emit(
        "child_complete",
        output_max_abs=output_max_abs,
        output_relative=output_relative,
        state_max_abs=state_max_abs,
        index_rows=len(expected_indices),
        sentinel=int(state_indices[-1].item()),
    )
    return 0


def _profile_d2h_once() -> dict[str, object]:
    import torch
    from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices

    torch.manual_seed(123)
    cumulative_lengths = torch.tensor([0, 1, 66], device="cuda", dtype=torch.int32)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        prepare_chunk_indices(cumulative_lengths, CHUNK_SIZE)
        torch.cuda.synchronize()
    memcpy_events = [
        event
        for event in profile.events()
        if "memcpy" in event.name.lower() or "dtoh" in event.name.lower()
    ]
    return {
        "stream": str(torch.cuda.current_stream()),
        "memcpy_event_count": len(memcpy_events),
        "memcpy_event_names": sorted({event.name for event in memcpy_events}),
    }


def _run_parent(output_path: Path) -> int:
    python = "/opt/venv/bin/python"
    script = Path(__file__).resolve()
    results = {
        "trials_per_case": TRIALS_PER_CASE,
        "trial_timeout_seconds": TRIAL_TIMEOUT_SECONDS,
        "cases": {},
        "profile": {},
    }
    for case in ("single_request", "multi_request"):
        case_results = []
        for trial in range(TRIALS_PER_CASE):
            command = [
                python,
                str(script),
                "--child",
                case,
                "--seed",
                str(1000 + trial),
            ]
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=TRIAL_TIMEOUT_SECONDS,
                    check=False,
                )
                case_results.append(
                    {
                        "trial": trial,
                        "returncode": completed.returncode,
                        "elapsed_seconds": time.monotonic() - started,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                case_results.append(
                    {
                        "trial": trial,
                        "returncode": "timeout",
                        "elapsed_seconds": time.monotonic() - started,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                )
            if case_results[-1]["returncode"] != 0:
                break
        results["cases"][case] = case_results

    profile_command = [python, str(script), "--profile-d2h"]
    completed = subprocess.run(
        profile_command,
        text=True,
        capture_output=True,
        timeout=TRIAL_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode == 0:
        results["profile"] = json.loads(completed.stdout)
    else:
        results["profile"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    all_ok = all(
        len(case_result) == TRIALS_PER_CASE
        and all(trial["returncode"] == 0 for trial in case_result)
        for case_result in results["cases"].values()
    )
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("single_request", "multi_request"))
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--profile-d2h", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("kda_gfx942_results.json"))
    args = parser.parse_args()
    if args.child:
        return _run_child(args.child, args.seed)
    if args.profile_d2h:
        print(json.dumps(_profile_d2h_once(), sort_keys=True))
        return 0
    return _run_parent(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
