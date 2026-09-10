from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    AcceptGreedy,
    FinalizeAcceptLens,
)
from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    CommitKvProj,
    _STACKED_WEIGHT_CACHE,
)
from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.models.dspark import (
    DSparkDraftMixin,
    VanillaMarkov,
    project_through_lm_head,
    run_markov_block,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)


DEFAULT_CASES = (
    {"name": "small", "batch": 1, "proposal": 2, "hidden": 1024, "vocab": 512},
    {"name": "medium", "batch": 4, "proposal": 4, "hidden": 2048, "vocab": 1024},
    {"name": "large", "batch": 8, "proposal": 4, "hidden": 4096, "vocab": 2048},
)


@dataclass
class CaseConfig:
    name: str
    batch: int
    proposal: int
    hidden: int
    vocab: int

    @property
    def tokens(self) -> int:
        return self.batch * (self.proposal + 1)


class FakeLinear(nn.Module):
    quant_method = None

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, inputs: torch.Tensor):
        return F.linear(inputs, self.weight), None


class FakeLmHead(nn.Module):
    quant_method = None

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight


class FakePool:
    def __init__(self, slots: int, heads: int, head_dim: int, device: torch.device) -> None:
        self.k_buffer = torch.zeros(
            slots, heads, head_dim, dtype=torch.bfloat16, device=device
        )
        self.v_buffer = torch.zeros_like(self.k_buffer)

    def set_kv_buffer(self, attn, cache_loc, key, value, k_scale, v_scale) -> None:
        self.k_buffer.index_copy_(0, cache_loc, key)
        self.v_buffer.index_copy_(0, cache_loc, value)


class FakeAttention:
    num_kv_heads = 1

    def __init__(self, hidden: int, head_dim: int, device: torch.device) -> None:
        self.head_dim = head_dim
        self.kv_size = head_dim
        self.k_weight = torch.randn(head_dim, hidden, dtype=torch.bfloat16, device=device)
        self.v_weight = torch.randn(head_dim, hidden, dtype=torch.bfloat16, device=device)
        self.k_norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
        self.k_linear = FakeLinear(self.k_weight)
        self.v_linear = FakeLinear(self.v_weight)
        self.attn = SimpleNamespace(k_scale=None, v_scale=None)

    def kv_proj_only(self, hidden_states: torch.Tensor):
        key, value = CommitKvProj.execute(
            main_x=hidden_states,
            wkv_linears=[self.k_linear, self.v_linear],
        )
        return key, value

    def apply_k_norm(self, key: torch.Tensor) -> torch.Tensor:
        return key * self.k_norm_weight

    def apply_k_rope(self, positions: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        scale = (1.0 + 0.001 * positions.float()).to(key.dtype).unsqueeze(1)
        return key * scale


class SyntheticState:
    def __init__(self, case: CaseConfig, device: torch.device) -> None:
        self.case = case
        self.device = device
        self.head_dim = 64
        self.target_projection_weight = torch.randn(
            case.hidden, case.hidden, dtype=torch.bfloat16, device=device
        )
        self.lm_head_weight = torch.randn(
            case.vocab, case.hidden, dtype=torch.bfloat16, device=device
        )
        self.attention = FakeAttention(case.hidden, self.head_dim, device)
        self.pool = FakePool(case.tokens, 1, self.head_dim, device)
        self.lm_head = FakeLmHead(self.lm_head_weight)
        self.markov_head = VanillaMarkov(
            vocab_size=case.vocab, markov_rank=64
        ).to(device=device, dtype=torch.bfloat16)
        self.draft_model = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=self.attention)],
            project_target_hidden=lambda hidden: F.linear(
                hidden, self.target_projection_weight
            ),
            _fused_kv_write_bundle=lambda _pool: None,
            _stacked_ctx_kv_params=lambda: None,
        )
        self.draft_model.write_target_hidden_kv = MethodType(
            DSparkDraftMixin.write_target_hidden_kv, self.draft_model
        )
        self.injector = TargetHiddenKvInjector(
            draft_model=self.draft_model,
            draft_model_runner=SimpleNamespace(token_to_kv_pool=self.pool),
            model_runner=SimpleNamespace(device=device),
            device=device,
            verify_num_draft_tokens=case.proposal + 1,
            block_pos_offsets=torch.arange(case.proposal + 1, device=device),
        )

    def weight_bytes(self) -> int:
        weights = (
            self.target_projection_weight,
            self.lm_head_weight,
            self.attention.k_weight,
            self.attention.v_weight,
            self.attention.k_norm_weight,
            self.markov_head.markov_w1.weight,
            self.markov_head.markov_w2.weight,
        )
        return sum(weight.numel() * weight.element_size() for weight in weights)


def make_inputs(case: CaseConfig, device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    proposal_offsets = torch.arange(case.proposal + 1, device=device)
    prefix_lens = torch.randint(
        10, 100, (case.batch,), dtype=torch.int32, device=device, generator=generator
    )
    return {
        "target_hidden": torch.randn(
            case.tokens, case.hidden, dtype=torch.bfloat16, device=device, generator=generator
        ),
        "draft_hidden": torch.randn(
            case.batch,
            case.proposal,
            case.hidden,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
        "first_prev_tokens": torch.randint(
            0, case.vocab, (case.batch,), dtype=torch.int64, device=device, generator=generator
        ),
        "prefix_lens": prefix_lens,
        "cache_loc": torch.randperm(case.tokens, device=device, generator=generator).long(),
        "positions": (prefix_lens[:, None] + proposal_offsets).reshape(-1).long(),
        "target_logits": torch.randn(
            case.tokens,
            case.vocab,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ),
    }


def run_block(inputs: dict[str, torch.Tensor], state: SyntheticState) -> dict[str, torch.Tensor]:
    case = state.case
    target_hidden = pack_aux_hidden_states([inputs["target_hidden"]])
    state.injector.inject_target_hidden(
        target_hidden=target_hidden,
        cache_loc=inputs["cache_loc"],
        positions=inputs["positions"],
    )
    base_logits = project_through_lm_head(inputs["draft_hidden"], state.lm_head)
    base_logits = base_logits.view(case.batch, case.proposal, case.vocab)

    def greedy(logits: torch.Tensor, _step: int) -> torch.Tensor:
        return logits.argmax(dim=-1)

    sampled_tokens, corrected_logits = run_markov_block(
        state.markov_head,
        base_logits,
        first_prev_tokens=inputs["first_prev_tokens"],
        hidden_states=None,
        sampler=greedy,
        collect_corrected=True,
    )
    candidates = torch.cat(
        (inputs["first_prev_tokens"][:, None], sampled_tokens), dim=1
    )
    correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
        candidates=candidates,
        target_logits=inputs["target_logits"],
        verify_num_draft_tokens=case.proposal + 1,
    )
    commit = FinalizeAcceptLens.execute(
        correct_len=correct_len,
        cap_trim_lens=cap_trim_lens,
        prefix_lens=inputs["prefix_lens"],
    )
    return {
        "k_buffer": state.pool.k_buffer,
        "v_buffer": state.pool.v_buffer,
        "base_logits": base_logits,
        "sampled_tokens": sampled_tokens,
        "corrected_logits": corrected_logits,
        "correct_len": correct_len,
        "bonus": bonus,
        "cap_trim_lens": cap_trim_lens,
        "commit_lens": commit.commit_lens,
        "new_seq_lens": commit.new_seq_lens,
        "commit_cap_trim_lens": commit.cap_trim_lens,
    }


def independent_reference(
    inputs: dict[str, torch.Tensor], state: SyntheticState
) -> dict[str, torch.Tensor]:
    case = state.case
    ctx_hidden = F.linear(inputs["target_hidden"], state.target_projection_weight)
    key = F.linear(ctx_hidden, state.attention.k_weight)
    value = F.linear(ctx_hidden, state.attention.v_weight)
    key = key * state.attention.k_norm_weight
    rope_scale = (1.0 + 0.001 * inputs["positions"].float()).to(key.dtype).unsqueeze(1)
    key = (key * rope_scale).to(torch.bfloat16)
    value = value.to(torch.bfloat16)
    expected_k = torch.zeros_like(state.pool.k_buffer)
    expected_v = torch.zeros_like(state.pool.v_buffer)
    expected_k.index_copy_(0, inputs["cache_loc"], key.view(-1, 1, state.head_dim))
    expected_v.index_copy_(0, inputs["cache_loc"], value.view(-1, 1, state.head_dim))

    base_logits = torch.matmul(
        inputs["draft_hidden"].to(state.lm_head_weight.dtype), state.lm_head_weight.T
    ).view(case.batch, case.proposal, case.vocab)
    previous = inputs["first_prev_tokens"].long()
    sampled_tokens = []
    corrected_logits = []
    for step in range(case.proposal):
        previous_embedding = state.markov_head.markov_w1(previous)
        step_bias = F.linear(previous_embedding, state.markov_head.markov_w2.weight)
        step_logits = base_logits[:, step, :] + step_bias
        next_token = step_logits.argmax(dim=-1)
        sampled_tokens.append(next_token)
        corrected_logits.append(step_logits.unsqueeze(1))
        previous = next_token
    sampled_tokens = torch.stack(sampled_tokens, dim=1)
    corrected_logits = torch.cat(corrected_logits, dim=1)
    candidates = torch.cat(
        (inputs["first_prev_tokens"][:, None], sampled_tokens), dim=1
    )
    target_predict = inputs["target_logits"].argmax(dim=-1).view(
        case.batch, case.proposal + 1
    )
    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1).to(torch.int32)
    rows = torch.arange(case.batch, device=state.device)
    bonus = target_predict[rows, correct_len.long()].to(torch.int64)
    cap_trim_lens = torch.zeros_like(correct_len)
    commit_lens = correct_len.to(torch.int32) + 1
    new_seq_lens = inputs["prefix_lens"] + commit_lens.to(inputs["prefix_lens"].dtype)
    return {
        "k_buffer": expected_k,
        "v_buffer": expected_v,
        "base_logits": base_logits,
        "sampled_tokens": sampled_tokens,
        "corrected_logits": corrected_logits,
        "correct_len": correct_len,
        "bonus": bonus,
        "cap_trim_lens": cap_trim_lens.to(torch.int32),
        "commit_lens": commit_lens,
        "new_seq_lens": new_seq_lens,
        "commit_cap_trim_lens": cap_trim_lens.to(torch.int32),
    }


def compare_outputs(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, bool]:
    results = {}
    for name, expected_tensor in expected.items():
        actual_tensor = actual[name]
        if actual_tensor.dtype.is_floating_point:
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2
            )
        else:
            torch.testing.assert_close(
                actual_tensor, expected_tensor, rtol=0, atol=0
            )
        results[name] = True
    return results


def capture_graph(
    run_graph: callable, warmup_runs: int
) -> tuple[torch.cuda.CUDAGraph, dict[str, torch.Tensor]]:
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(warmup_runs):
            run_graph()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run_graph()
    torch.cuda.synchronize()
    return graph, outputs


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def measure_mode(
    mode: str,
    input_sets: list[dict[str, torch.Tensor]],
    state: SyntheticState,
    static_inputs: dict[str, torch.Tensor],
    graph: torch.cuda.CUDAGraph | None,
) -> dict[str, object]:
    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for inputs in input_sets:
        if mode == "graph":
            for name, tensor in inputs.items():
                static_inputs[name].copy_(tensor)
        start.record()
        if mode == "eager":
            outputs = run_block(inputs, state)
        else:
            graph.replay()
            outputs = None
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
        if outputs is not None:
            independent = {
                name: tensor.clone() for name, tensor in outputs.items()
            }
            del independent
    summary = timing_summary(timings)
    return {"summary": summary, "raw_ms": timings}


def module_path(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    return str(spec.origin) if spec else None


def run_case(
    config: dict[str, int | str],
    device: torch.device,
    measured_runs: int,
    warmup_runs: int,
    validation_sets: int,
) -> dict[str, object]:
    case = CaseConfig(**config)
    _STACKED_WEIGHT_CACHE.clear()
    state = SyntheticState(case, device)
    first_seed = 10_000 + case.batch * 100 + case.proposal
    first_inputs = make_inputs(case, device, first_seed)
    static_inputs = {
        name: tensor.clone() for name, tensor in first_inputs.items()
    }

    def run_graph() -> dict[str, torch.Tensor]:
        return run_block(static_inputs, state)

    graph, captured_outputs = capture_graph(run_graph, warmup_runs)
    validation = []
    for offset in range(validation_sets):
        seed = first_seed + 101 + offset * 202
        inputs = make_inputs(case, device, seed)
        eager_outputs = run_block(inputs, state)
        reference = independent_reference(inputs, state)
        eager_comparison = compare_outputs(eager_outputs, reference)
        for name, tensor in inputs.items():
            static_inputs[name].copy_(tensor)
        graph.replay()
        torch.cuda.synchronize()
        graph_outputs = {
            name: tensor.clone() for name, tensor in captured_outputs.items()
        }
        graph_comparison = compare_outputs(graph_outputs, reference)
        validation.append(
            {
                "seed": seed,
                "eager_vs_reference": eager_comparison,
                "graph_vs_reference": graph_comparison,
            }
        )

    timing_inputs = [
        make_inputs(case, device, first_seed + 1_000 + index)
        for index in range(measured_runs)
    ]
    eager_timing = measure_mode(
        "eager", timing_inputs, state, static_inputs, None
    )
    graph_timing = measure_mode(
        "graph", timing_inputs, state, static_inputs, graph
    )
    speedup = eager_timing["summary"]["median_ms"] / graph_timing["summary"]["median_ms"]
    result = {
        "name": case.name,
        "dimensions": {
            "batch": case.batch,
            "proposal_tokens": case.proposal,
            "tokens": case.tokens,
            "hidden_size": case.hidden,
            "vocab_size": case.vocab,
            "kv_heads": 1,
            "head_dim": state.head_dim,
        },
        "dtypes": {
            "hidden_logits_kv": "bfloat16",
            "positions_cache_loc": "int64",
            "prefix_and_accept_lens": "int32",
            "tokens": "int64",
        },
        "synthetic_weight_bytes": state.weight_bytes(),
        "validation": validation,
        "timing": {
            "warmup_runs": warmup_runs,
            "measured_runs_per_mode": measured_runs,
            "eager": eager_timing,
            "graph_replay": graph_timing,
            "median_graph_speedup_x": speedup,
        },
        "memory": {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
    }
    del timing_inputs, graph, captured_outputs, state
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--validation-sets", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= len(DEFAULT_CASES) <= 6:
        raise ValueError("The reduced study allows at most six workload cases.")
    if not 1 <= args.measured_runs <= 100:
        raise ValueError("Measured runs must remain bounded between 1 and 100.")
    if not 1 <= args.warmup_runs <= 10:
        raise ValueError("Warmup runs must remain bounded between 1 and 10.")
    if not 1 <= args.validation_sets <= 5:
        raise ValueError("Validation sets must remain bounded between 1 and 5.")
    if not torch.cuda.is_available():
        raise RuntimeError("One CUDA/HIP GPU is required for this reduced study.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one assigned GPU, found {torch.cuda.device_count()}."
        )

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    torch.manual_seed(0)
    results = []
    for config in DEFAULT_CASES:
        torch.cuda.reset_peak_memory_stats()
        results.append(
            run_case(
                config,
                device,
                args.measured_runs,
                args.warmup_runs,
                args.validation_sets,
            )
        )
    payload = {
        "schema_version": 1,
        "label": "reduced DSpark block eager versus captured replay study",
        "source_commit": "0084030179bfba86bfeb6d43f7997d4076329d2c",
        "command": (
            "PYTHONPATH=/job/sglang/python /opt/venv/bin/python "
            "/job/sglang/reports/j-0f672a2e3c66/reduced_replay.py "
            f"--output {args.output} --measured-runs {args.measured_runs} "
            f"--warmup-runs {args.warmup_runs} --validation-sets {args.validation_sets}"
        ),
        "environment": {
            "python": "/opt/venv/bin/python",
            "torch": torch.__version__,
            "torch_path": module_path("torch"),
            "sglang_path": module_path("sglang"),
            "triton_path": module_path("triton"),
            "gpu": {
                "name": properties.name,
                "gcn_arch_name": properties.gcnArchName,
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            },
        },
        "limits": {
            "gpu_count": 1,
            "workload_cases": len(DEFAULT_CASES),
            "synthetic_weights_limit_bytes": 4 * 1024**3,
            "live_allocations_limit_bytes": 48 * 1024**3,
            "wall_limit_seconds": 7200,
        },
        "static_buffer_reuse": (
            "One torch.cuda.CUDAGraph per case captures the reduced block over "
            "preallocated static input tensors. Each replay copies a fresh input "
            "set into those same buffers, replays the complete block, and clones "
            "captured outputs outside the timed region."
        ),
        "accuracy_gates": {
            "integer_outputs": "rtol=0, atol=0",
            "bfloat16_outputs": "rtol=1e-2, atol=1e-2",
        },
        "unsupported_capture_boundaries": [
            "No model-specific DSpark constructor is instantiated: DSparkDraftMixin is a mixin and model classes require full HF configuration and checkpoint state; a synthetic adapter supplies the target-hidden projection boundary.",
            "The full DecodeCudaGraphRunner is not captured because it requires ModelRunner, attention-backend, KV-pool, and distributed runtime state; this study uses one direct torch.cuda.CUDAGraph per reduced case.",
            "The reduced fake KV pool does not expose the MLA set_swa_key_buffer_radix_fused_norm_rope path, so TargetHiddenKvInjector uses the generic set_kv_buffer path.",
            "No GLM weights are downloaded and no distributed or model-level throughput claim is made.",
        ],
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(results)}, indent=2))


if __name__ == "__main__":
    main()
