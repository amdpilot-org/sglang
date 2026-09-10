# Bounded MI300X Llama reduced-ModelRunner study

## Scope

This is a real SGLang offline `Engine` / `ModelRunner` forward-path study, not a
plain-Torch substitute. It uses a locally generated Llama-style configuration and
random BF16 weights, with no tokenizer, checkpoint download, or model download.
The independent plain-Torch implementation is used only as the numerical
reference.

The selected block bottleneck is attention dispatch. Four existing supported
configurations were measured:

- `triton`
- `torch_native`
- `aiter`
- `wave`

## Model and workload

- Vocabulary size: `512`
- Hidden size: `256`
- Intermediate size: `512`
- Hidden layers: `2`
- Attention heads: `4`
- KV heads: `2`
- Head dimension: `64`
- Maximum position embeddings: `512`
- Dtype: `bfloat16`
- Random parameters: `2,886,144` bytes
- Safetensors file: `2,888,376` bytes
- Batch size: `1`
- Sequence lengths: `8`, `32`, `64`, `128`, `256`, `480`

The original probe used a 512-token case. SGLang correctly rejected it because a
512-token prompt plus one generated token exceeds the 512-token context. The
largest case was therefore bounded to 480 tokens before the recorded run.

Each configuration runs 2 warmup and 10 measured real Engine forwards per case.
The complete study contains `288` real forwards, bounded by the task limits.

## Numerical control

The independent control is a plain-Torch BF16-to-FP32 Llama forward implemented
from the safetensor weights. It computes causal attention, GQA, RoPE, RMSNorm,
SiLU-gated MLP, and the LM head without SGLang model code.

The final gates are:

- Top-1 token must match exactly.
- The top-5 token set must match exactly.
- Maximum absolute top-5 logprob error must be `<= 0.05`.

Top-5 order is intentionally not gated because tied BF16 logprobs can reorder.
The initial strict order check falsely rejected Aiter on the 32-token case; the
raw outputs show the same top-1 token and unordered top-5 set. The order check
was corrected to top-1 plus unordered top-5 membership, while the unchanged
`0.05` logprob threshold was not relaxed. Gates were recomputed from the
already-recorded raw outputs without repeating GPU work.

All four configurations pass all six cases. The largest recorded logprob error is
below `0.0052`.

## Timing and uncertainty

Timing uses SGLang's `meta_info.e2e_latency` for one-token offline Engine
generation. This is a real ModelRunner forward path, but includes scheduler and
IPC overhead; it is not a pure attention-kernel timing.

Each case reports mean, median, p95, minimum, maximum, and standard deviation
from 10 measured calls on the shared MI300X. Raw values are retained in
`results.json`.

| Configuration | Case-median mean (ms) | Geometric mean vs Triton |
|---|---:|---:|
| `triton` | 8.556 | 1.000 |
| `torch_native` | 8.083 | 0.955 |
| `aiter` | 12.298 | 1.439 |
| `wave` | 13.322 | 1.559 |

`torch_native` is fastest in this reduced study. The largest p95/median ratio is
`2.268` (Aiter, 32-token case), reflecting a shared-hardware outlier. The result
is configuration- and workload-specific and makes no serving-wide speedup claim.

Peak device use is `45,411,729,408` bytes, below the 48 GiB live-allocation
limit. Random weights are far below the 4 GiB limit.

## Installed-source baseline

The required pre-clone baseline is saved at `/job/baseline-first.json`. It is
clearly labeled as installed-source evidence and is not proof for checkout
changes.

- First GPU execution: existing ROCm RoPE test, 41.445 seconds.
- Concrete installed-test error: stale `forward_hip` API.
- Supported neighboring control: `RotaryEmbedding.forward_cuda`.
- Control accuracy: zero absolute error versus `forward_native`.
- Control timing: 0.037577 ms mean over 30 CUDA-event-timed calls.

## Reproduction

From the repository root:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-9a07dd1052cc/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-9a07dd1052cc/inductor
export HF_HOME=/tmp/sglang-cache-j-9a07dd1052cc/hf

/opt/venv/bin/python reports/j-9a07dd1052cc/benchmark_reduced_model_runner.py \
  --configuration all \
  --output reports/j-9a07dd1052cc/results.json
```

A single configuration can be rerun with:

```bash
/opt/venv/bin/python reports/j-9a07dd1052cc/benchmark_reduced_model_runner.py \
  --configuration torch_native \
  --output /tmp/torch_native.json
```

The model and caches are generated under
`/tmp/sglang-cache-j-9a07dd1052cc/`, outside `JOB_WORKDIR`.

## Context and boundaries

- Upstream context: sgl-project/sglang issue 35003, the AMD 2026 Q3 roadmap.
- The issue had no comments at the time of the read-only check.
- Related changes referenced by the issue include PRs 19975, 23388, 25090, and
  33939; none were checked out or modified.
- Checkout base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- GPU: one AMD Instinct MI300X (`gfx942`).
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`.
- No upstream issue, PR, or comment was posted or changed.
- No full model weights, tokenizer, or alternate framework stack were downloaded.
