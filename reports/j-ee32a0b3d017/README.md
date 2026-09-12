# CUDA graph request-capacity investigation: j-ee32a0b3d017

Upstream issue: https://github.com/sgl-project/sglang/issues/33483

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1862

## Result

The independent defaults are still present at base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. For the issue's measured token
capacity and context length, `resolve_max_num_reqs()` returns 4096, while the
46 GiB, TP=1 memory heuristic sets the largest decode graph batch to 32.

This change adds a one-time startup warning after the effective per-DP-worker
request capacity is known. It is emitted only when decode graphs are enabled
and admission can exceed the largest captured decode graph batch. It names the
two relevant flags and does not silently change scheduling or graph-memory
tradeoffs.

Upstream PR https://github.com/sgl-project/sglang/pull/33900 was inspected
before implementation. It remains open and warns only when a decode batch has
already crossed the graph ceiling. This patch is the distinct startup warning
suggested in the source issue, so operators can see the mismatch before load.

## Evidence

- `raw/regression-before.txt`: the exact 3,145,118-token / 32,768-context
  regression failed before the implementation because no warning was emitted.
- `raw/unit-tests-final.txt`: 44 tests and 7 subtests pass, covering the issue
  values plus equality and disabled-graph boundaries.
- `raw/gpu-graph-final/server.log`: a source-checkout server on the assigned
  MI355X (`gfx950`) reports effective request capacity 4 versus captured graph
  ceiling 2, emits the warning, captures batches `[1, 2]`, and serves requests.
- `raw/gpu-graph-final/probe-summary.json`: `/generate`, OpenAI completion,
  batched generation, and streaming completion all return HTTP 200.
- `raw/gpu-numerical-reference.json`: an independent Torch GPU dot product
  exactly matches the CPU reference (32.0), recording Torch, HIP, device, and
  architecture.

The GPU fixture is the deterministic random tiny Llama fixture from
amdpilot-org/sglang PR649 at exact commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Weights were generated under
`/tmp/amdpilot-repo-j-ee32a0b3d017/tiny-random-llama`, outside the checkout.

## Limitations

The assigned GPU is AMD MI355X/gfx950, not the reported NVIDIA L40, and the
Qwen2.5-0.5B weights plus ShareGPT workload were not available. Therefore this
work does not claim reproduction or removal of the reported absorbing latency
state. The tiny random Llama run validates startup integration, graph capture,
engine execution, and HTTP transport only. The patch is diagnostic and leaves
the scheduling and graph-capacity policies unchanged.
