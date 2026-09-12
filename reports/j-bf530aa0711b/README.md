# Rust TreeCore concurrency investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38536

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3317

Outcome: **not reproduced** at base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

No production change is proposed. The reported v0.5.19 result used
Qwen2.5-1.5B-Instruct, the Qwen tool parser, an RTX 4090, and concurrency up to
256. Those weights and that GPU architecture were not available. The qualified
random tiny Llama fixture can test the serving transport and engine execution,
but cannot qualify Qwen tool parsing, model semantics, or the original hardware.

## Source review

The original Rust TreeCore commit (`9cf157c2`, PR #32710) already wrapped hot
binding calls such as `match_prefix`, `insert`, lock operations, and eviction in
PyO3 `allow_threads`. The same behavior is present in v0.5.19 and this base, so
the report does not establish a missing-GIL-release defect that can be narrowly
patched. Later related changes through this base primarily harden parity and add
features; no existing commit was identified as an issue-specific performance
fix.

## Measurements

The Rust extension was built from this checkout. A five-round alternating-order
GPU-backed cache benchmark used 1,200 generated sequences, a 96-token chunk,
Full attention, page size 1, and correctness checks. Median Rust throughput was
higher than Python for every measured operation:

| operation | Python ops/s | Rust ops/s | Rust delta |
|---|---:|---:|---:|
| match prefix | 35,393 | 81,284 | +129.7% |
| lock/unlock | 45,744 | 268,055 | +486.0% |
| cache finished | 6,511 | 10,370 | +59.3% |

Two fresh server sessions per backend were run in alternating order on the
assigned MI350X (`gfx950`). Each session issued three streaming rounds at
concurrency 8 and 64 with 12 events per request. The first round at each level
was excluded as compilation/warmup; medians over the remaining four rounds were:

| concurrency | metric | Python | Rust | Rust delta |
|---:|---|---:|---:|---:|
| 8 | e2e p50 | 40.58 ms | 37.89 ms | -6.6% |
| 8 | TTFT p50 | 12.97 ms | 12.42 ms | -4.3% |
| 8 | stream span p50 | 25.56 ms | 25.02 ms | -2.1% |
| 64 | e2e p50 | 71.03 ms | 69.40 ms | -2.3% |
| 64 | TTFT p50 | 20.66 ms | 19.90 ms | -3.7% |
| 64 | stream span p50 | 49.02 ms | 47.90 ms | -2.3% |

These tiny-fixture results reject a general Rust TreeCore concurrency slowdown
in the checked-out source, but do not disprove a model-, parser-, version-, or
RTX-4090-specific regression in the original report.

## Reproduction

Use the interpreter from `REPOSITORY.md`; keep the fixture outside the worktree:

```bash
export SGLANG_ISSUE_38536_FIXTURE=/tmp/amdpilot-repo-j-bf530aa0711b/tiny-random-llama
/tmp/amdpilot-repo-j-bf530aa0711b/venv/bin/python \
  reports/j-bf530aa0711b/create_tiny_llama.py
/tmp/amdpilot-repo-j-bf530aa0711b/venv/bin/python \
  reports/j-bf530aa0711b/run_server_benchmark.py \
  --fixture "$SGLANG_ISSUE_38536_FIXTURE" --backend python \
  --output /tmp/treecore-python
/tmp/amdpilot-repo-j-bf530aa0711b/venv/bin/python \
  reports/j-bf530aa0711b/run_server_benchmark.py \
  --fixture "$SGLANG_ISSUE_38536_FIXTURE" --backend rust \
  --output /tmp/treecore-rust
```

Complete server logs, requests-derived timing output, backend-selection log
lines, native test output, GPU cleanup evidence, and the direct cache benchmark
are retained under `raw/`.
