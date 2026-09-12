# Independent review of candidate PR 2993

Upstream issue: https://github.com/sgl-project/sglang/issues/11186

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2924

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3026

Candidate: https://github.com/amdpilot-org/sglang/pull/2993 at
`d0666577d91c31b154d6492cdb4eaacef2fddb1c`.

## Recommendation

Request changes. The candidate is a real partial fix: on the normal Python
serving path it adds a full `LoadSnapshot` to generation and embedding output
messages, preserves it through detokenization and multi-tokenizer fanout,
caches the newest snapshot per rank in the HTTP worker, and merges that cache
with watch-mode data. Its focused suite passes, and the same new regression
suite fails on the recorded base.

It does not fully implement the requested mechanism for the optional Rust
egress architecture. `SchedulerOutputStreamer` sends generation payloads to
`RustServer.push_generation` instead of the Python detokenizer in that mode.
Although the candidate attaches `payload.load_snapshot` before this branch,
`RustServer.push_generation` neither serializes nor consumes that field, and
the Python `TokenizerManager.handle_loop` which records piggyback snapshots is
bypassed. The existing watch reader may still answer `/v1/loads`, but that is
the pre-existing fallback, not piggyback reporting through token messages.

## Evidence

- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- The prepared checkout was exactly at that commit before review; no image
  checkout difference was observed.
- Candidate regression on base: 4 failed, demonstrating the feature was absent.
- Candidate focused suite at the exact candidate commit: 44 passed and 19
  subtests passed.
- Independent Rust-route contract check: 1 failed because
  `RustServer.push_generation` contains no `load_snapshot` handling.
- Source imports resolved to `/job/repo/python/sglang/...`; Torch resolved to
  `/opt/venv/lib/python3.12/site-packages/torch`, version
  `2.11.0+rocm7.2`, HIP `7.2.26015`.
- One assigned AMD Instinct MI350X was visible. The review did not run a model
  serving workload because the decisive counterexample is an egress encoding
  omission and does not require numerical GPU execution.
- No native C++, HIP, CUDA, or Rust source changed in the candidate, so no
  native rebuild was applicable.

Raw logs, the candidate diff, fetched issue/PR metadata, import-path output,
and the independent adversarial test are retained outside the revision-switched
checkout at `/job/review-evidence-j-1d699ebf3fcb/`.

## Remaining counterexample

Launch with Rust egress enabled and stream a generation response. The scheduler
constructs a `BatchTokenIDOutput` carrying `load_snapshot`, but the Rust egress
encoder drops that field and bypasses the Python HTTP-worker cache update.
Consequently the response does not piggyback the load report to an HTTP
consumer; any `/v1/loads` freshness comes only from watch mode.
