# Independent review of candidate PR 2110

- Upstream issue: https://github.com/sgl-project/sglang/issues/32475
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2063
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2144
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Candidate reviewed: `d1822c9272066f86263a87de1c8f7bfb39abce3e`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue.
It changes the two `kv_*_blocks` metrics from token capacity to the scheduler's
existing floor-divided page capacity, and passes `page_size` from the only
production constructor call site. This is a production fix with regression
coverage, not test-only hardening.

On the recorded base, an independent call through the loaded
`SchedulerKvEventsPublisher.emit_kv_metrics` implementation emitted 8192 total
blocks and 4096 active blocks for 8192 tokens at 50% usage. With page size 16,
the contract requires 512 and 256. At the exact candidate commit, the same
reported example and five independent boundary cases all matched an independent
integer page-count reference.

The candidate regression passed (one test, four subtests), adjacent KV-event
coverage passed (16 tests, 22 subtests), and all three changed Python files
compiled. The loaded module path was
`/job/repo/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py`,
so validation exercised the checked-out source rather than an installed copy.

## Review note

The candidate PR prose says its complete diff passes `git diff --check`, but an
independent check reports trailing whitespace in committed raw pytest logs.
This does not affect production code, tests, or the original metric contract,
so it is not a remaining functional counterexample.

## Architecture and limitations

The assigned device is one AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and
Torch 2.11.0+rocm7.2. GPU execution was intentionally not used as proof: the
defect is CPU-side arithmetic and socket payload construction, with no tensor,
kernel, model architecture, weights, serving transport, or distributed path.
No C++/HIP/native source changed, and repository metadata reports no prepared
native rebuild target, so a native rebuild was not applicable. No full model
or multi-node claim is made.

Raw outputs and the reviewed diff are retained under `raw/`.
