# Correction investigation for late prefill allocation recovery

- Upstream issue: https://github.com/sgl-project/sglang/issues/34676
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1686
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/1576
- Independent review PR: https://github.com/amdpilot-org/sglang/pull/1651
- Candidate commit reproduced: `b6ca9a3b9c47a1faaced6e588e46faec4693717f`
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding and correction

The review counterexample reproduced at the exact candidate commit. Starting
with an active `Scheduler.chunked_req`, an injected `KVCacheOOMError` from the
real `_get_new_batch_prefill_raw()` recovery path rolled back
`inflight_middle_chunks` but also appended that same request to
`waiting_queue`. This gave one request two scheduler owners and exposed it to
both `PrefillAdder.add_chunked_req()` and the ordinary waiting-queue loop on the
next pass.

The candidate's valid fixes were preserved: typed KV allocation failures,
cleanup of newly acquired request-pool slots, scheduler recovery for late
prefill misses, and hybrid-Mamba admission boundaries. The correction records
whether the failed batch contains a pre-existing active chunk. That request
remains owned only by `Scheduler.chunked_req`; other admitted requests are
returned to the waiting queue. A chunk first created in the failed round is
still cleared from `Scheduler.chunked_req` and requeued.

## Evidence

At the candidate commit, the new ownership regression failed. Raw output:
`reports/j-0796a3a0b351/raw/candidate_counterexample_before.txt`.

With the correction, the same regression passed. Raw output:
`reports/j-0796a3a0b351/raw/candidate_counterexample_after.txt`.

The complete candidate-focused suite plus the new active-chunk, mixed-batch,
and new-chunk boundary tests passed: 37 tests and 12 subtests. Raw output:
`reports/j-0796a3a0b351/raw/focused_suite_after.txt`.

## Limitations

The exact Kimi-K3 TP8/DCP4 sustained-traffic workload was not run. The job has
one AMD Instinct MI350X (`gfx950`) rather than eight NVIDIA B300 GPUs, and no
Kimi-K3 weights were provided. The tiny Llama transport fixture cannot
exercise hybrid Mamba allocation and was not used as substitute evidence.
The deterministic tests validate Python scheduler/allocation ownership and
recovery only. No model-semantic, distributed, or GPU numerical claim is made.
No native rebuild was required because no native source changed.
