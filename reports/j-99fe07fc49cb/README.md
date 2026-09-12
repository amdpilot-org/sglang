# Abort cleanup correction review: j-99fe07fc49cb

Outcome: **candidate_rejected** as a full fix, with its valid narrow correction
preserved.

This correction independently reviewed candidate
https://github.com/amdpilot-org/sglang/pull/1925 at exact commit
`13901903c9cf5d6f7ffe304888a189f41dbf7ec6` and review
https://github.com/amdpilot-org/sglang/pull/2019.

The candidate regression fails on its exact parent (the prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`) with three
`AttributeError: 'GenerateReqInput' object has no attribute 'is_single'`
failures, and passes 4/4 on the exact candidate. The correction therefore
preserves the candidate's narrow change: streaming cleanup derives request
cardinality from the public `rid` shape, which exists before request
normalization sets `is_single`.

The review's remaining counterexamples were also confirmed. The public
`/abort_request` endpoint is identical on base and candidate and directly
calls `TokenizerManager.abort_request`; it does not invoke
`create_abort_task`. HTTP 200 acknowledges local dispatch only. Identical
real-GPU HTTP probes on base and corrected code both aborted the active stream
cleanly, and neither server log contained the claimed exception. There is no
issue-specific evidence justifying a speculative downstream completion-
acknowledgement protocol.

## Results

- Exact candidate regression on parent/base: 3 failed, 1 passed.
- Exact candidate regression at `13901903...`: 4 passed.
- Consolidated corrected focused regression: 4 passed.
- Consolidated corrected full related test file: 30 passed and 3 subtests passed.
- Base GPU/HTTP probe: HTTP 200; stream finished as aborted after 3 tokens.
- Corrected GPU/HTTP probe: HTTP 200; stream finished as aborted after 4 tokens.

Raw commands/results, server logs, HTTP payloads, GPU identity, fixture digest,
and source control-flow extraction are retained under `evidence/`.

## Limitations

The reported `meta-llama/Llama-3.2-1B-Instruct` weights and RTX 4090/CUDA
environment were unavailable. Live testing used one AMD MI350X/gfx950 with
ROCm 7.2 and the qualified deterministic tiny random Llama fixture. This
validates transport and engine execution only; it does not qualify semantic
accuracy, the original model setup, a different GPU architecture, or a
distributed workload. Both owned server groups exceeded the runner's graceful
shutdown timeout and were killed and reaped. No native source changed, so no
native rebuild was applicable.
