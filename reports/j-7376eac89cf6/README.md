# Abort cleanup correction investigation

This correction independently reviewed candidate
https://github.com/amdpilot-org/sglang/pull/1721 at exact commit
`38c4cbff0291764be04514426e2654e2e08d3b9f` and the counterexamples in
https://github.com/amdpilot-org/sglang/pull/1817 against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

The candidate's narrow source fix is valid and is preserved: a streaming
response background cleanup can execute before `GenerateReqInput` is
normalized, so `is_single` does not yet exist. Deriving request cardinality
from the already-public `rid` shape removes that exception. The regression
fails on base and passes after the fix, with single, batch, active, unstarted,
and empty cases covered.

The candidate does not resolve the original public endpoint claim. Static
control-flow evidence shows that `/abort_request` calls
`TokenizerManager.abort_request` directly and never calls the changed
`create_abort_task`. Independently repeated live GPU probes also returned HTTP
200 and cleanly ended the stream with an abort finish reason on both base and
the corrected tree; neither log contains the reported exception. HTTP 200
therefore remains an acknowledgement of local dispatch rather than downstream
cancellation. The current path exposes no completion acknowledgement, and the
reported endpoint failure was not reproduced, so this correction does not add
an unevidenced cross-process protocol.

Raw unit, source, GPU, request, response, runner, and server-log evidence is in
`evidence/`. The original model weights and CUDA hardware were unavailable;
the deterministic tiny Llama fixture qualifies transport and engine execution
only. See `result.json` for exact commands and limitations.
