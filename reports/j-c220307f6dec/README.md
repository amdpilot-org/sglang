# Investigation report: streaming disconnect cleanup

Upstream issue: https://github.com/sgl-project/sglang/issues/36333

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1190

## Outcome

The prepared `main` checkout at `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains the solution merged upstream in PR #35255 at merge commit
`f478b2bb2d582c09e7f1b4e49f0c2d039da8747a` (2026-09-04). No production
source change is justified.

The current implementation adds `dispatched` and `abort_sent` to `ReqState`.
`generate_request` calls `_release_req_states_on_failure`, which discards only
undispatched states and calls `abort_request` for dispatched states while
retaining them until the scheduler response cleans them up. The scheduler also
retries a deferred chunked-prefill abort after the request leaves the chunked
slot. This is the invariant missing from the regression described by the issue.

## Evidence

- `raw/focused_regressions.log`: 13 issue-specific and boundary tests passed.
  They cover cancellation after dispatch, pre-dispatch single/batch cleanup,
  parallel-sampling generated IDs, abort dispatch failure, deduplication, and
  the chunked-prefill transition race.
- `raw/unit_tests.log`: the two complete related test modules passed (28 tests
  and 3 subtests).
- `raw/http_disconnect/run-metadata.json`: a real SGLang server on the assigned
  gfx950 GPU accepted a long streaming OpenAI completion. The curl client was
  killed after receiving 4,248 bytes. `/get_load` reported one running request
  immediately after the kill and zero after 0.103 seconds. The server log had
  zero occurrences of `but the state was deleted in TokenizerManager` during
  the retained observation window.
- `raw/http_disconnect/server.log`, `request.json`, and `client-stream.txt` are
  the raw server, request, and partial-response evidence.
- `raw/http_disconnect_startup_blocked/` retains the first failed startup,
  which rejected an intentional 4096-token probe against the fixture's declared
  256-token context. The successful runner sets SGLang's explicit
  `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` override because this is a
  transport/lifecycle probe, not a semantic-quality test.
- `raw/http_disconnect_runner_bug/` retains the first successful engine run;
  its reporting code mishandled `/get_load`'s list response. The checked-in
  runner normalizes that response, and the final run exits successfully.

## Reproduction

Generate the qualified deterministic tiny Llama fixture using
`reports/j-1fc9870820b9/create_tiny_llama.py` from amdpilot-org/sglang PR649 at
exact commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`, with
`SGLANG_QUAL_FIXTURE` pointing outside the checkout. Then run:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
  /tmp/amdpilot-repo-j-c220307f6dec/venv/bin/python \
  reports/j-c220307f6dec/run_disconnect_probe.py \
  /tmp/amdpilot-repo-j-c220307f6dec/tiny-random-llama \
  reports/j-c220307f6dec/raw/http_disconnect
```

## Limitations

The deterministic tiny random Llama fixture validates HTTP transport and real
engine execution only. It does not validate semantic output, GLM-5.3 or the
production models mentioned in issue comments, DSPARK/speculative decoding,
TP=2, or a multi-node topology. No production model weights were available.
The server required SIGKILL after its 30-second graceful-shutdown timeout, but
the subreaper runner reaped descendants and confirmed the owned process group
was gone; this happened after the disconnect/load evidence was collected.

