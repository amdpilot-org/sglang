# Independent review of PR 3300

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3300 at exact commit `a5b60944c1762c4b9196fec907c2a72a58e9143b`

Upstream issue: https://github.com/sgl-project/sglang/issues/28157

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3303

## Conclusion

Request changes. The candidate's subprocess isolation, singleflight rejection, and timeout address the original synchronous `/metrics` event-loop starvation mechanism. On the recorded base, a deterministic 350 ms in-process collection delayed unrelated async work by 332 ms; at the candidate commit, the same probe measured 1 ms of event-loop lag. The three counterexamples inherited from the prior review also pass for ordinary single-value request headers.

However, the candidate does not fully preserve Prometheus ASGI negotiation for repeated HTTP header fields. Prometheus joins every `Accept` and `Accept-Encoding` field before negotiation. The replacement endpoint uses `request.headers.get(...)`, which returns only the first field. An independent request with two `Accept` fields therefore produced OpenMetrics from the reference app but Prometheus 0.0.4 from the candidate. Likewise, two `Accept-Encoding` fields produced gzip from the reference and no compression from the candidate. These are legal HTTP representations equivalent to comma-joined field values.

This protocol counterexample is separate from the original starvation mechanism: the source-level original fix is effective, but the candidate's stated exposition-compatibility correction remains incomplete.

## Evidence

- `evidence/base-starvation.txt`: failing-before result on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` (`event_loop_lag_seconds=0.332`, exit 1).
- `evidence/candidate-starvation.txt`: passing-after result at exact candidate commit (`event_loop_lag_seconds=0.001`, exit 0).
- `evidence/candidate-contracts-run.txt`: candidate passes `name[]`, gzip, and OpenMetrics checks for single-value headers.
- `evidence/adversarial-contracts.txt`: repeated-header comparison fails; reference negotiates OpenMetrics and gzip while candidate does not.
- `evidence/candidate-unit-tests.txt`: five focused candidate tests passed.
- `evidence/candidate-static.txt`: changed Python files compile and `git diff --check` passes.
- `evidence/candidate-import-path.txt`: confirms tests imported `/job/repo/python/sglang/srt/utils/common.py` from the checked-out source.
- `evidence/environment.txt`: records the prepared interpreter and ROCm/GPU visibility.

## Reproduction

Use the prepared interpreter. The base probe is expected to exit 1 on the recorded base and 0 on the candidate:

```bash
/tmp/amdpilot-repo-j-4d39ca524529/venv/bin/python reports/j-4d39ca524529/reproduce_starvation.py
```

At candidate commit `a5b60944c1762c4b9196fec907c2a72a58e9143b`, the ordinary exposition contract test passes:

```bash
/tmp/amdpilot-repo-j-4d39ca524529/venv/bin/python reports/j-4d39ca524529/candidate_contracts.py
```

The independent repeated-header comparison is expected to exit 1 on the candidate:

```bash
/tmp/amdpilot-repo-j-4d39ca524529/venv/bin/python reports/j-4d39ca524529/adversarial_contracts.py
```

## Limitations

The original eight-H100, multi-node, single-tokenizer PD deployment and decode-side connection-loss logs were unavailable. The prepared host exposed one AMD GPU through PyTorch 2.11.0+rocm7.2; no GPU execution was needed for this CPU-side ASGI/Prometheus behavior. No model weights or tiny-model server were used because they would not strengthen the deterministic event-loop and HTTP negotiation evidence. The candidate changes only Python, so no native code was changed or rebuilt.
