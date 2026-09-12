# Correction review of PR 1719

Reviewed candidate https://github.com/amdpilot-org/sglang/pull/1719 at exact commit
`eba02374b616ef4a417dded3b89b1b36f0fc84c4`, using the independent review
https://github.com/amdpilot-org/sglang/pull/1812 as untrusted diagnostic input.

## Result

The candidate's ordinary final-prefill cancellation fixes are preserved. The
review's beam-group counterexample is valid: when a final-prefill request has
both `to_finish` and `beam_group`, the candidate enters
`beam_coordinator.commit_prefill()` before its `drop_prefill_result` branch.
The request therefore remains unfinished in the deterministic fixture and the
pending abort is not consumed.

The correction gives the already-computed drop condition precedence over the
beam relay. The regression also verifies that the request finishes with the
original pending reason, emits no token or metadata, and never calls
`commit_prefill`. Existing live-request logprob-offset coverage remains active.

## Evidence

Failing on the candidate branch ordering, with the new regression present:

```text
FAILED TestPrefillHiddenStateOffsets::test_pending_abort_wins_over_beam_prefill_commit
AssertionError: False is not true
1 failed, 17 warnings in 9.05s
```

The failure occurs after `commit_prefill(req, up_to_tick=7)` is selected; the
mock beam coordinator does not consume `req.to_finish`, so the request remains
unfinished.

Passing after the correction:

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-7f2f098353f3/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_scheduler_chunked_req_gate.py \
  test/registered/unit/managers/test_scheduler_chunked_abort_race.py \
  test/registered/unit/managers/test_batch_result_processor_hidden_states.py \
  test/registered/unit/managers/test_batch_result_processor_spec_grammar.py

23 passed, 17 warnings, 2 subtests passed in 9.68s
```

The assigned GPU was visible and executed a tensor operation using Torch
2.11.0+rocm7.2 on one AMD Instinct MI355X (`gfx950`), producing the expected
sum `5.0`. This is environment evidence only; the regression is CPU control
flow.

## Remaining limitation

The reported `meta-llama/Llama-3.2-1B-Instruct` weights and NVIDIA RTX
4090/CUDA setup were unavailable. Therefore the original HTTP workload and its
negative scheduler `output_ids` observations were not reproduced or ruled out.
Current main stores per-request output tokens in `Req.output_ids`; no evidence
from this environment justifies an additional speculative negative-ID source
change. The qualified tiny Llama fixture cannot establish the missing model- or
workload-specific claim, so it was not substituted for that reproduction.
