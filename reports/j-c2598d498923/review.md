# Independent review of PR 859

Candidate: https://github.com/amdpilot-org/sglang/pull/859 at exact commit `924a36e64c82d79df51d1609c0bda97dacc834e4`

Upstream issue: https://github.com/sgl-project/sglang/issues/38002

Mirror issue: https://github.com/amdpilot-org/sglang/issues/895

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue within the directly exercised request-lifecycle contract. No remaining counterexample was found.

## Base reproduction

The prepared checkout was already on `amdpilot/j-c2598d498923` at the required recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference between the image-prepared checkout and recorded base.

Using `/tmp/amdpilot-repo-j-c2598d498923/venv/bin/python` with `PYTHONPATH=python:test/registered/unit/entrypoints/openai`, imports resolved to:

- `/job/repo/python/sglang/srt/entrypoints/openai/protocol.py`
- `/job/repo/python/sglang/srt/entrypoints/openai/serving_responses.py`

On the base, `ResponsesResponse` had no `background` field. The issue was reproduced for explicit `background=false` and omitted background. Terminal foreground responses returned HTTP 200 unchanged. More severely, synthetic queued/in-progress foreground responses returned HTTP 200, changed to `cancelled`, and invoked `abort_request` once.

## Exact candidate inspection

The candidate was fetched and temporarily checked out detached at the required exact SHA. It adds the request's `background` value to `ResponsesResponse.from_request()` and rejects `response.background is not True` before terminal-status handling. The diff contains no C++, HIP, CUDA, FlyDSL, or other native source changes, so no native rebuild was applicable.

The candidate source imports were confirmed to resolve to the same checkout paths above, and `ResponsesResponse.model_fields` contained `background`.

## Candidate and adversarial validation

The candidate's focused suite passed:

```text
58 passed, 7 warnings, 2 subtests passed in 9.11s
```

Independent cases covered the cross product of background omitted/false/true and statuses queued, in_progress, completed, cancelled, failed, and incomplete:

- Omitted or false background: HTTP 400, exact state preserved, zero abort calls.
- True background with queued/in-progress: HTTP 200, state changed to cancelled, exactly one abort call.
- True background with every tested terminal status: HTTP 200, state preserved, zero abort calls.
- Unknown response ID: HTTP 404 and zero abort calls.
- Legacy/externally constructed response with `background=None`: HTTP 400, state preserved, zero abort calls (fail-closed behavior).

This independently verifies the original reported case, the active foreground boundary that could otherwise abort incorrectly, valid background cancellation, terminal idempotency, and missing metadata behavior.

## Limitations

No GPU, live HTTP server, model weights, full model architecture, semantic accuracy, distributed workload, or multi-node execution was used. Those are not involved in this in-memory API control-flow defect. The test imports initialized the environment's existing AITer extension, but the reviewed change did not touch or rebuild native code. Raw command logs were preserved during revision switching under `/tmp/amdpilot-review-j-c2598d498923/`; the durable claims are recorded in `result.json` and this report.
