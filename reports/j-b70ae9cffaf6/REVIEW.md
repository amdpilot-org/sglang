# Independent review of PR 1579

Candidate: https://github.com/amdpilot-org/sglang/pull/1579  
Exact commit: `163165746ac4235926508ad617b5f19edcb0e14c`  
Upstream issue: https://github.com/sgl-project/sglang/issues/34716  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/1615

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's
serialization contract at the common final SSE serialization seam. This is a
source fix with focused regression coverage, not test-only hardening.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the prepared
OpenAI SDK coerced an integer `created_at` to `float` and emitted
`1786588600.0`. At the exact candidate commit, the focused 10-test suite passed.
An independent check confirmed integer JSON values for
`response.created`, `response.in_progress`, `response.completed`, and
`response.failed`, and exact passthrough for a non-snapshot Unicode delta event.

The candidate changes both Harmony and non-Harmony SSE send functions to call
the same serializer. No native source changed; a native rebuild was therefore
not applicable. Imports resolved to `/job/repo/python/sglang` with the prepared
interpreter.

## Commands and evidence

```text
# Failing-before SDK reproduction on the recorded base
/tmp/amdpilot-repo-j-b70ae9cffaf6/venv/bin/python - <<'PY'
... construct ResponseCreatedEvent with created_at=1786588600 ...
PY
# observed: float; JSON contained "created_at":1786588600.0

# Candidate regression suite
/tmp/amdpilot-repo-j-b70ae9cffaf6/venv/bin/python \
  test/registered/unit/entrypoints/openai/test_serving_responses_stream.py
# exit 0; Ran 10 tests; OK

# Independent candidate cases
PYTHONPATH=python /tmp/amdpilot-repo-j-b70ae9cffaf6/venv/bin/python <adversarial script>
# exit 0; all four response snapshot event classes: float internally, int on wire;
# non-response delta serialization exactly matched model_dump_json(indent=None)

/tmp/amdpilot-repo-j-b70ae9cffaf6/venv/bin/python -m py_compile \
  python/sglang/srt/entrypoints/openai/serving_responses.py \
  test/registered/unit/entrypoints/openai/test_serving_responses_stream.py
# exit 0

git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365..163165746ac4235926508ad617b5f19edcb0e14c
# exit 0
```

Raw logs captured during revision switching were preserved outside the checkout
under `/job/review-evidence/raw/`.

## Limitations

- No live model server was launched. The issue is fully reproduced and tested
  at the actual source serialization seam, independent of model execution.
- No GPU kernel was executed. One gfx950 GPU was visible through ROCm 7.2, but
  GPU output is unrelated to this Pydantic/JSON wire-type defect.
- The non-Harmony generator received end-to-end fixture coverage. Harmony was
  verified by source inspection to use the same helper, not by a Harmony model.
- The prepared environment has OpenAI 2.6.1 and Pydantic 2.13.5, rather than the
  reporter's 2.48.0/2.13.4. The coercion reproduced identically.

No remaining counterexample tied to the original contract was found.
