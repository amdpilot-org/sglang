# Independent review of PR 2758

Candidate: https://github.com/amdpilot-org/sglang/pull/2758  
Exact candidate commit: `6834184e7f6eb2b51fdffd277d08c5472be0f0f6`  
Upstream issue: https://github.com/sgl-project/sglang/issues/33504  
Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2730  
Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2792

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's stated contract for the shared Python OpenAI serving path. This is a source fix with a discriminating regression, not test-only hardening.

At recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, `OpenAIServingBase.create_error_response` emitted a flat JSON object, while the streaming helper emitted the same error under `{"error": ...}`. The candidate wraps only the non-streaming response content, preserving the HTTP status and every inner field.

The candidate's three new regression cases all failed unchanged against the recorded base and all passed at the exact candidate commit. The focused suite at the candidate passed: 160 tests plus 68 subtests. Independent cases verified empty strings, Unicode, 401/422 statuses, nullable parameters, content type, and an actual official OpenAI SDK request/error path using an HTTP transport mock. No remaining counterexample tied to the reported response-envelope contract was found.

## Checkout and provenance

- Prepared checkout initially matched the requested recorded base exactly: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Candidate inspection and tests used detached commit `6834184e7f6eb2b51fdffd277d08c5472be0f0f6` fetched from the maintained mirror.
- Imports resolved to `/job/repo/python/sglang/__init__.py` and `/job/repo/python/sglang/srt/entrypoints/openai/serving_base.py`, proving the checked-out source was tested rather than an installed copy.
- Before this report commit, the prepared branch was restored and fast-forwarded to current `origin/main` (`bd45cd50ca900dd821f829ca9adfbf9aa3336bda`) as required for the review PR. Current main still has no change to `serving_base.py` relative to the recorded base.
- Upstream PR https://github.com/sgl-project/sglang/pull/33534 was also open and proposed the same narrow source correction when inspected.

## Evidence and commands

Base reproduction:

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-28497160325c/venv/bin/python ...
nonstream 400 {"object":"error","message":"max_tokens=999999 cannot be greater than limit","type":"BadRequestError","param":"max_tokens","code":400}
stream {"error": {"object": "error", "message": "max_tokens=999999 cannot be greater than limit", "type": "BadRequestError", "param": null, "code": 400}}
```

Candidate regression applied unchanged to the base:

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-28497160325c/venv/bin/python -m pytest -q /tmp/j-28497160325c-evidence/test_serving_base_candidate.py
3 failed
```

Exact candidate focused suite:

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-28497160325c/venv/bin/python -m pytest -q \
  test/registered/unit/entrypoints/openai/test_serving_base.py \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  test/registered/unit/entrypoints/openai/test_serving_transcription.py
160 passed, 61 warnings, 68 subtests passed in 9.86s
```

Independent adversarial check:

```text
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-28497160325c/venv/bin/python /tmp/j-28497160325c-evidence/adversarial.py
serializer 400 {"error": {"object": "error", "message": "", "type": "BadRequestError", "param": "", "code": 400}}
serializer 422 {"error": {"object": "error", "message": "unicode: café 🚀", "type": "ValidationError", "param": null, "code": 422}}
serializer 401 {"error": {"object": "error", "message": "unauthorized", "type": "AuthenticationError", "param": "api_key", "code": 401}}
sdk BadRequestError {'object': 'error', 'message': 'max_tokens=999999 cannot be greater than the model context length', 'type': 'BadRequestError', 'param': 'max_tokens', 'code': 400}
```

Raw outputs are retained under `reports/j-28497160325c/raw/`.

## Architecture and environment limitations

The assigned device is an AMD Instinct MI355X reporting `gfx950`. GPU execution was not used because this change is exclusively Python JSON serialization and does not enter a GPU kernel. No C/C++/HIP/FlyDSL/native source changed, so a native rebuild was neither required nor performed.

No model weights or live SGLang model server were used. Therefore this review does not claim model semantic accuracy, architecture-specific model behavior, or distributed/multi-node coverage. The checked shared serializer, real serving caller unit tests, and official SDK transport behavior directly cover the original response-envelope defect; a full model startup would not add issue-specific evidence to that contract.
