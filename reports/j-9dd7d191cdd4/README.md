# Independent review of PR 2315

Reviewed candidate commit `12421813cdd2059cfe5aa105559c62b726392b36` against upstream issue https://github.com/sgl-project/sglang/issues/31404 and candidate mirror issue https://github.com/amdpilot-org/sglang/issues/2251.

## Recommendation

Request changes. The candidate is a verified partial fix for the Python OpenAI serving path, but it does not fully resolve the original issue.

The candidate regression failed on the recorded base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) with four missing-helper failures and passed at the exact candidate commit (4 tests plus 2 subtests). Source imports resolved to `/job/repo/python/sglang/...`, confirming the checked-out source was tested.

The remaining original-issue counterexample is the model-gateway HTTP path. At the candidate commit, `sgl-model-gateway/src/routers/router_manager.rs::resolve_model_id` still returns every explicitly supplied model ID without checking `worker_registry.get_models()`. The candidate has no model-gateway diff, so a bogus explicit model still crosses this validation boundary. This path was expressly included in the original report.

There is also an exact response-contract difference: the issue's expected error has `"param": null`, while the candidate returns and tests `"param": "model"`. The important 404 / `model_not_found` behavior is otherwise implemented for Python handlers using `OpenAIServingBase.handle_request`, with separate Responses wiring.

## Commands and retained evidence

- Base: `PYTHONPATH=python .../venv/bin/python -m pytest -q /job/review-evidence-j-9dd7d191cdd4/test_candidate.py` — 4 failed because the fix was absent.
- Candidate: `PYTHONPATH=python .../venv/bin/python -m pytest -q test/registered/unit/entrypoints/openai/test_serving_model_validation.py` — 4 passed, 2 subtests passed.
- `git diff BASE..CANDIDATE -- sgl-model-gateway` — empty.
- Direct inspection of candidate `resolve_model_id` confirmed `Some(id) => Ok(id.to_string())` remains.

Raw logs were retained outside the revision-switching checkout at `/job/review-evidence-j-9dd7d191cdd4/` during review.

## Environment limitations

One AMD Instinct MI350X (`gfx950`) was visible with Torch 2.11.0+rocm7.2. This review did not rerun the candidate's full tiny-Llama server probe or claim model semantic coverage. No native/C++ files changed, so a native rebuild was not applicable. The Rust model-gateway was source-inspected rather than built or runtime-tested; the unchanged early-return counterexample is direct source evidence. No multi-node, alternate architecture, or production model-weight validation was performed.
