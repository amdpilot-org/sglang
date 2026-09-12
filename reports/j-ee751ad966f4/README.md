# Correction generation 1: OpenAI model validation

This correction preserves the valid Python serving-path work from candidate PR https://github.com/amdpilot-org/sglang/pull/2315 at exact commit `12421813cdd2059cfe5aa105559c62b726392b36` and addresses the concrete counterexamples reported by independent review PR https://github.com/amdpilot-org/sglang/pull/2368.

## Independently reproduced before changing implementation

- The candidate's original focused Python suite passed: 4 tests and 2 subtests.
- Tightening the expected OpenAI error contract to `"param": null` failed on the candidate because it returned `"param": "model"`. Raw output: `evidence/candidate-python-failing.log`.
- Direct inspection of the exact candidate confirmed `RouterManager::resolve_model_id` returned every explicit caller model before consulting `worker_registry.get_models()`. The new Rust regression covers known and unknown explicit names plus empty, singleton, and multiple-model omitted-name boundaries.

## Correction

- Unknown Python serving requests retain the candidate's 404 `model_not_found` behavior but now return `param: null`.
- IGW HTTP model resolution now validates explicit model names against the worker registry and returns the same OpenAI-shaped 404 contract for unknown names.
- Existing implicit routing behavior is unchanged: no workers is unavailable, one model is selected, and multiple models require an explicit choice.

## Verification

```text
PYTHONPATH=python /tmp/amdpilot-repo-j-ee751ad966f4/venv/bin/python -m pytest -q test/registered/unit/entrypoints/openai/test_serving_model_validation.py
4 passed, 2 subtests passed
```

The prepared image has no `cargo` or `rustfmt` executable. Consequently the Rust regression was retained but could not be compiled or run here; `evidence/candidate-router-failing.log` records the missing-tool failure. No GPU run is claimed because both corrections execute before inference.
