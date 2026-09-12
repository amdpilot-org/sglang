# Anthropic tool-reference correction generation 2

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Candidate: https://github.com/amdpilot-org/sglang/pull/1557 at `5fd904c6cd24fe460d6446117b7ea7f541f81db8`

Independent review: https://github.com/amdpilot-org/sglang/pull/1642

The exact candidate was checked out before edits. The review's adversarial fixture reproduced both claims:

- `false_positive_detected=True`, followed by real Jinja rendering raising `ValueError: Unexpected item type in content.`
- `bracketed_native_detected=False`; the reference was degraded to text and `defer_loading` metadata was absent.

The consolidated correction preserves the candidate's generic/native conversion and deferred-tool routing. It makes Jinja attribute and bracket access equivalent for `function.defer_loading`, and requires the `tool_reference` type comparison to target a message-content access path rather than an unrelated object such as `telemetry.type`.

After the correction, the same fixture reports `false_positive_detected=False` and `bracketed_native_detected=True`. The generic payload contains `[tool reference: DemoTool]`; the native payload retains the complete tool catalog and `defer_loading` metadata.

## Validation

```bash
PYTHONPATH=python /tmp/amdpilot-repo-j-d8d60d045e1e/venv/bin/python -m pytest -q \
  test/registered/unit/entrypoints/anthropic/test_tool_reference.py \
  test/registered/unit/entrypoints/anthropic/test_serving.py
```

Result: `80 passed, 5 subtests passed`.

`git diff --check` and `py_compile` of the changed source and test also passed.

This defect and its deterministic regression are CPU-only request conversion and Jinja rendering. No GPU execution, model weights, HTTP server, semantic generation, or distributed workload was needed or claimed. No native source changed, so no native rebuild was applicable.
