# Independent review of amdpilot-org/sglang PR 1663

- Candidate: https://github.com/amdpilot-org/sglang/pull/1663
- Exact candidate commit: `e7796b0b99fe3b260c8d2bd140e9cd07e35d9139`
- Upstream issue: https://github.com/sgl-project/sglang/issues/34434
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1699
- Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**
- Fully resolves the original issue: **yes**

## Findings

The recorded base independently reproduces the source-level defect. An AST scan for discarded `*.create_task(...)` expression results finds all five issue-named sites: `mesh_api.py:222`, `multi_tokenizer_mixin.py:725`, and `tokenizer_manager.py:1817`, `:2528`, and `:3083`. The first base assertion therefore fails as expected. The issue's narrower `rg` expression does not find the `loop.create_task(...)` call in `multi_tokenizer_mixin.py`, but that call has the same missing strong-reference ownership.

At the exact candidate commit, all five sites retain their tasks. `TokenizerManager._create_background_task` uses the manager's existing `asyncio_tasks` set and removes completed tasks with a done callback. The mesh endpoint adds equivalent module-level ownership for mesh jobs. No reported site remains a discarded create-task expression.

The candidate regression passed together with the existing multi-tokenizer tests. Independent adversarial checks also passed: two concurrent manager tasks remained independently retained; completing one did not release the other; cancelled manager and mesh tasks were released; and a failed task was released after its exception was observable. These checks validate ownership and cleanup rather than relying only on the candidate's prose or static smoke.

The helper intentionally does not consume exceptions. That matches the original issue's reference-retention contract and preserves normal asyncio reporting semantics; it is not a remaining counterexample to this issue.

## Source and build validation

Tests used `/tmp/amdpilot-repo-j-c663ee45b7bd/venv/bin/python` with `PYTHONPATH=/job/repo/python`. Runtime inspection resolved `sglang` and all three changed modules under `/job/repo/python`, so the candidate source—not an installed copy—was exercised.

The candidate changes only Python files and a Python test. No C++, FlyDSL, extension, or other native source changed, so a native rebuild was not applicable. `compileall` and `git diff --check` both succeeded.

## Limitations

No GPU, model weights, HTTP server, full model, or multi-node workload was used. Those are unnecessary for this environment-independent asyncio ownership issue, but this review makes no serving, model-semantic, performance, or distributed-runtime claim. The host emitted an unrelated NUMA-balancing warning while importing `aiter`; imports and all relevant tests still completed successfully.

Raw outputs are retained in `reports/j-c663ee45b7bd/evidence/`.
