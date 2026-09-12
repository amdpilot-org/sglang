# Independent review of amdpilot-org/sglang PR 1771

- Candidate: https://github.com/amdpilot-org/sglang/pull/1771
- Exact candidate commit: `9cbb2c10d016f2a29780af58a81fd626960544dd`
- Recorded base and candidate parent: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Upstream issue: https://github.com/sgl-project/sglang/issues/33901
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1821

## Recommendation

Accept. The source change fully resolves the original issue's parser-level contract.

On the recorded base, one final increment containing two complete no-argument calls emitted only index 0 and left the complete second block in `_buffer`. The same loss was independently reproduced for two argument-bearing calls and for three complete calls.

At the exact candidate commit, both the candidate regression and independent cases emitted every complete call in the same invocation with sequential indices. A complete first call followed by a partial second call remained resumable and completed as index 1 on the next increment. The three-call case drained indices 0, 1, and 2. The detector buffer was empty after inputs ending at a complete tool-call boundary.

The imported module and class source both resolved to `/job/repo/python/sglang/srt/function_call/glm47_moe_detector.py`, so validation used the checked-out candidate rather than an installed copy. The change is Python-only and touches no native source, so a native rebuild is not applicable.

## Independent evidence

- Base reproduction: `PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-772512b57fba/venv/bin/python /job/review-evidence/review_glm47_streaming.py 358c163250ad3b1f62939b01ce1314a0a31a0365` — exit 0; recorded the reported failure and independent argument/three-call failures.
- Candidate adversarial run: the same harness at `9cbb2c10d016f2a29780af58a81fd626960544dd` — exit 0; verified two no-argument calls, two calls with typed arguments, complete-plus-partial continuation, three complete calls, and text surrounding calls.
- Candidate regression: `python -m pytest -q test/registered/unit/function_call/test_function_call_parser.py::TestGlm47MoeDetector` — 15 passed.
- Related parser suite: `python -m pytest -q test/registered/unit/function_call/test_function_call_parser.py::TestGlm4MoeDetector test/registered/unit/function_call/test_function_call_parser.py::TestGlm47MoeDetector` — 29 passed.

Raw issue/PR metadata, patches, import paths, baseline output, candidate output, and test logs are preserved under `/job/review-evidence/`, outside the revision-switched checkout.

## Limitations and non-counterexamples

This was a deterministic CPU text-parser review. No model weights, HTTP server, semantic model behavior, multi-node workload, GPU kernel, compiler, or ISA behavior is implicated by the source change. The assigned device was visible as one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`, but no GPU execution was needed or claimed.

Independent surrounding-text coverage observed that ordinary text after the final tool call remains buffered until another increment. That pre-existing single-call behavior is outside issue 33901's contract about losing additional complete tool calls, and it does not provide a counterexample to this candidate's fix. No remaining counterexample to the original reported contract was found.
