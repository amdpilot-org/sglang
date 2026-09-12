# Independent review of amdpilot-org/sglang PR 1034

Candidate reviewed: `0361a214d3a72f831ea23570c09c6453bb01df73`

Recorded base and prepared checkout before review: `358c163250ad3b1f62939b01ce1314a0a31a0365` (no difference).

Recommendation: **accept**. The candidate fully resolves the original issue's parser contract. It is a source fix with regression coverage, not test-only hardening: `BaseFormatDetector.parse_base_json` now skips each non-dictionary entry while continuing to parse valid neighboring calls.

## Evidence

- On the exact base, qwen25 mixed arrays raised `AttributeError` for strings, integers, nulls, and arrays. Invalid entries before, after, and between valid calls discarded those calls. Trinity reproduced the same shared-parser failure.
- The exact candidate was checked out detached. Imports resolved to `/job/repo/python/sglang/...`, so tests exercised checkout source rather than an installed copy.
- Candidate focused regression: 7 passed.
- Candidate broader parser suite: 250 passed and 4 subtests passed.
- Independent adversarial checks preserved valid calls with malformed entries before, after, and between them; covered integer, string, null, list, boolean, and float entries; covered top-level scalar input; and preserved existing unknown-tool forwarding behavior in both modes.
- Independent parser checks passed for qwen25, trinity, llama3, and the current mistral JSON-array path.
- Direct invocation of the actual `/parse_function_call` route function returned status 200 with the valid call for the mixed array and status 200 with an empty call list for `[1, 2]`, rather than propagating `AttributeError`.
- `git diff --check` and Python `compileall` passed.

Raw command output was preserved outside the revision-switching checkout at:

- `/job/baseline-parser-evidence.txt`
- `/job/candidate-focused-pytest.txt`
- `/job/candidate-parser-suite.txt`
- `/job/candidate-adversarial-evidence.txt`
- `/job/candidate-http-route-evidence.txt`
- `/job/candidate-0361a214.diff`

## Scope and limitations

The patch changes Python only; it changes no C/C++/HIP/CUDA/native source, so no native rebuild was applicable. The prepared environment has one AMD Instinct MI355X (`gfx950`) with PyTorch `2.11.0+rocm7.2`, but this CPU-side JSON parsing defect required no GPU execution. No model weights, semantic generation, distributed workload, or live socket-bound server were exercised. The HTTP result was validated by invoking the registered route implementation with the prepared request types and tokenizer state stub, so transport/middleware behavior remains outside the evidence. None of those limitations leaves a counterexample to the original parser contract.
