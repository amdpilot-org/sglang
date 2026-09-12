# Independent review of candidate PR 2896

Candidate: https://github.com/amdpilot-org/sglang/pull/2896

Exact commit: `3f01b12dfb5ac76975f41964819f07ebf3b8696a`

Upstream issue: https://github.com/sgl-project/sglang/issues/19352

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2867

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2930

## Verdict

Recommendation: **request changes**. The candidate is useful partial hardening, but it does not fully resolve the original common-test-entry refactor contract.

The original feature was substantially implemented by merged upstream PR 19354, which introduced shared model-name constants and utilities. At the recorded base, later tests had reintroduced 20 exact copies of model IDs that already had shared constants. An independent base scanner reproduced those 20 copies. At the exact candidate commit, the candidate's two regression tests passed and those 20 copies were removed without changing their string values.

The new guard is narrower than the original maintenance contract. It builds its prohibited set only from constants already declared in `test_utils.py`. Consequently, repeated runnable model IDs that were never centralized are invisible. Concrete remaining cases include four `Lightricks/LTX-2.3` literals in `server/gpu_cases.py`, `Lightricks/LTX-2` across GPU and Ascend configurations, `FastVideo/FastHunyuan-diffusers` across GPU and MUSA configurations, and two `Rabinovich/LongLive-2.0-5B-Diffusers` literals. The guard also excludes the entire unit tree rather than narrowly exempting intentional literal test vectors.

This distinguishes the result as a partial fix plus test hardening. It is not test-only—the candidate does replace 20 duplicated values—but its `outcome: fixed` claim overstates completeness against the broad original issue.

## Environment and source paths

- Prepared base and review branch: `358c163250ad3b1f62939b01ce1314a0a31a0365`, `amdpilot/j-82b8488c8505`.
- Candidate was tested detached at the exact requested commit, then the checkout was returned to the prepared review branch before this report was added.
- Interpreter: `/tmp/amdpilot-repo-j-82b8488c8505/venv/bin/python`.
- Imported SGLang source: `/job/repo/python/sglang/__init__.py`.
- Torch: `/opt/venv/lib/python3.12/site-packages/torch`, version `2.11.0+rocm7.2`, HIP `7.2.26015`.
- Hardware: one AMD Instinct MI350X. Ascend and MUSA hardware were unavailable; the touched DP test requires two GPUs.
- No native source changed. `repository-environment.json` records no native build target or wheel, so no native rebuild was applicable.

## Evidence

Raw evidence was preserved outside the checkout during revision switching under `/job/review-evidence/`, with the base reproduction at `/job/base-duplicate-scan.log`. Detailed commands, exit codes, and measurements are in `result.json`.

The candidate regression tests, affected-test collection, compilation, pre-commit hooks, and diff whitespace check passed. No GPU inference was run because it would not validate this AST/source-refactor contract, one required runtime path needs two GPUs, and the other architecture-specific paths need unavailable Ascend or MUSA devices.
