# Independent review of PR 2465

Upstream issue: https://github.com/sgl-project/sglang/issues/29738

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2432

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2503

Candidate: https://github.com/amdpilot-org/sglang/pull/2465 at exact commit `f2422973285f9887a12aaa110cb0ce0fefff1d6e`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original Python import/control-flow defect. It is a source fix with regression hardening, not a test-only change.

On the recorded base, a non-empty call through the real `entrypoint.py` with `ENABLE_JIT_DEEPGEMM=False` raised the reported `NameError`, even though an installed `deep_gemm` substitute was available to import. At the exact candidate commit, its regression and an independently written adversarial harness passed. The harness verified argument forwarding, the zero-token fast path, and the missing-dependency boundary. The latter now raises `ImportError`, rather than the false `NameError` caused by a skipped global import.

The local import is consistent with the existing large-batch callsite in `mhc.py`. DeepSeek-V4's platform hook disables this kernel path on HIP, but not on the reported H100/SM90 class of CUDA device. No native source changed, so no native rebuild was required.

## Evidence

- Failing-before: `raw/base-independent-contract.log`
- Candidate regression: `raw/candidate-regression.log`
- Independent passing-after cases: `raw/candidate-independent-contract.log`
- Pytest collection/execution: `raw/candidate-pytest.log`
- Environment and source paths: `raw/environment.log`

An initial `python -m unittest <path>` command was not a valid invocation for this repository layout and failed to import `test.registered`; the candidate test was then run directly and through pytest, both successfully. A pre-commit attempt only initialized hook environments before the invocation ended and is not claimed as a pass or failure. `py_compile` passed for both changed Python files.

## Architecture and environment limitations

The assigned device was one AMD Instinct MI355X (`gfx950`) under ROCm 7.2, while the report requires CUDA DeepGEMM on H100/SM90. The prepared interpreter had no installed `deep_gemm`, and DeepSeek-V4 weights were unavailable. Therefore this review does not claim a real CUDA kernel launch, numerical qualification, a full DeepSeek-V4 forward, or serving reproduction. These limitations do not leave a source-level counterexample to the import fix: the exact Python failure was reproduced at the real source boundary and the candidate was verified with an importable dependency substitute.
