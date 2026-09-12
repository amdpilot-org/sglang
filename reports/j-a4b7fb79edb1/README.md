# Independent review of PR 2728

Reviewed exact candidate commit `8ae549b1d88e2d33a945914af80a20efea8e63e1`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and
the complete contract in https://github.com/sgl-project/sglang/issues/34899.

## Finding

Recommendation: **request changes**. This is plausible test-only hardening, but
it is not a verified completion of the original issue.

The base collected 18 cases in the existing Inkling bit-exact module and had no
unified-memory exact-KL class. The candidate collected 24 cases, adding baseline,
prefill-cache-hit, and decode-cache-hit checks for both Python and Rust tree
cores. Independent structural checks confirmed that the new class requests
unified memory, deterministic inference, Triton attention, disabled prefill
graphs, `extra_buffer`, 32 samples, 1024 new tokens, and the existing `1e-9`
exactness guard. A deliberately false `enable_unified_memory` server response was
rejected.

The actual Inkling serving path could not execute on the assigned AMD Instinct
MI355X (`gfx950`, ROCm 7.2). Both the base approximate test and the candidate
exact test failed during model registration because
`sglang.srt.models.inkling` imports the NVIDIA FA4/CUTLASS implementation and
the pinned environment has no `cutlass` module. Therefore no candidate KL value
was measured, no FULL+SWA+MAMBA cache restoration ran, and neither tree backend
was qualified end to end.

The source issue also requires revert-then-red validation proving each guard
detects the defect it claims to cover. The candidate contains no unified-memory
defect revert, fault injection, failing numerical trace, or equivalent evidence.
Its lower-level MLA parity test passed bit-for-bit on the MI355X, but that
existing test does not exercise Inkling, SWA, MAMBA recurrent checkpoints, the
radix cache-hit helpers, or either new test class.

## Environment and source paths

- Python: `/tmp/amdpilot-repo-j-a4b7fb79edb1/venv/bin/python`
- Imported SGLang: `/job/repo/python/sglang/__init__.py`
- KL helper: `/job/repo/python/sglang/test/kl_test_utils.py`
- Unified pool: `/job/repo/python/sglang/srt/mem_cache/unified_memory_pool.py`
- GPU: AMD Instinct MI355X, `gfx950`, ROCm 7.2
- Native rebuild: not applicable. The candidate changes only Python tests and
  report artifacts; `repository-environment.json` declares no native target.

## Evidence

- `base-collect.log`: 18 tests at the recorded base.
- `base-inkling-serving.log`: base serving attempt and CUTLASS import blocker.
- `candidate-collect.log`: 24 tests at the exact candidate.
- `candidate-structural-adversarial.log`: independent argument/helper checks and
  rejection of a false runtime configuration.
- `candidate-exact-serving.log`: candidate serving attempt and the same blocker.
- `candidate-gpu-parity.log`: existing independent stock-pool comparison, 3
  tests plus 4 subtests passed bit-for-bit on the assigned GPU.
- `candidate-import-paths.log`: measured source imports and missing `cutlass`.

The prepared checkout initially matched the recorded base and was clean. Review
evidence was stored outside the checkout before revision switches, and the
checkout was returned to `amdpilot/j-a4b7fb79edb1` before this report was added.
