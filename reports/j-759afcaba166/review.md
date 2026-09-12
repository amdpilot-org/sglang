# Independent review of amdpilot-org/sglang PR #2391

Candidate commit: `f78d5c8d7cc80c05c8b452d0972d488a4c25040b`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/31000

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2339

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2424

## Verdict

Recommendation: **accept**.

This candidate is test-only hardening, not a new source fix. The exact recorded base already contains the source correction merged upstream in PR #31311. The candidate changes no production or native source; it adds five focused regression tests and investigation evidence. The exact candidate test suite passes, and independent probes confirmed the source behavior it targets.

The original RoPE token-count contract is resolved in the candidate tree: with a non-none EP backend and attention TP greater than one, scattered dense-branch hidden and residual rows are gathered to the full `positions` row count before `self_attn[1]`. The no-EP and TP=1 paths remain unchanged. The branch output is subsequently aligned to the MoE branch row count before addition.

## Evidence

- The prepared checkout initially matched the requested base exactly. Production source in `python/sglang/srt/models/longcat_flash.py` is byte-identical between the recorded base and candidate.
- The imported module resolved to `/job/repo/python/sglang/srt/models/longcat_flash.py`, confirming tests used the checkout rather than an installed copy.
- The candidate regression suite passed: 5 tests.
- An independent harness used the real `_scmoe_align_rows` implementation with a mocked collective. For TP=4 it gathered both hidden and residual from 1 row to 4 rows before the second attention and preserved exact expected values. With the backend disabled, it performed no gather.
- A real tensor probe on the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) reproduced the reported `view(16, -1, 64)` failure from a 256-element scattered query and validated the aligned `(16, 4, 64)` result with maximum absolute error 0.
- Focused pre-commit checks for the new test file passed.

## Important qualification

The candidate's legacy failure test is a controlled regression demonstration: it patches the new alignment helper to a no-op. It is not a literal failure of the requested recorded base, because that base already contains the fix. This is correctly interpreted as failing-without-the-fix coverage, not evidence that base commit `358c163...` itself is broken.

Only one AMD gfx950 GPU was available. The reported 4x H200, TP16/EP16, PD-disaggregated DeepEP deployment and LongCat-2.0-FP8 weights were unavailable. Therefore live multi-rank collective execution, DeepEP/UCCL transport, full serving behavior, and semantic accuracy were not reproduced. The deterministic fixture validates the original shape/alignment contract only.

No C++, HIP, CUDA, or FlyDSL source changed. A native rebuild was therefore not applicable.
