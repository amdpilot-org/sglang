# Independent review of amdpilot-org/sglang PR 3075

Candidate reviewed: `b6797a2b4acad47e97629129cd30378115c55395`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate is a useful partial fix and closes every counterexample named by PR 2986, but it does not fully resolve the original request to centralize common diffusion test entries.

## Evidence

On the prepared base, an independent AST inventory reproduced 20 executable literals equal to existing shared constants. It separately reproduced the reviewed uncentralized values with exact counts: LTX-2.3 four times, LTX-2 twice, FastHunyuan twice, and LongLive2 twice.

At the exact candidate commit, the candidate regression passed (3 tests), the entire changed unit-file set passed (323 tests plus 50 subtests), and the diffusion test tree compiled. The prepared interpreter resolved `sglang` to `/job/repo/python/sglang/__init__.py`, confirming that tests exercised the checked-out candidate source.

Independent adversarial testing found that `_find_repeated_uncentralized_model_names` is called only with non-unit paths. `Comfy-Org/Ideogram-4` is a concrete demonstration: the helper reports its occurrences in `server/gpu_cases.py` and `unit/test_ideogram4.py` when both are supplied, but the real regression discards the unit path and passes. LTX-2.5 and FastH3 provide additional real repetitions hidden by the unit exclusion and by the detector's narrow syntax/resolver list.

These are ordinary registry/model fixtures, not the explicitly exempt parser/config vectors in `unit/test_server_args.py`. The candidate therefore hardens the test policy but leaves counterexamples to the original common-entry contract.

## Environment limits

No native source changed, so no native rebuild applied. No numerical implementation changed, and model inference would be unrelated evidence for this refactor. The host uses Torch 2.11.0 with ROCm 7.2; Ascend and MUSA execution is unavailable, and the modified DP serving case requires two GPUs while one was assigned. Source compilation, exact import paths, focused regressions, affected unit behavior, and adversarial guard behavior were validated.

Raw logs were retained outside the checkout at `/job/review_evidence/j-7f7c15d21cec/` before returning to `amdpilot/j-7f7c15d21cec`.
