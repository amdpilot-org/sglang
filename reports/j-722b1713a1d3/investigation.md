# Hybrid SWA ignore-eos starvation correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31205

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3342

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3275 at exact commit `75b30f8215eca5480055578e2bc4c02efcddcf18`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/3311

## Reproduction

The candidate's two commits were applied unchanged to the prepared base before testing. A regression using the review's exact geometry (`size_swa=rem_swa=512`, `page_size=512`, 2,000-token prompt, `max_new_tokens=1`, `ignore_eos=True`, and `tree_cache.disable=True`) called `add_one_req` three times. At the candidate state all three attempts returned `NO_TOKEN`; no extend range or terminal rejection was recorded. The failing pytest output is retained in `raw/ignore-eos-before.txt`.

The source confirms why: `add_one_req` immediately dispatches this configuration to `add_one_req_ignore_eos`, before the candidate's ordinary-request safe-chunk/reject logic. That specialized path cannot chunk a prefill and compared only against currently remaining SWA capacity, so no future pool drain could make an intrinsically oversized request admissible.

## Correction

The candidate's valid changes are preserved. In `add_one_req_ignore_eos`, an SWA capacity miss now uses the same total-pool predicate and rejection mechanism as ordinary admission. It rejects only when the complete request cannot fit after all transient SWA pressure drains. A companion boundary test verifies that `rem_swa=512` with `size_swa=4096` still returns `NO_TOKEN`, because that request may become admissible later.

The scheduler handling inherited from the candidate removes `REJECT` requests from the queue, releases cache-side state, and returns the existing HTTP 400 error response.

## Evidence

- Candidate regression: 1 failed; three attempts ended in `NO_TOKEN` rather than `REJECT`.
- Corrected focused cases: 2 passed (permanent oversize rejects, transient pressure defers).
- Complete `test_prefill_adder.py`: 36 passed, 12 subtests passed.
- Python byte-compilation and `git diff --check` both exited 0.

## Limitations

DeepSeek-V4-Pro/DSpark weights and the reported 1P1D TP4+TP4 mooncake TCP multi-node 8xB300 deployment were unavailable. The assigned device enumerated as one AMD Instinct MI355X (`gfx950`), but deterministic scheduler-admission tests do not launch GPU kernels. This work therefore does not claim a full HTTP serving, model-semantic, transfer, or distributed reproduction. No native source changed, so no native rebuild was applicable.
