# Independent review of amdpilot-org/sglang PR 3275

Candidate: https://github.com/amdpilot-org/sglang/pull/3275 at exact commit `75b30f8215eca5480055578e2bc4c02efcddcf18`

Upstream issue: https://github.com/sgl-project/sglang/issues/31205

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3220

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3283

## Recommendation

Request changes. The candidate is a real partial fix, not test-only hardening: it terminates the reviewed zero-page-capacity branch for ordinary requests, retains deferral under transient pressure, removes rejected requests from the scheduler queue, and prepares an HTTP 400 response. It does not fully implement the original fail-fast contract because the `ignore_eos` admission path bypasses the new rejection logic.

## Failing-before evidence

The prepared branch was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image-checkout difference. A deterministic `PrefillAdder` reproduction used the actual checkout source and the issue/review geometry.

On the base, the reported 20,742-token prompt with a 20,992-token SWA pool already made progress through the earlier correction: it admitted a 19,968-token chunk. The remaining independently reviewed cases did fail: total SWA pools of 512 and 1,023 tokens, with page size 512 and a 2,000-token prompt, returned `NO_TOKEN` on three consecutive attempts, never set an extend range, and never admitted or rejected the request. The 1,023-token case demonstrates the positive-but-sub-page capacity boundary.

## Candidate evidence

The checkout was temporarily detached at exact candidate commit `75b30f8215eca5480055578e2bc4c02efcddcf18`. Imports resolved to:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/managers/schedule_policy.py`
- `/job/repo/python/sglang/srt/managers/scheduler.py`
- Torch `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`, version `2.11.0+rocm7.2`, HIP `7.2.26015`

The candidate's complete focused file passed: 34 tests plus 12 subtests. Independent cases showed:

- reported 20,742/20,992 geometry: admitted a 19,968-token chunk;
- permanent 512-token pool: `REJECT`;
- permanent 1,023-token pool: `REJECT`;
- current 512 free tokens with a 4,096-token total pool: `NO_TOKEN`, correctly treated as transient pressure;
- `ignore_eos=True`, disabled prefix cache, 512-token total pool: `NO_TOKEN` on three consecutive attempts, no extend, admission, or rejection.

The last case reaches `PrefillAdder.add_one_req_ignore_eos` because `add_one_req` dispatches there when `req.sampling_params.ignore_eos` and `tree_cache.disable` are true. That method retains the old SWA-only `NO_TOKEN` gate and never calls `_reject_swa_req`. Consequently, the intrinsically impossible request remains at the FCFS head and reproduces the same terminal-condition failure the issue requires the scheduler to avoid.

## Native, GPU, and architecture scope

The candidate changes only Python scheduler code and Python tests; no native source or build definition changed, so no native rebuild was applicable. The prepared interpreter imported the checkout's Python files, not an installed SGLang copy.

The assigned device enumerated as one AMD Instinct MI355X, `gfx950`, under ROCm 7.2. These deterministic admission tests launch no GPU kernels, so `gpu_execution` is false. DeepSeek-V4-Pro/DSpark weights and the reported 1P1D TP4+TP4, mooncake TCP, multi-node, 8xB300 deployment were unavailable. This review therefore validates scheduler admission/control flow, not full HTTP transport, model semantics, transfer, or distributed execution.

## Raw evidence

The principal outputs are retained in `reports/j-cc8f265fd4d1/raw/`. The revision-independent working evidence was also preserved at `/job/review-evidence-j-cc8f265fd4d1` while switching commits.
