# Independent review of PR 3375

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3375 at exact commit `eb796a5798ccc3692728b84ce04e7daeae4baa38`

Upstream issue: https://github.com/sgl-project/sglang/issues/31205

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3342

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3379

## Recommendation

Accept. The candidate fully resolves the scheduler-admission contract described by the original issue in deterministic tests. It preserves the earlier candidate's normal hybrid-SWA safe-chunking and terminal rejection behavior, and closes the independently reported `ignore_eos`/disabled-prefix-cache path that otherwise returned `NO_TOKEN` forever.

## Evidence

The prepared branch was at the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, the exact 2,000-token, 512-token-pool, `max_new_tokens=1`, `ignore_eos=True`, disabled-prefix-cache fixture returned `NO_TOKEN` three consecutive times, never set an extend range, and had no rejection mechanism. The exact earlier candidate `75b30f8215eca5480055578e2bc4c02efcddcf18` reproduced the review counterexample: three `NO_TOKEN` results, no extend range, and zero rejected requests.

At exact candidate commit `eb796a5798ccc3692728b84ce04e7daeae4baa38`:

- The full focused `test_prefill_adder.py` suite passed: 36 tests and 12 subtests.
- The original 20,742-token/20,992-token-pool shape returned `OTHER` after admitting a safe 19,968-token page-aligned chunk, demonstrating progress rather than FCFS starvation.
- The exact-capacity boundary also made page-aligned progress.
- The `ignore_eos` impossible request returned `REJECT`; the same current availability with a 4,096-token total pool remained `NO_TOKEN`, correctly distinguishing permanent impossibility from transient pressure.
- A pool with positive but sub-page usable capacity rejected rather than looping.
- An intrinsically impossible head returned `REJECT`, and a short tail request on the same adder returned `CONTINUE` and entered `can_run_list`.
- Imports resolved to `/job/repo/python/sglang/...`, confirming the checked-out source was tested.
- `compileall` and `git diff --check` passed.

The candidate changes Python scheduler code only. No C/C++/HIP/native source changed, and the prepared environment declares no separate native artifact, so a native rebuild was not applicable.

## Architecture and environment limitations

The assigned device was one AMD Instinct MI350X reporting `gfx950`; the reported deployment used 8×B300 with DeepSeek-V4-Pro/DSpark, TP4 prefill + TP4 decode, 1P1D PD disaggregation, and mooncake TCP. Those weights and that multi-GPU/distributed topology were unavailable. The deterministic admission tests launched no GPU kernels, so this review does not claim model-semantic, transport, multi-node, or end-to-end HTTP reproduction. In particular, the scheduler's source path produces `HTTPStatus.BAD_REQUEST`, but the client-visible HTTP 400 was not exercised through the unavailable reported deployment.

Raw command output, source diffs, issue/PR snapshots, import paths, and GPU enumeration are retained in `raw/`.
