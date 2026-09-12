# Fatal diffusion scheduler correction

Reviewed candidate https://github.com/amdpilot-org/sglang/pull/843 at exact
commit `f0cb5a958e600a68ea16a35b98b877f781e2046a` and independent review
https://github.com/amdpilot-org/sglang/pull/936 against base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

The candidate correctly identifies the two reported fatal accelerator strings and
returns restart-required text to the triggering request. Its regression suite
passed (19 tests). The review's remaining transport counterexample was confirmed
in source: the candidate set `_running = False`, the scheduler worker returned
normally, the HTTP server continued running, and `scheduler_rpc_timeout` defaults
to `None`, leaving later receives unbounded.

The correction latches the fatal error while keeping the scheduler RPC loop alive.
All later requests receive the same actionable error without calling the worker;
`ShutdownReq` remains dispatchable for clean process teardown. It deliberately
does not call the initial error an OOM or add resolution admission logic.

## Evidence

- Exact candidate: `probe_fatal_scheduler_state.py` failed with exit 1 because
  `_running` became false.
- Corrected branch: the same probe passed with exit 0 and confirmed no later
  worker handler call.
- Focused suite: 34 passed across fatal-state, warmup degradation,
  scheduler-client timeout, and launch shutdown tests.

## Limitations

The RTX 5070 12 GB / WSL2 / CUDA 13.0 MiniMax-H3 model, weights, and LoRA were
not available. The prepared system uses ROCm 7.2, so neither the real 1344x768
failure nor the post-failure 480p workload was reproduced. There is therefore no
evidence that `device not ready` means insufficient memory, and no admission
threshold was invented. The candidate's fatal handling remains specific to the
monolithic scheduler execution path; disaggregated role loops use different
transport/error paths and were not changed without a qualified reproduction.

