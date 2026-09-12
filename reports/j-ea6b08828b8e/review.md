# Independent review of PR 829

Reviewed `https://github.com/amdpilot-org/sglang/pull/829` at exact commit
`8e704efbc07b4cb2fe1a3772f8d525f75e9fdb17` against upstream issue
`https://github.com/sgl-project/sglang/issues/35241` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/874`.

## Finding

Recommendation: **accept as a partial fix**, not as resolution of the original
performance issue.

The recorded base was exactly the prepared checkout commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Running the candidate's tests
unchanged against that base produced five relevant failures: health probes
advanced the user round-robin counter, changed total-request and total-token
budgets, refreshed the load snapshot for a health-only batch, and did not
preserve bootstrap-room routing in the test setup. The exact candidate passed
all 28 focused tests.

Independent mixed-batch and stale-worker interleavings also passed. They
confirmed that probe-only round robin remains separate from user round robin,
mixed health/user batches refresh the user load snapshot once, unavailable
workers are skipped, and a failing health-only dispatch does not refresh user
load state.

The scheduler change restores the important model-path contract that the prior
candidate violated: an idle probe reaches ordinary generation admission rather
than receiving a synthetic immediate success. A busy scheduler continues to
defer the health reply until a model result is available.

## Scope and limitations

This is controller routing-state isolation plus regression hardening. It is not
a demonstrated full fix for the original issue. An idle probe is deliberately
admitted as a real generation request and can therefore participate in actual
scheduler and PrefillDelayer state. The scheduler regression uses a mocked
dispatcher to establish admission versus bypass; it does not demonstrate
distributed PrefillDelayer synchronization or throughput stability.

The assigned host has one AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and
Torch 2.11.0, not eight NVIDIA B30Z GPUs, and DeepSeek-V4-Pro weights are not
available. Consequently no TP8/DP8 reproduction or qualification of the
62--79k tok/s LOW regime, 16,384-token rank divergence, persistent
`mixed/delay`, or running-request collapse was possible. The tiny Llama fixture
could validate transport and single-GPU execution only, so it was not used as
evidence for those distributed performance claims.

No native/C++/FlyDSL source changed. Imports were confirmed from the candidate
checkout under `/job/repo/python/sglang/...`; no native rebuild was applicable.
