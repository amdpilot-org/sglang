# Independent review of amdpilot-org/sglang#531

Reviewed exact commit `75e10e4bdbfdfa2f05a5b5258eda2bb48cb3db02` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: request changes. This is a useful partial controller-state fix, but it is not evidence that the original PrefillDelayer/throughput issue is fully resolved.

## Verified

The candidate's tests were transplanted onto the prepared base first. Five failed, showing that health probes advanced the shared round-robin counter, changed request/token budgets, and refreshed load state. At the exact candidate commit all 25 controller tests passed. The imported implementation came from the checkout's `python/sglang/srt/managers/data_parallel_controller.py`.

The change is Python-only. No native/FlyDSL rebuild applies.

## Remaining original-contract gap

The new controller route sends an idle health probe to one real DP worker. Scheduler `process_input_requests` skips health checks only while busy; an idle probe otherwise enters the normal request dispatcher and waiting queue. `get_new_batch_prefill` then constructs and finalizes the ordinary `PrefillDelayerSinglePassExecutor`, and schedule-policy admission reports local prefillability through it.

Thus controller accounting is isolated, but scheduler/delayer state is not shown to be isolated. A one-rank probe can still be the sole locally prefillable request and contribute a `mixed` cross-rank observation. The candidate tests terminate at mocked `sock_send`, so they cannot detect this counterexample or measure the scheduler counters named in the issue.

## Environment

The assigned host exposed one AMD Instinct MI355X, ROCm gfx950, with PyTorch 2.11.0+rocm7.2. A seeded real-GPU matmul matched an independent NumPy float64 reference with maximum absolute error `1.0059784056437593e-05`. The original setup requires eight NVIDIA B30Z GPUs, TP=8, DP=8, DeepSeek-V4-Pro, and long chunked prefills. Consequently the throughput bifurcation and per-rank progress/delayer counters remain unverified here; the GPU smoke is not used as proof of the scheduler fix.

Complete raw command output was preserved outside revision switching under `/job/review-evidence/`.
