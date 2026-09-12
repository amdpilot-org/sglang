# Consolidated correction

Candidate: https://github.com/amdpilot-org/sglang/pull/531 at `75e10e4bdbfdfa2f05a5b5258eda2bb48cb3db02`

Independent review: https://github.com/amdpilot-org/sglang/pull/672

The candidate correctly isolates generation health probes from the controller's user round-robin position, speculative request/token budgets, and health-only load-budget refresh. Those changes are retained.

The remaining counterexample was reproduced by presenting a health request to a scheduler reporting fully idle. At the candidate behavior, `_request_dispatcher` was called for the probe, and the regression's ordinary-admission recorder observed changes to waiting-queue length, attempted prefill batch size, PrefillDelayer outcome accounting, and local prefill progress. In an interleaving, the subsequent user was the second admission rather than the first. See `raw/pytest-candidate-failing.txt`.

The consolidated correction intercepts every generation health probe before `_request_dispatcher`. Busy probes retain the existing behavior of replying after the next batch result. Fully idle schedulers emit the same `HealthCheckOutput` immediately, avoiding a model request that would make only the selected DP rank locally prefillable. The focused controller and scheduler suite passes; see `raw/pytest-focused-passing.txt`.

No TP8/DP8 long-prefill run was possible because the prepared environment does not provide the reported eight NVIDIA B30Z topology or model weights. Accordingly, no throughput or full-system elimination claim is made.
