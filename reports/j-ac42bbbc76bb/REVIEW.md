# Independent review of PR 2942

Candidate: https://github.com/amdpilot-org/sglang/pull/2942

Exact commit: `a6ff9d544d5e1cc56bac200b18cbe425dbed89ad`

Upstream issue: https://github.com/sgl-project/sglang/issues/35808

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2974

## Recommendation

Request changes. The candidate is a meaningful partial fix, but it does not fully implement full-lifecycle L1/L2/L3 observability.

The candidate adds bounded Prometheus result labels for the ordinary write-through acknowledgement paths, L3 request/token classification for both cache implementations, and additional request-correlated log records. Its focused CPU suite and direct one-GPU file-storage regression pass.

However, an independent adversarial GPU run exercised UnifiedRadixCache's registered decode-retraction host-pool path. A real host-capacity-declined L1-to-L2 backup and a successful target+draft L1-to-L2-to-L1 round trip both produced zero calls to the candidate's transfer-result metric. `retraction_backup` and `retraction_restore` directly submit and synchronize cross-tier transfers, but the candidate instruments only the ordinary cache write/load acknowledgement paths. This leaves a concrete supported production path outside the promised transfer success/failure observability.

Request-level tracing also remains partial. The retraction path emits no `[HICACHE] rid` lifecycle records despite having `req.rid`. The newly added ordinary lookup record always says `tier=l1 result=complete`; it does not classify L1 hit/miss or report L1 hit tokens, and L2-to-L1 logs only scheduling, not a request-correlated terminal success/failure. Thus a request still cannot reliably be followed through truthful terminal outcomes across all three tiers.

## Evidence

- Recorded-base probe at `358c163250ad3b1f62939b01ce1314a0a31a0365`: failed as expected because no cross-tier transfer API, structured cache events, or legacy prefetch snapshot existed. See `evidence/base-contract-probe.log`.
- Candidate collector regressions: 17 passed. See `evidence/candidate-observability.log`.
- Candidate direct file-backed L3/host/GPU round trip: 13 passed, 6 repository-gated skips, 2553 deselected. See `evidence/candidate-unified-gpu.log`.
- Candidate legacy HiRadixCache checks: 4 passed. See `evidence/candidate-hiradix.log`.
- Registered decode-retraction GPU tests: 2 passed, confirming the omitted path performs valid target and draft KV transfers. See `evidence/candidate-retraction-gpu.log`.
- Independent instrumentation of those same real operations: the capacity failure and successful round trip each recorded `transfer_metric_calls=[]`, then the contract assertion failed. See `evidence/candidate-adversarial-retraction.log`.
- `git diff --check` passed. No native source changed, so a native rebuild was not applicable.

## Environment and limitations

Testing used the prepared interpreter and source imports from `/job/repo/python`, with Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015, and one AMD Instinct MI355X. The GPU runs establish cache transport and exact K/V restoration for the exercised fixtures; they do not establish model semantic accuracy.

Mooncake, HF3FS, NIXL, UMBP, multi-node, TP/PP distributed, model-specific storage paths, and NVIDIA/CUDA were unavailable and remain unverified. Those unavailable paths are limitations, not evidence against the candidate. The request-changes recommendation rests on the reproduced supported single-GPU retraction counterexample.
