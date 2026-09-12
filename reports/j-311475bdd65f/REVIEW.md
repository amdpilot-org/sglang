# Independent review of PR 2809

Candidate: https://github.com/amdpilot-org/sglang/pull/2809  
Exact commit: `16c841bb1b6f1e5bb44ec52e3b97d701e85c5149`  
Upstream issue: https://github.com/sgl-project/sglang/issues/33625  
Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2841

Recommendation: **request changes**. The candidate is a substantial partial fix, but it does not fully resolve the original contract.

The original issue defines the acceptable-load predicate solely as:

```text
candidate_load <= average_healthy_worker_load * max_load_skew
```

The candidate adds a second public parameter, `min_load_gap` (default `2`), and only spills when both the relative bound and this absolute gap are exceeded. Thus, at loads `[2,0,0,0]` with `max_load_skew=1.5`, the preferred worker is over the specified bound (`2 > 0.5 * 1.5`) but remains selected. A temporary independent Rust regression asserting the issue contract failed with `Some(0)`; it was removed before leaving the candidate checkout.

The rest of the reviewed implementation is meaningful rather than test-only hardening. The candidate policy tests, load-guard lifetime tests, and real two-worker HTTP routing test passed. The PyO3 extension was rebuilt from the exact candidate and its parser/binding tests passed with imports verified against the checkout. These results validate the candidate's implemented two-threshold behavior, not the source issue's one-threshold contract.

No GPU execution was performed. This is a CPU/Rust HTTP routing change, and the environment exposed no assigned GPU agent. No model-backed benchmark was run, so performance, cache/prefill, semantic, architecture-specific, and distributed claims remain outside the evidence.
