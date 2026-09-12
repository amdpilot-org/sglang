# Independent review of PR 1002

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1002 at exact commit `0304e79b3e03e26c910ad4ac9091fb52504c20e9`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38448

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1044

## Finding

Recommendation: **accept**. The candidate fully resolves the original source-level ordering contract. The prepared base hands the HiCache burst over before batch preparation and leaves mixed prefill IDs for a later forward-entry H2D. The candidate moves hand-over into `run_batch`, pre-uploads the final batch's round-head tensors first, and specifically stages the mixed batch's prefill slice while retaining the required late decode-token gather.

The candidate is more than test-only hardening: its scheduler and forward-batch changes remove the demonstrated enqueue-order counterexample. No remaining functional counterexample was found.

## Evidence

- Prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`: deterministic source-order reproduction reported `handover_before_prepare=True`, `mixed_prefill_deferred_to_resolve=True`, and `contract_failure=True`.
- Exact candidate: focused candidate regression plus neighboring auxiliary-output tests passed, 53/53.
- Independent adversarial gfx950 run: `mixed_prefill_h2d` preceded `burst_record`; forward entry gathered decode token 13 and produced `[3, 5, 8, 13]`; the staged tensor was consumed once.
- Replacement-source boundary: after staging `[1, 2]`, replacing the CPU source forced `replacement_h2d` and produced `[9, 10, 13]`, rather than consuming stale staging.
- Imports resolved to `/job/repo/python/sglang/...` at the detached candidate revision. Torch was `/opt/venv/.../torch`, version `2.11.0+rocm7.2`, HIP `7.2.26015`.
- Candidate changes no native source, so native rebuilding was not applicable.

Raw logs and the exact candidate diff were preserved outside the checkout at `/job/review-evidence-j-d3991689f8b4/` while revisions were switched.

## Limitations

The original NVIDIA H100 TP=8 deployment was unavailable. This review used one AMD Instinct MI355X gfx950 and did not reproduce a multi-GiB HiCache burst, collect an nsys trace, exercise TP all-reduce delay, or measure end-to-end speedup. Those architecture/performance observations remain unverified, while the source-level stream-order contract and real-device tensor behavior are verified.
