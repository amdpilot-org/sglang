# Independent review of amdpilot-org/sglang PR 2965

Reviewed exact commit `f924d21cd65995706d9180fe289f088ec3ef8e2b` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/32928
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2908
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2999
- Candidate: https://github.com/amdpilot-org/sglang/pull/2965
- Parent candidate: https://github.com/amdpilot-org/sglang/pull/2789 at `93773a6d3a67ed79e50d198c20f9964be234832e`
- Parent review: https://github.com/amdpilot-org/sglang/pull/2874

## Recommendation

`request_changes`. The candidate correctly fixes the reviewed 1.2-second delayed-response counterexample, and its focused unit suite passes. It does not fully establish or satisfy the original metrics contract because Prometheus series are not cleared when a condition recovers or a communicator disappears.

## Reproductions and independent checks

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` has no `sglang.srt.distributed.nccl_ras` module or candidate regression file, reproducing the absence of the requested feature.

The exact parent `93773a6d3a67ed79e50d198c20f9964be234832e` was exercised with a real loopback TCP server which read both protocol commands, delayed 1.2 seconds, and returned valid JSON. Its default `read_timeout` was 1.0 second and `poll_status()` returned `None`. The exact reviewed commit used 5.0 seconds and returned a parsed status after about 1.203 seconds. This independently verifies the candidate's targeted timeout correction.

The candidate's regression suite passed: 35 tests plus 2 subtests. The changed Python integration modules passed `compileall`, and imports resolved to `/job/repo/python`, not an installed SGLang wheel.

An independent Prometheus exposition test then published a communicator with `AllReduce` divergence 7, followed by a healthy finding with no divergence. `review_div{collective="AllReduce",comm_hash="c"}` remained `7.0`. A later snapshot with no communicators also retained the communicator's old gauge series. `SchedulerMetricsCollector.log_nccl_ras()` only writes current findings and current divergence keys; it neither zeros/removes disappeared divergence labels nor removes series for communicators absent from subsequent STATUS snapshots. Thus resolved/nonexistent faults can remain externally reported and keep alerts firing, contrary to current-state health reporting.

## Scope and limitations

No native source changed, so a native rebuild was not applicable. The prepared environment provides one AMD Instinct MI350X, Torch `2.11.0+rocm7.2`, and RCCL API compatibility `2.27.7`. It cannot verify live NVIDIA NCCL 2.28.7+ RAS polling, two-node unresponsive/considered-dead transitions, or TP=2 serving metrics. The deterministic tiny Llama fixture cannot supply the missing NVIDIA NCCL implementation, second node, or second assigned GPU, so it was not used as substitute evidence.

The candidate is therefore a verified partial correction, not a fully verified resolution of the original issue.
