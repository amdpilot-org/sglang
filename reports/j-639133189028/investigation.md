# Investigation: SGLang issue #33978

Upstream issue: https://github.com/sgl-project/sglang/issues/33978

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1698

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

Outcome: `environment_blocked`.

The original combined prefill/decode hang was not reproduced. Its essential
conditions are an eight-rank `--tp 8 --dp 8 --enable-dp-attention` run, the
GLM-5.2 MXFP4 architecture and weights, TileLang DSA prefill/decode, and an
as-yet unspecified long-context workload that overlaps decode. This prepared
environment has TileLang and one MI355X gfx950, but only one visible GPU and no
GLM-5.2 weights.

The deterministic tiny-Llama serving fixture was deliberately not substituted:
it cannot execute GLM DSA, validate model semantics, or create the reported
eight-rank collective. A DP1 startup or transport smoke would not test the bug.

No source change is proposed. Without a failing issue-specific reproduction or
a causal defect in the actual path, changing collective ordering or scheduling
would be speculative and could introduce a new distributed deadlock.

## Current related implementation and changes

The current source coordinates DP-attention scheduler metadata in
`python/sglang/srt/managers/scheduler_components/dp_attn.py`. Each participating
rank contributes token counts, forward mode, and prefill graph eligibility;
the gathered eligibility uses an all-rank minimum. DP1 intentionally skips the
collective because no cross-DP state exists.

`python/sglang/srt/managers/prefill_delayer.py` performs a rank-wide negotiation
before admitting prefill when the optional prefill delayer is enabled. The
reported command did not enable that option.

`python/sglang/srt/layers/dp_attention.py` contains the gatherv/reduce-scatterv
path activated by `SGLANG_DP_USE_GATHERV=1`, including the explicit invariant
that every rank must issue the same collective sequence. This is a general
DP-MoE communication path rather than a TileLang-internal collective.

Related merged PRs inspected:

- https://github.com/sgl-project/sglang/pull/35640 coordinates FullCG prefill
  admission and shape padding across DP-attention ranks. It was merged after
  the report, but it claims CUDA-graph utilization rather than a fix for the
  reported TileLang combined-server hang.
- https://github.com/sgl-project/sglang/pull/34474 handles empty DP-attention
  batches in Qwen3.5 attention layers. It is model-specific and addresses a
  reshape crash, not a silent GLM/TileLang collective stall.
- Searches for merged PRs explicitly describing a TileLang DP-attention hang or
  deadlock found no matching fix.

The live upstream issue has one response from an investigator who used the
stated 8x MI355X setup and could not reproduce the hang despite concurrent
decode and 900k-token prefill attempts. They requested the reporter's exact
workload. This supports neither a fix nor a local non-reproduction; it shows
that the missing workload is material.

## Evidence

See `raw-output.txt` for retained command output. The focused current-source
tests passed:

- Scheduler metadata and prefill-delayer unit boundaries: 9 passed and 3
  subtests passed.
- Four-process Gloo prefill-delayer negotiation: 1 passed.
- GPU probe: torch 2.11.0+rocm7.2, HIP 7.2.26015, one MI355X gfx950, TileLang
  installed. A deterministic FP32 GEMM was finite and agreed with a CPU FP64
  reference to max absolute error `7.801873202595289e-06`.

The GPU GEMM is only a device-health/numerical sanity check. The CPU Gloo test
is only scheduler-negotiation coverage. Neither is evidence about the missing
eight-rank RCCL/TileLang execution.

## What remains unverified

- Whether the reported hang occurs on current main with GLM-5.2 MXFP4.
- Whether it requires mixed per-rank prefill/decode modes, a particular prompt
  distribution, CUDA-graph mode, or TileLang kernel shape.
- Whether the post-report FullCG coordination change affects the failure.
- The exact collective and rank at which a reproducing run stalls.

A qualifying follow-up needs eight MI355X GPUs, the reported checkpoint, the
reporter's exact concurrent workload, per-rank scheduler logs, and RCCL traces.
