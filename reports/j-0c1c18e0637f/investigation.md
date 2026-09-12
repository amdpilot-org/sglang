# Investigation: decode starvation under sustained prefill

Upstream issue: https://github.com/sgl-project/sglang/issues/32549

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2044

## Finding

The prepared base already contains the requested mitigation. Upstream PR
https://github.com/sgl-project/sglang/pull/35017 merged as commit
`6f69f927da9e5692bb4709821faecff6be9b5a8a` on 2026-08-19, after the issue's
2026-08-07 snapshot. It added `--prefill-decode-interval N` as an opt-in bound:
after an extend/prefill batch, the scheduler defers new prefills for `N` rounds,
allowing a non-empty running batch to decode. In DP-attention mode the interval
is armed from the already synchronized `is_extend_in_batch` flag, keeping the
cadence rank-consistent.

The default remains `0`, deliberately preserving existing prefill-first
behavior. A deterministic trace using the checked-out `Scheduler` methods
shows continuous eligible prefill winning every round at interval 0, while
interval 2 produces `prefill, decode, decode` cadence. See
`raw/cadence_trace.log`.

The upstream fix also carries issue-shaped hardware evidence in its PR: on an
8x GB300 DeepSeek-V4-Pro speculative-decoding workload, interval 0 had 521.68 ms
TPOT p99 and 8501.45 ms ITL p99, while interval 4 reduced those to 62.75 ms and
229.91 ms respectively. This is related upstream evidence, not a reproduction
performed by this job.

No scheduler source change is justified on this base. The focused regression
and argument-validation tests already present in the repository pass.

## Limitations

- The reported DeepSeek-V4-Flash/DSPARK model weights were unavailable.
- The report used 8x NVIDIA B200 with TP8; this job has one AMD MI350X
  (`gfx950`). It cannot reproduce or qualify that model, NVIDIA backend, TP8,
  DP-attention multi-rank synchronization, or production arrival profile.
- No GPU model execution was performed because the local evidence is scheduler
  control-flow behavior and its registered CPU regression. GPU inventory alone
  is retained in `raw/gpu_inventory.log` and is not counted as GPU execution.
- The tiny Llama HTTP fixture was not used because it cannot qualify the
  different architecture, speculative decoder, or distributed workload, and a
  transport/startup smoke would not add evidence for this scheduler decision.
- The mitigation is opt-in. Launches that leave the interval at its default of
  zero retain strict prefill-first behavior.

