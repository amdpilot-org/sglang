# Investigation notes

- Upstream issue: https://github.com/sgl-project/sglang/issues/31896
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2177
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Related upstream change: https://github.com/sgl-project/sglang/pull/31909 was closed
  without merging. It proposed the same CP-rank guard in an older version of the
  metrics collector. The prepared base still lacked that guard.
- Supporting precedent: https://github.com/sgl-project/sglang/pull/30038 uses an
  attention-CP-rank-zero guard for another singleton scheduler responsibility.

## Reproduction and fix

`SchedulerMetricsCollector.init_new` selected the stats/export rank with only
`ps.attn_tp_rank == 0`. When attention TP size is one, every rank in an
attention CP group satisfies that predicate. A deterministic CP=8 regression
using the actual checked-out method therefore observed eight enabled default
exporters before the fix (`8 == 1` assertion failure in `pytest_before.log`).

The correction also requires `ps.attn_cp_rank == 0`. The same test observes one
enabled exporter after the fix. Independent cases cover nonzero attention TP
ranks, a nonzero attention CP rank, the single-rank topology, disabled metrics,
the explicit all-schedulers override, and the pre-existing KV-event publisher
rule.

## Scope and limitations

The assigned device inventory contains one AMD Instinct MI355X (gfx950). The
reported eight-GPU CP serving topology could not be launched on one GPU, and no
model weights were needed for this rank-selection defect. No GPU workload or
full HTTP/model-serving reproduction was claimed. This change fixes default
metric/log exporter selection; it deliberately does not divide request values
or alter the explicit `--enable-metrics-for-all-schedulers` behavior.

Raw command output is retained in `evidence/`.
