# Investigation report: remote-instance TransferEngine PP rank collision

Upstream issue: https://github.com/sgl-project/sglang/issues/37216

Mirror issue: https://github.com/amdpilot-org/sglang/issues/994

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The prepared base still registered seed TransferEngine metadata under
`tp_rank`, stored it under that key, and queried it using the destination
worker's `LoadConfig.tp_rank`. Under TP1/PP2, both stages therefore use key 0
and the second registration overwrites the first.

The source issue has two related upstream pull requests. PR 37246 is closed;
PR 37259 remains open and proposes the same core direction used here: use the
distributed world rank across registration, bootstrap storage, and lookup.
Neither change was present in the prepared base.

## Correction

The TransferEngine metadata path now uses `get_world_rank()` end to end. This
rank is initialized by SGLang as `tp_size * pp_rank + tp_rank` for the normal
model-parallel group, so TP1/PP2 stages use distinct keys 0 and 1. The NCCL
remote-loading path remains keyed by TP rank because its ports and peer groups
are intentionally paired per TP shard.

## Evidence

- `raw/failing-before.log`: the new regression suite on the original behavior;
  four failures show that the server rejects `rank`, the transporter has only
  `tp_rank`, and the loader has no world-rank lookup.
- `raw/final-focused.log`: 42 tests passed plus 14 subtests after the change.
- `raw/pre-commit.log`: repository pre-commit checks passed.
- `raw/environment-and-rank-evidence.log`: one visible AMD Instinct MI350X,
  Mooncake installed, and source excerpts showing the world-rank formula.

## Limitation

Only one GPU was assigned, while the reported TP1/PP2 reproduction requires
two concurrent pipeline stages. The Qwen3-8B weights from the report were also
not provided. Therefore no full model, two-stage pipeline, RDMA transfer, or
semantic-output claim is made. The deterministic regression covers the actual
HTTP bootstrap server and the producer/consumer Python call sites.
