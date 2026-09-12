# Investigation of sglang#34582

## Outcome

`environment_blocked`: the reported two-node, 16-rank H100 GLM-5.2 startup
deadlock cannot be reproduced or rejected with the assigned single gfx950 GPU
and without the GLM-5.2 W4AFP8 weights. No runtime behavior was changed.

The most important source-specific finding is that the issue's standalone NCCL
test does not exercise the default collective selected by the reported command.
In current `main`, `prepare_mlp_sync_batch_raw` selects `tp_group.cpu_group` and
`device="cpu"` when overlap scheduling is enabled, there are no offload tags,
and `SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH` is unset. The
reported command does not disable overlap scheduling or set that variable, so
the implicated call is expected to use the CPU process group (Gloo on the
reported CUDA deployment), not `tp_group.device_group` (NCCL). The exact source
branch is retained in `raw/source_transport_selection.txt`.

Upstream PR https://github.com/sgl-project/sglang/pull/34665 already reached the
same conclusion and proposes diagnostics rather than a synchronization change.
It remains open. Upstream PR https://github.com/sgl-project/sglang/pull/34338,
merged immediately before the report, changes post-collective device-to-host
consumption but does not change collective participation or the default group.
Their captured state and descriptions are retained in
`raw/related_upstream_state.jsonl`.

## Local evidence

`reproduce_gloo_mlp_sync.py` calls the checked-out
`MLPSyncBatchInfo.all_gather` implementation with heterogeneous metadata: the
first DP replica contributes a 64-token extend batch and the second contributes
an idle batch. Both a minimal 2-rank boundary and a topology-shaped 16-rank
localhost Gloo run completed, and every rank observed `[64, 0]`. This verifies
the local tensor sizing and metadata decoding for those cases. It does not
verify cross-node Gloo routing, scheduler collective ordering, CUDA/NCCL, the
GLM architecture, speculative decoding, or model semantics.

The existing focused scheduler metadata unit suite also passed all four tests.
The assigned AMD Instinct MI355X (gfx950) was visible to ROCm Torch and a basic
tensor sum completed; this was only an environment check. The source issue's
H100/CUDA two-node serving path was not executed.

## Reproduction commands

```bash
/tmp/amdpilot-repo-j-d1dfc90c8371/venv/bin/python \
  test/registered/unit/managers/scheduler_components/test_dp_attn.py

/tmp/amdpilot-repo-j-d1dfc90c8371/venv/bin/python \
  reports/j-d1dfc90c8371/reproduce_gloo_mlp_sync.py --world-size 2

/tmp/amdpilot-repo-j-d1dfc90c8371/venv/bin/python \
  reports/j-d1dfc90c8371/reproduce_gloo_mlp_sync.py --world-size 16
```

## What remains needed

Run the original command (or a faithful GLM-5.2 fixture) on two H100 nodes with
per-rank logging immediately before the first MLP-sync collective. Capture the
backend, group identity and size, global/group rank, sequence number, token
count, and forward mode on all 16 ranks. That evidence can distinguish a Gloo
transport failure from missing/out-of-order rank participation. Switching the
default group or changing scheduler synchronization without this evidence is
not justified by the available reproduction.
