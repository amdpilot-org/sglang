# Investigation report: Mamba state-cache concurrency cap

Upstream issue: https://github.com/sgl-project/sglang/issues/36889

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3330

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The capacity mechanism reproduces in the actual implementation: with eight
requested requests, ten state slots, and a five-slot-per-request ratio,
`KVCacheConfigurator.resolve_max_num_reqs` returns an effective concurrency of
two.

The reported diagnostic defect does not reproduce. The recorded base emits an
actionable `WARNING` containing the effective cap, pool size, slot ratio, and
configuration suggestions. Direct inspection of the issue reporter's cited
revision, `aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e`, shows the same
`logger.warning` call. The scheduler also publishes its resolved value as
`effective_max_running_requests_per_dp` in each `/server_info` internal-state
entry.

Because the source already contains one of the issue's requested remedies, no
production change was justified. This PR adds regression coverage that would
fail if the diagnostic were reduced to INFO, along with exact-capacity and
below-one-request boundary cases.

The separate top-level API convenience field is being pursued in upstream PR
https://github.com/sgl-project/sglang/pull/38931 and is intentionally not
duplicated here.

## Evidence

- `raw/recorded_base_cap_path.txt`: cap and WARNING path at the recorded base.
- `raw/reporter_base_cap_path.txt`: the same path at the reporter's cited base.
- `raw/pytest_mamba.txt`: 12 passing focused and adjacent Mamba tests.
- `raw/gpu_inventory.txt`: assigned device inventory (MI355X/gfx950); no GPU
  execution was relevant to the CPU-side resolver regression.

## Limitations

The GLM-5.3-Flash NVFP4 target weights, DFlash2 drafter weights, two DGX Spark
GB10 nodes, and CUDA 13 topology from the report were unavailable. Therefore
this investigation does not claim to reproduce the published throughput curve,
qualify that model architecture, or validate distributed serving semantics.
