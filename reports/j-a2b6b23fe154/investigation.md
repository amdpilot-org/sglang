# Investigation: NCCL symmetric-window registration deadlock

Upstream issue: https://github.com/sgl-project/sglang/issues/36943

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1039

## Finding

The reported failure could not be reproduced in the assigned environment. The
report is specific to CUDA/NCCL symmetric-window registration on a multi-rank
NVIDIA system. The prepared environment exposes one AMD Instinct MI350X
(`gfx950`) through ROCm 7.2, and PyTorch reports no CUDA runtime. Consequently it
cannot execute the embedded allocator's `ncclMemAlloc` /
`ncclCommWindowRegister` path or reproduce a TP4 NCCL UDS rendezvous.

The risk described by the issue is still present in the inspected source at
base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`:

- `SymmetricMemoryContext.__exit__` invokes registration synchronously from the
  forward-path context exit.
- `nccl_allocator_register_segments_with_comm` calls
  `ncclCommWindowRegister` directly, with no timeout or progress reporting.
- The call is made while `g_segment_mutex` is held.
- `nccl_free_plug` clears every tracked segment and all per-communicator indices
  when any segment is freed.

The current upstream `main` file fetched during the investigation retained the
same implementation. Searches of current related pull requests found no fix
for the reported UDS deadlock. PR 37768 is related to symmetric memory and NCCL
versions, but guards a separate DCP/CUDA-graph correctness race rather than the
blocking registration rendezvous.

## Why no code fix was attempted

A Python thread or signal wrapper cannot safely recover a native thread stuck
inside NCCL, and killing a scheduler after a timeout would require lifecycle and
multi-rank validation. Pre-sizing or coordinating allocation rounds changes
allocator/collective behavior and likewise requires the unavailable CUDA/NCCL
TP fixture. Adding only log messages would improve attribution but would not
correct the permanent deadlock. Without the affected architecture, NCCL build,
and at least two ranks, none of these candidates can be honestly validated as a
fix.

## Retained evidence

- `environment.txt`: prepared interpreter and actual GPU/runtime inventory.
- `source-path.txt`: issue-specific call sites in the checked-out source.
- `related-prs.json`: current upstream PR search results.
- `upstream-main-pynccl_allocator.py`: current upstream implementation inspected
  for an already-landed correction.

## Remaining verification

Reproduction and candidate validation require a CUDA host with NCCL 2.29.7,
multiple NVIDIA ranks (the report used TP4 on B200), and a workload that causes
post-warmup symmetric-memory pool growth. No full-model, multi-rank, or NCCL UDS
behavior was exercised here.
