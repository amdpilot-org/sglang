# Independent review of PR 2640

Upstream issue: https://github.com/sgl-project/sglang/issues/31117

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2643

Candidate: https://github.com/amdpilot-org/sglang/pull/2640 at exact commit
`0ff2c1d35ce43c13eb822cf0a502dce5df535044`.

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Request changes. The candidate is a meaningful partial fix, but the available
evidence does not establish bounded failure for the original permanent-hang
contract.

The candidate adds a per-communicator host lock and cross-stream event ordering
for eager launches. It also prevents both communicator implementations from
being selected during graph capture and rejects a direct capture-time launch.
Those changes close the two known Python-level entry paths tested here.

However, the JIT custom-all-reduce payload poll remains an unconditional
`do { ... } while (true)`, and the legacy AOT barriers still use unconditional
device-side `while` polls. The candidate changes no native source and supplies
no bounded failure path for a rendezvous collision or an unforeseen/bypassing
entry path. Consequently it hardens the known host paths but does not fully
resolve the issue's defining property: a protocol mismatch can remain a silent,
permanent GPU hang with no timeout or recovery.

## Reproduction and checks

All Python imports in candidate testing resolved from `/job/repo/python`. The
prepared interpreter was
`/tmp/amdpilot-repo-j-38d0ca26e0ac/venv/bin/python` (Torch 2.11.0+rocm7.2).

On the recorded base, the candidate's graph-contract checker exited 1 because
`SingleStreamGuard` did not exist. Its focused test module failed collection for
the same reason. This is the expected failing-before result for the added host
contract, not a reproduction of the original CUDA deadlock.

At the exact candidate commit:

- `python reports/j-ab640134d06f/check_graph_capture_contract.py /job/repo`
  exited 0 and reported direct capture rejection plus capture-time exclusion
  for both legacy and V2 communicators.
- `python -m pytest -q test/registered/unit/distributed/test_custom_all_reduce_stream_guard.py test/registered/unit/distributed/test_custom_all_reduce_v2_capability.py`
  exited 0 with 12 passing tests.
- `HIP_VISIBLE_DEVICES=0 python reports/j-ab640134d06f/gpu_capture_guard_probe.py`
  exited 0 on an AMD Instinct MI350X and observed a real HIP capture reject the
  guard call without changing its stream/event state.
- `HIP_VISIBLE_DEVICES=0 python reports/j-e89cf3de41b7/gpu_host_thread_probe.py`
  exited 0 and produced ordered numeric result 2, equal to CPU reference 2.
- A source audit found no candidate diff under the JIT/AOT native source trees.
  The unbounded loops remain at
  `python/sglang/kernels/jit/csrc/distributed/custom_all_reduce.cuh:177-188`
  and `python/sglang/kernels/aot/csrc/allreduce/custom_all_reduce.cuh:211-216`
  (with further unconditional barriers at lines 400 and 458).

Evidence captured while revisions were switched is retained outside the
checkout at `/job/review-evidence/`.

## Environment limitations

The assigned machine has one AMD Instinct MI350X (`gfx950`) with ROCm 7.2. It
does not have the two NVIDIA A100 CUDA GPUs, NVLink topology, CUDA green
contexts, or two-rank setup required by the original reproducer. Therefore the
original deadlock, concurrent replay of two CUDA graphs, CUDA compiler/ISA
behavior, and timeout behavior were not executable here. The real GPU probes
measure host ordering and capture refusal only; no custom-all-reduce kernel or
distributed workload ran. No native source changed, so a native rebuild was not
applicable. No model weights or serving workload were used.
