# Independent review of PR 2567

Reviewed https://github.com/amdpilot-org/sglang/pull/2567 at exact commit
`7666e8a3ae6ddc1fef450f226d5b14ea009034e0` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/31117
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2505
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2571

## Verdict

Recommendation: **request changes**. The candidate is a valid partial fix for
eager host issuance, including the host-thread race, but it does not fully
resolve the original permanent-hang contract.

The per-communicator lock correctly spans stream/event bookkeeping and the
native enqueue. The exact candidate passed its six focused tests, its source
check, and a real two-host-thread/two-stream ordering probe on the assigned AMD
Instinct MI355X (`gfx950`). The measured result was `2`, matching the independent
CPU reference.

However, the guard explicitly returns during graph capture. Replaying captured
graphs does not call Python, so two graphs containing collectives from the same
communicator can still run concurrently without this guard. Independent source
inspection also found unbounded polling in both the JIT and legacy AOT native
implementations. Thus a bypass or protocol collision can still become the
silent, permanent GPU hang reported upstream.

## Reproduction and checks

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` fails
`check_host_thread_serialization.py`: it has no `SingleStreamGuard`. The exact
candidate passes that check and all six candidate unit tests. The independent
`check_contract.py` rejects both revisions as full original-issue fixes; on the
candidate it records the eager guard, explicit capture bypass, and unbounded JIT
and AOT polling.

Commands and complete output are retained under `raw/`. Imports were verified
to resolve to `/job/repo/python`, rather than an installed SGLang copy.

## Architecture and build limitations

The assigned machine has one AMD MI355X (`gfx950`) with ROCm 7.2. The original
reproduction requires two NVIDIA A100 GPUs, CUDA green contexts, and NVLink.
Consequently the original multi-rank CUDA deadlock and concurrent CUDA graph
replay could not be executed here. The single-GPU probe validates only the host
guard's ordering mechanics; it is not a custom-all-reduce or original-issue
reproduction.

No native source differs between the base and candidate, so no native rebuild
was applicable. Native modules were not used as evidence for the Python-only
guard. No full model or distributed workload was run.
