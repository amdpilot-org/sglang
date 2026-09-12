# Independent review of candidate PR 2371

Upstream issue: https://github.com/sgl-project/sglang/issues/31117

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2295

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2407

Candidate reviewed at exact commit
`39673680441fb67c3934e0139f317742a2aa4353`, whose sole parent is the recorded
base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

**Request changes.** The candidate is a useful partial fix for eager calls made
from the normal single host thread: its per-communicator event dependency orders
a launch on a new stream after the preceding stream, and the focused tests and a
real gfx950 ordering probe pass. It does not fully resolve the original failure
contract because it deliberately does nothing during graph capture/replay and
has no device-side deadline. Concurrent graph replays sharing a communicator can
therefore still enter the same aliased rendezvous state and retain the original
silent, permanent spin. Concurrent calls from multiple host threads also race
the unlocked guard; the upstream proposed fix explicitly documents this
assumption and relies on its omitted device deadline to turn that race into an
error rather than an unbounded hang.

This is not merely a missing test. The candidate's own capture test asserts that
the guard records no dependency, and the independent adversarial probe confirmed
that captured work followed by an eager call on another stream leaves both
`Event.record` and `Stream.wait_event` unused. The current upstream issue comment
and upstream PR 31135 describe concurrent graph replay as a same-communicator
entry path requiring device-side bounded spins.

## Evidence and limitations

The candidate regression fails on the base at import/collection because the
guard is absent, then passes 5/5 at the candidate commit; the candidate plus the
adjacent V2 capability suite passes 9/9. Imports resolve to `/job/repo/python`,
not an installed stale copy. A real two-stream event-ordering probe on the one
assigned AMD Instinct MI355X (`gfx950`, ROCm 7.2) returned `2`, matching its CPU
reference, and confirmed event creation.

The original failure requires two CUDA/NVLink GPUs and CUDA custom-all-reduce
kernels. This environment has one AMD gfx950 GPU, so the two-rank A100 deadlock,
green-context split, CUDA kernel execution, and full model/serving behavior were
not reproduced. The candidate changes Python only; no native source changed and
no native rebuild was applicable. The gfx950 probe validates the host event
ordering primitive, not CUDA CustomAllReduceV2 or the original collective.

Full raw evidence was preserved outside revision switches at
`/job/review-evidence-j-a7de22d0ff63`. A concise command/result record is included
in `raw/review-evidence.txt`.
