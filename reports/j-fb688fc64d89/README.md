# Independent review of PR 799

Reviewed exact candidate `4f47bef5921d53d3a40186d03225601411530334`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The added tests correctly harden generic
advisory-lock behavior, but this is a test-only candidate and it does not fully
resolve the original issue's bounded-hang contract.

The prepared base already routes `expert_pack_mxfp4` through SGLang
`load_jit()`, whose `_build_lock` uses `fcntl.flock`. Independent checks confirm
that an ownerless lock path is harmless and that `SIGKILL` of a lock holder
allows immediate reacquisition. This is a real fix for the legacy PyTorch
`FileBaton` stale-file failure mode.

The remaining counterexample is a live, stalled holder. A second acquisition
waited until an external two-second timeout because `_build_lock` has no
deadline. The candidate does not implement the issue-proposed isolated process
group, bounded timeout, process-tree cleanup, fresh cache, or retry. Therefore
it cannot claim containment of compiler or kernel stalls, both explicitly named
in the original issue.

The candidate PR body also cites mirror issue `#757`, not the required mirror
issue `#839`.

The candidate's full JIT-cache suite passed (`44 passed`). The actual MXFP4
test was also attempted from a private cold cache. On the assigned AMD Instinct
MI355X (`gfx950`), the real pinned `hipcc` targeted gfx950 and failed because
the NVIDIA-specific source includes `cuda_bf16.h`. No MXFP4 kernel executed.
No native rebuild was applicable because the candidate changes no native code.

Raw command outcomes and source/import paths are recorded in `raw/review.txt`.
