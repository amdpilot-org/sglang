# Independent review of PR 2129

Candidate: https://github.com/amdpilot-org/sglang/pull/2129 at
`61a5235257fc808054bc3fe0061f763b49c5293f`

Upstream issue: https://github.com/sgl-project/sglang/issues/32470

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2069

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2164

## Recommendation

`request_changes`. The candidate provides useful test-only hardening, and its
runtime planner tests pass on the available gfx950 GPU, but it does not itself
contain the production fix and its source regression is narrower than the bug's
actual synchronization contract.

The prepared base already contains the production correction merged by upstream
PR 32467. Historical inspection shows that upstream first fixed the race by
placing `__syncthreads()` between scratch initialization and per-warp reduction
(commit `c9c96b63f765db4ac7c88e824a755851f1049e30`), then removed the now-useless
initialization (commit `58124fb983136b903108c8ef1305c135eb7d376f`). The candidate's static test requires
the latter implementation and would reject the former correct, race-free fix.
It should instead check the synchronization/ownership contract without rejecting
a valid barrier-based implementation.

The candidate also records `git diff --check` with exit code 0, but the exact
candidate commit exits nonzero because its two retained pytest logs contain
trailing whitespace.

## Validation

The exact candidate was checked out detached and imported from `/job/repo/python`.
PyTorch came from `/opt/venv/lib/python3.12/site-packages`, reported
`2.11.0+rocm7.2`, and used one AMD Instinct MI350X (`gfx950`). The checked-out
DSV4 compress Python path was `/job/repo/python/sglang/kernels/ops/attention/dsv4/compress.py`.

The candidate suite passed all four tests. Its JIT planner library was compiled
from the checkout at
`/job/.cache/sglang/jit/gfx950/sgl_kernel_jit_dpsk_v4_compress_plan/build-0bbc057a34dba450/deps-666e3e1963b72122/sgl_kernel_jit_dpsk_v4_compress_plan.so`.
Independent GPU cases also matched the CPU planner for reversed and alternating
ragged layouts, first/last-warp outliers, batch sizes 2/31/32/33, and a mixed
32/33 extend case.

The historical pre-fix lines used by the candidate's failing-before mutation do
match the actual source before upstream PR 32467. Nevertheless, the original
nondeterministic CUDA illegal-memory-access was not reproduced on the prepared
base, which is already fixed, nor on this AMD system.

## Limitations

Only one AMD gfx950 GPU was available. The reported CUDA H20 TP=2 graph-capture
path, DeepSeek-V4-Flash-DSpark weights, CUDA scheduling behavior, full serving
path, and multi-rank execution were unavailable. Therefore this review verifies
planner metadata and bounds on HIP, not the complete original deployment. No
monolithic native build is configured; the changed candidate content is tests
and reports only, while the exercised planner JIT extension was rebuilt from the
candidate checkout.

Raw commands and output are retained in `evidence/`.
