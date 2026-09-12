# Independent review of amdpilot-org/sglang#2390

Candidate commit: `0550ec5195800baec69bdf9c67cf7a7feb2affa6`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/31023

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2327

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2425

## Recommendation

Accept the candidate as **test-only hardening**. It adds a focused regression
for the corrected planner implementation already present in the recorded base.
It is not itself the production fix and it does not fully resolve or fully
verify the original open issue.

The candidate changes only a test and its own reports. Its test exercises the
real GPU-input `CompressorPrefillPlan.generate` path and checks that valid
write-plan `ragged_id` values are exactly `range(sum(extend_lens))`. This is a
relevant producer-side oracle, unlike a serving startup smoke.

## Source and native-path verification

The interpreter was
`/tmp/amdpilot-repo-j-007d5124beab/venv/bin/python`. `sglang` imported from
`/job/repo/python/sglang/__init__.py`, and Torch imported from
`/opt/venv/lib/python3.12/site-packages/torch/__init__.py` as
`2.11.0+rocm7.2`.

The tested wrapper is
`python/sglang/kernels/ops/attention/dsv4/compress.py`, which calls
`load_jit(... cuda_files=["deepseek_v4/c_plan.cuh"] ...)`. The source under
test was therefore
`python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` from the checked-out
candidate/base. A fresh gfx950 build produced
`/job/.cache/sglang/jit/gfx950/sgl_kernel_jit_dpsk_v4_compress_plan/build-0bbc057a34dba450/deps-666e3e1963b72122/sgl_kernel_jit_dpsk_v4_compress_plan.so`.
The build completed at 2026-09-12 14:12 UTC during this review. Candidate
commit 0550ec5 changes no C++/HIP/CUDA source, so no additional native rebuild
was required; the fresh JIT compilation verifies the actual checkout source.

The recorded base already lacks the five-line shared-scratch initialization
removed by upstream PR #32467. Each warp overwrites its own slot, and the
existing block barrier precedes warp 0's consumption. Thus the production
correction predates the candidate and is identical at base and candidate.

## Results

- Candidate regression: `3 passed` on one AMD Instinct MI350X (gfx950).
- Adjacent DSV4 planner suite: `5 passed, 82 subtests passed`.
- Independent adversarial harness: 500 successful GPU planner launches across
  25 cases. Cases included batch sizes 1 through 1024, zero-length rows,
  partial/full warp boundaries, extend length 32/31, and deterministic random
  ragged vectors. The oracle checked cardinality, uniqueness, exact sorted IDs,
  and `max(ragged_id) < sum(extend_lens)`.
- Historical failing-before attempt: after temporarily restoring exactly the
  five initialization lines removed by #32467, the two reported ragged shapes
  completed 4,000 gfx950 launches with zero bad plans. The temporary source
  change was reverted before leaving the candidate checkout.

The last result means the original timing-sensitive failure was **not
reproduced** on the assigned AMD architecture. It does not refute the issue's
NVIDIA B300 evidence. It prevents this review from claiming a local
failing-before/passing-after reproduction.

Raw logs and fetched issue/PR metadata were preserved across revision switches
outside the checkout at `/job/review-evidence-j-007d5124beab/`.

## Scope classification and limitations

This candidate is relevant regression coverage and is safe to accept, but
`fully_resolves_original` is false. The original issue describes a larger TP8
CUDA Graph contract and remains open. The assigned environment had one AMD
MI350X gfx950 GPU, not eight NVIDIA B300/B30Z devices. No TP8 rank agreement,
CUDA Graph startup capture, sustained replay, NCCL error propagation, or exact
historical A1 c256/2560-request workload was run. DeepSeek-V4-Pro-DSpark weights
were unavailable, so no full-model or semantic claim is made. The tiny Llama
fixture would not qualify this DSV4-specific kernel and was intentionally not
used as substitute evidence.
