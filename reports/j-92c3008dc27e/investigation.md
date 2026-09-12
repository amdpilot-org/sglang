# Independent review of PR 1455

Upstream issue: https://github.com/sgl-project/sglang/issues/35437

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1491

Candidate: https://github.com/amdpilot-org/sglang/pull/1455 at
`df5991303c19a1854c4010e9d03b24dcdf9da8b2`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
The image-prepared checkout was exactly this commit and was clean before review.

## Finding

Recommendation: **accept as a partial fix**, not as a full resolution of the
original issue.

The candidate makes a narrow Python-only change to
`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`.
Before each prefill bucket it samples free device memory, stops before entering
another capture below 0.3 GiB, and replaces the replay bucket list with only
the successfully captured buckets. This directly implements the upstream
reporter's confirmed workaround for failure 1. The candidate adds no native
source and therefore requires no native rebuild.

The new regression is meaningful rather than test-only hardening. With the
candidate test file temporarily applied to the recorded base, the low-memory
stop and clean-first-bucket-error cases failed: the base captured all three
buckets and did not raise. At the exact candidate commit, the complete focused
file passed (13 tests and 7 subtests). Independent checks at 0.299999 GiB also
confirmed that replay retains exactly the successfully completed buckets, and
that a chunked-prefix bucket is retained only after all configured prefix
variants return.

This evidence verifies the guard's control flow, but not the original CUDA
allocator failure end to end. The assigned device is an AMD Instinct MI350X,
gfx950, with Torch 2.11.0+rocm7.2. The report requires an RTX 5090 SM120,
CUDA 13, Torch 2.13, and two unavailable Qwen3.8 NVFP4/DFLASH model sets.
Consequently the actual allocator bookkeeping failure and the 0.3 GiB safety
margin could not be independently reproduced on the reported stack.

Failure 2 is not fixed by this candidate. The Full prefill backend still has
the reported `state_indices_list[bs - 1]` accesses in
`python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`, and the
candidate changes neither that file nor the DFLASH request-padding/state-index
path. The original first-request Full-backend counterexample therefore remains
open and could not be executed without the target and draft model weights.

The candidate's gfx950 graph smoke passed and compared GPU FP32 matrix
multiplication with a CPU reference. It establishes that the assigned GPU and
HIP graph runtime execute, but it is unrelated to DFLASH, NVFP4, BCG allocator
bookkeeping, Full-backend state indices, serving transport, or semantic model
output. It is not treated as proof of the original issue.

## Import and source verification

- `sglang`: `/job/repo/python/sglang/__init__.py`
- reviewed runner: `/job/repo/python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
- Torch: `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`
- Torch version: `2.11.0+rocm7.2`
- device: `AMD Instinct MI350X`
- architecture: `gfx950:sramecc+:xnack-`
- visible device count: 1
- native diff: empty; native rebuild not applicable

## Test evidence

1. Base plus candidate regression tests, selected new cases:
   `python -m pytest -q test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py -k 'capture_stops_before_low_memory_bucket or capture_fails_cleanly_when_first_bucket_has_low_memory or capture_allows_exact_memory_boundary'`
   exited 1 with 2 failed and 1 passed. The base made calls for 4, 2, and 1
   tokens instead of stopping, and did not raise at 0.29 GiB.
2. Exact candidate focused suite:
   `python -m pytest -q test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py`
   exited 0 with 13 passed and 7 subtests passed.
3. Independent mocked boundary/prefix script exited 0. At memory samples
   `[1.0, 0.3, 0.299999]`, the resulting replay grid was `[2, 4]`; with
   chunked-prefix variants `[1, 2]`, all three calls for bucket 4 completed
   before that bucket was retained.
4. Candidate gfx950 smoke:
   `python -m pytest -q reports/j-6a298a97f0f9/test_gfx950_graph_smoke.py`
   exited 0 with 1 passed. This is architecture evidence only, not an
   original-issue reproduction.

JUnit output from the candidate runs was preserved outside the checkout under
`/tmp/amdpilot-repo-j-92c3008dc27e/review-evidence/` while revisions were
switched.
