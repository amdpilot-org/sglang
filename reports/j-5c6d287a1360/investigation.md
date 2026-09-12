# GLM-5.3-Flash raw-FP8 TileLang zero-tail correction

Upstream issue: https://github.com/sgl-project/sglang/issues/36830

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1284

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1148

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1248

The reviewed candidate was reproduced at exact source commit
`99ffeba90a6607025df41a45eb7ebbbd3375ece5`, whose parent is the prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Its changes were replayed intact
onto the delivery branch before making this correction.

## Reproduction

The candidate's focused suite reported `10 passed, 2 skipped`. Both numerical
tests were CUDA-only and therefore skipped on the prepared ROCm host, despite
using the GLM-5.3-Flash zero-tail shape.

A direct call to `tilelang_sparse_fwd` with BF16 query `[1, 32, 512]`, raw FP8
KV `[128, 1, 512]`, 64 indices, and `d_v=512` reached
`sparse_mla_fwd_decode_partial_fp8` with `d_tail=0`. Compilation failed before
launch in TileLang layout inference:

```text
tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
```

The otherwise equivalent `d_tail=64` control compiled for gfx950, launched,
synchronized, and returned finite BF16 output. This differential reproduces
the review's concrete counterexample without relying on unavailable CUDA
hardware or model weights.

## Correction

The FP8 partial kernel now uses Python compile-time `d_tail > 0` guards around
the two tail shared-memory allocations, query and KV tail copies, and tail
GEMM. The four 128-wide content tiles and all softmax/value work remain
unchanged, preserving the candidate's valid raw-FP8 routing and layout fixes.

The new GPU regression uses the actual GLM-5.3-Flash boundary (`dim=d_v=512`),
a single valid selected token, and an independent gathered-KV reference. On
the assigned gfx950 it compiled, executed, and matched exactly. A separate
`d_tail=64` one-hot control also remained exact.

## Evidence and limitations

Raw command output is retained under
`/tmp/amdpilot-repo-j-5c6d287a1360/evidence/`; compilation caches are under
`/tmp/amdpilot-repo-j-5c6d287a1360/tilelang-cache*`. Imports resolved from
`/job/repo/python/sglang`, and TileLang loaded from `/opt/tilelang/build`.

The assigned accelerator is one AMD Instinct MI355X/gfx950 with ROCm 7.2, not
NVIDIA H20/SM90. GLM-5.3-Flash weights were unavailable. Consequently this
does not claim full-model serving, CUDA code generation/execution, semantic
accuracy, speculative decoding, capacity, multi-GPU, or multi-node coverage.
No native C++ changed, so a FlyDSL native rebuild was not applicable.
