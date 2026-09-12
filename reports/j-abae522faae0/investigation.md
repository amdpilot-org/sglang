# Independent review of amdpilot-org/sglang PR 1148

Reviewed exact candidate commit `99ffeba90a6607025df41a45eb7ebbbd3375ece5`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/36830

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1183

## Recommendation

Request changes. The candidate adds most of the routing needed for an all-TileLang
raw-FP8 KV path, but it does not fully resolve the original GLM-5.3-Flash issue.

GLM-5.3-Flash has `qk_rope_head_dim = 0`, so `tilelang_sparse_fwd` invokes
`sparse_mla_fwd_decode_partial_fp8` with `d_tail = 0`. At the exact candidate,
that kernel unconditionally allocates and copies zero-width tail buffers and
unconditionally emits the tail `T.gemm`. A direct model-shaped call on the
assigned gfx950 fails in TileLang layout inference with:

```text
tvm.error.InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
```

The same candidate kernel with `d_tail = 64` compiled and executed successfully
on the same device and toolchain. This differential result makes the zero tail,
not a general prepared-toolchain failure, the relevant counterexample.

The candidate's CUDA numerical tests also instantiate `q.shape[-1] == d_v ==
512`, hence cover the important zero-tail shape in principle, but both are
skipped on the prepared ROCm host. The candidate report treated its gfx950
divide-by-zero as an environment limitation rather than recognizing the
model-specific zero-tail failure. Upstream PR 36904 contains independent
hardware feedback reporting the same `Unsupported k_dim 0` failure for
GLM-5.3-Flash and the required compile-time `d_tail > 0` guards; those guards
are absent from candidate commit 99ffeba.

## Scope and limitations

The prepared base reproduced the pre-fix rejection through the candidate's
validation regression (4 failures, 6 passes), including rejection of the SM90
TileLang/FP8 route. On the candidate, its focused suite reported 10 passes and
2 CUDA skips, confirming its argument validation and layout plumbing but not
the original CUDA execution contract.

The assigned accelerator is one AMD Instinct MI350X/gfx950 with ROCm 7.2, not
an NVIDIA H20/SM90. No GLM-5.3-Flash weights were available. Therefore no full
model, CUDA, speculative-decoding, multi-GPU, semantic-accuracy, or capacity
claim is made. The direct GPU differential is nevertheless valid evidence that
the exact candidate retains a zero-tail kernel-generation defect on the shape
required by the original model.

No native C++ or FlyDSL source changed, so no native rebuild was applicable.
Imports resolved to `/job/repo/python/sglang`, including the candidate's
`tilelang_kernel.py`, rather than an installed wheel.
