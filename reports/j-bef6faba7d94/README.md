# DeepSeek-V4 gfx942 FP8 writer investigation

## Result

A real single-GPU probe confirmed three defects in the gfx942 software path of
`sglang::deepseek_v4::fp8::pack_fp8`:

- The FNUZ top exponent was treated as overflow. Inputs such as `128`, `160`,
  and `224` became `0x7F`/`0xFF` (`±240`) instead of their representable values.
- Normal and subnormal rounding always rounded ties upward instead of using
  round-to-nearest-even.
- The subnormal underflow boundary flushed values just above half the minimum
  subnormal. An exact negative tie also produced `0x80`, which is FNUZ NaN.

The fix is limited to the ROCm software conversion path. CUDA conversion and the
`gfx950`/`gfx1200`/`gfx1201` hardware conversion branch are unchanged.

## Measured identities

- Control: detached worktree at exact commit
  `484c2286c993d36e862343c390a77439a003d244`, clean before and after probing.
- Runtime candidate: same exact commit plus the fix; candidate commit
  `fe30cca45`.
- Delivery branch: `amdpilot/j-bef6faba7d94`, cut from mirror `main`, containing
  only this cherry-picked fix, regression, and report.
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, driver
  `6.19.14.31400000`.
- Compiler: `/opt/rocm/bin/hipcc`, HIP `7.2.26015-fc0010cf6a`, AMD clang
  `22.0.0git`.

The normal production JIT emitted:

```text
-fPIC -D__HIP_PLATFORM_AMD__=1 -fno-gpu-rdc
--offload-arch=gfx942:sramecc+:xnack-
-DUSE_ROCM -std=c++20 -O3 -DHIP_FP8_TYPE_FNUZ=1
```

Device metadata reported producer policy maximum `224`, FNUZ macro value `1`,
and inactive `SGL_ROCM_FP8_HW_CVT`. FNUZ format maximum is `240`; the call-site
clipping policy remains `224`.

## Direct production-wrapper probe

The wrapper includes the production `fp8_utils.cuh` first and calls the real
`pack_fp8` in both lanes. It contains no copied conversion algorithm.

For inputs `[1, -1, 128, 160, 224, -1e-8]`:

| Variant | Raw bytes | Consumer-visible values |
| --- | --- | --- |
| Baseline | `40 C0 7F 7F 7F 00` | `1, -1, 240, 240, 240, 0` |
| Candidate | `40 C0 78 7A 7E 00` | `1, -1, 128, 160, 224, 0` |

The final baseline failed 23 required scalar cases and 1,839 of 4,096 seeded
random cases. The candidate failed none. Lane-order tests failed none on either
variant. Nonfinite inputs were observed separately and were not used for finite
verification status.

## Real producer path

The production `fused_norm_rope_v2.cuh`
`FusedNormRopeKernel<bf16,128,64,1,false,0,false>::forward` path was also run.
With the same input, plan, scale, and cache layout:

- Baseline wrote 128 `0x7F` value bytes, all decoding to `240`.
- Candidate wrote 128 `0x7E` value bytes, all decoding to the intended policy
  maximum `224`.
- Both variants wrote the same scale bytes `F2 B6 DE 43`.

This confirms the fix reaches the real compressed-cache producer path. It does
not establish whole-model generation correctness.

## Reproduction

From a ROCm PyTorch environment on one gfx942 GPU:

```bash
PYTHONPATH=python python -m pytest \
  python/sglang/test/kernels/deepseek_v4/test_fp8_pack.py
```

The test skips non-ROCm and non-gfx942 devices. It compares the production
wrapper against Torch's real `float8_e4m3fnuz` cast after applying the producer's
declared `224` clipping policy.

## Limitations

- Only gfx942 was run. No gfx950, gfx1200, gfx1201, or CUDA device was run.
- No checkpoint was loaded, no TP8 server was started, and no other GPUs were
  reserved.
- The passing microprobe and producer test do not close issue 35122 or the
  documented long-generation failure. A separate 8-GPU same-checkpoint A/B is
  still required.
