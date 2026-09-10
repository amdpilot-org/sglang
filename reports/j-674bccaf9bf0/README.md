# gfx942 low-token MoE investigation

This report-only branch records the bounded one-GPU investigation for sgl-project/sglang issue 32312. It does not modify the MoE implementation or duplicate upstream PR 34131.

## Context

Issue 32312 is an open feature request asking whether SGLang should consider warp-decode-style kernels for low-latency small-batch MoE inference. It had no comments at investigation time.

The closest existing upstream candidate found was open PR 34131, which optimizes tiny-batch `moe_align` on the pair axis. Its exact head `4ba78e8ba97491479afc96a9e4a3dfbb5c36ece1` was tested; it does not compile on gfx942 because it references the CUDA-only `atomic::Event` and `cudaMemsetAsync`. No upstream issue, PR, or comment was changed.

## Environment

- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- The container does not expose its image ID through runtime metadata. Hostname is not used as image identity.
- GPU: one assigned AMD Instinct MI300X, gfx942, device ID `0x74a1`, serial `692440004372`, 206 GiB VRAM.
- Interpreter: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Installed SGLang source: `/sgl-workspace/sglang/python/sglang` at `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Installed native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Delivery source: `/job/sglang/python/sglang` at base `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Build and JIT caches were kept under `/tmp/sglang-cache-j-674bccaf9bf0`, outside `/job`.

## First installed-source baseline

The first GPU execution used the preinstalled interpreter and existing fused-MoE test:

```bash
/opt/venv/bin/python -m pytest -q -s \
  /sgl-workspace/sglang/test/registered/moe/test_fused_moe.py::TestFusedMOE::test_various_configurations \
  --maxfail=1
```

The case includes `m=1` and compares Triton fused MoE with an independent `torch_naive_moe` expert-loop reference using `torch.testing.assert_close`. It passed all 384 subtests in 105.98 seconds; the complete first process took 111.079 seconds. This is evidence for the installed source only, not for later checkout changes. The raw artifact is `/job/baseline-first.json`.

## Supported delivery controls

The current-main small-align suite passed 245 tests in 11.51 seconds against its plain-Torch contract oracle:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q -s \
  /job/sglang/test/registered/kernels/ops/moe/test_moe_align_small_numel.py \
  --maxfail=1
```

The focused harness in `gfx942_low_token_moe.py` then exercised six low-token shapes across the installed AOT native path and the in-tree Triton pairwise path. Every case passed its independent plain-Torch reference check. Timing used 5 warmups, 100 calls, and one synchronize after the timed loop.

| Tokens × top-k | Experts | AOT mean | AOT kernel | Triton mean |
| ---: | ---: | ---: | --- | ---: |
| 1 × 2 | 8 | 9.78 µs | small-batch expert kernel | 19.52 µs |
| 1 × 7 | 64 | 11.13 µs | generic align kernel | 25.75 µs |
| 1 × 7 | 257 | 10.82 µs | generic align kernel | 29.00 µs |
| 1 × 16 | 896 | 10.83 µs | generic align kernel | 24.72 µs |
| 5 × 7 | 257 | 7.74 µs | generic align kernel | 19.63 µs |
| 4 × 16 | 896 | 10.56 µs | generic align kernel | 24.53 µs |

The AOT comparison is blockwise multiset equality because its intra-bucket order is atomic scheduling order. The Triton pairwise path is exactly equal to the reference. Raw values and commands are in `gpu-evidence.json`.

## Architecture findings

- gfx942 has a 64-lane wavefront. Code that assumes CUDA warp32 semantics is not portable without an explicit architecture-specific design.
- Current-main `align_single_token.cuh` fails HIP compilation before launch because it passes a 32-bit `0xffffffff` shuffle mask; ROCm requires a 64-bit mask.
- PR 34131 also fails before launch on gfx942 because its CUDA-only `atomic::Event` and `cudaMemsetAsync` are unavailable under HIP without adaptation.
- The AOT small-batch expert kernel is reached only for expert counts no greater than 64. Wider low-token cases use the generic AOT align kernel.
- The Triton pairwise path removes the expert-count limit and passes exact reference checks, but it is slower than the measured AOT paths on these gfx942 align shapes.
- These timings characterize dispatch alignment only. They are not full MoE FFN latency or an end-to-end model benchmark.

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-674bccaf9bf0/sglang-jit
export TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-674bccaf9bf0/tvm-ffi
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-674bccaf9bf0/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-674bccaf9bf0/torch-inductor
/opt/venv/bin/python /job/sglang/reports/j-674bccaf9bf0/gfx942_low_token_moe.py
```

No full model weights or additional framework stack were downloaded. The upstream repository was read only.
