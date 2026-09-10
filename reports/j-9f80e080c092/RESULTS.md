# Block-scaled FP8 GEMM on gfx942

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440004359`
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Installed Python source: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Installed kernel module: `/sgl-workspace/sglang/python/sglang/kernels/ops/quantization/fp8_kernel.py`
- Installed native package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Installed AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Delivery checkout: `/job/sglang`

## Format and layout

- FP8 dtype on gfx942: `torch.float8_e4m3fnuz`
- FP8 maximum: `224.0`
- Activation scale layout: `[M, ceil(K / 128)]`
- Weight scale layout: `[ceil(N / 128), ceil(K / 128)]`
- Weight matrix layout: `[N, K]`
- Block size: `[128, 128]`
- Output dtype: `torch.bfloat16`

## Selected kernel

- Before the selector fix, HIP always selected `_w8a8_block_fp8_matmul`.
- After the selector fix, HIP selects `_w8a8_block_fp8_matmul_unrolledx4` for small grids.
- The unrolled kernel now uses one stage on gfx942 to stay within the 64 KiB shared-memory limit.

## Numerical gate

All comparisons use the unchanged gate:

```python
torch.testing.assert_close(output, reference, atol=0.5, rtol=1e-4)
```

The reference is an independent dequantized Torch matmul:

```python
A_dequant.to(torch.bfloat16) @ B_dequant.to(torch.bfloat16).T
```

## Covered cases

- `M=1, N=1, K=128`
- `M=7, N=129, K=129`
- `M=63, N=128, K=255`
- `M=256, N=1088, K=512`
- `M=256, N=1088, K=1024`
- Zero block: `M=7, N=129, K=256`
- Outlier block: `M=7, N=129, K=256`

Each case uses sentinel output storage initialized to `-12352.0` and verifies that no sentinel remains after the kernel.

## Bounded timing

Timing uses 10 warm-up launches and 30 timed launches per shape with `torch.cuda.Event`.

| Shape | Base median | Unrolled median |
| --- | ---: | ---: |
| `7x129x256` | `0.065872 ms` | `0.074232 ms` |
| `256x1088x512` | `0.065972 ms` | `0.074312 ms` |
| `256x1088x1024` | `0.065892 ms` | `0.074812 ms` |

The corrected unrolled path is slower than the base path on these MI300X shapes, but it is now reachable and numerically correct.

## Commands

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/quantization/test_w8a8_block_fp8_matmul_hip_edges.py -s
```

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/quant/test_fp8_kernel.py::TestW8A8BlockFP8Matmul::test_w8a8_block_fp8_matmul -s
```

## First GPU execution

- The first real GPU kernel execution completed approximately 16 seconds after the initial GPU probe.
- The stock `test_w8a8_block_fp8_matmul` is vacuous on HIP because its platform branch returns before launching.
- The direct gfx942 control case completed in `10.76 seconds` including Python import and Triton compilation.

## Upstream context

- Read-only issue: `sgl-project/sglang` issue `31783`
- Tested upstream candidate: `sgl-project/sglang` PR `37682`, commit `16ac180da36a9c1a7830e639d28ae28694d052f2`
- PR `37682` makes the unrolled selector reachable but does not fix short-K overshoot or the gfx942 shared-memory limit.
- This change preserves the tested commit and extends it with correctness fixes and edge-case coverage.

## Uncertainty

- The unrolled path is not faster than the base path on the tested MI300X shapes.
- No full-model weights were downloaded.
- No node-wide state was modified.
