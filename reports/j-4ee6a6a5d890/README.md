# gfx942 fused SiLU post-quant investigation

This report covers the contiguous BF16-to-FP8 execution path from sgl-project/sglang issue 32065. It intentionally does not repeat the original zero-row trigger from amdpilot-org/sglang issue 208; zero rows are covered only as a validated no-op contract.

## Evidence

- `baseline-first.json` is the installed-source baseline captured before checkout changes. The target JIT build failed because `cuda_fp8.h` was unavailable to `hipcc`; the supported neighboring `sgl_kernel.silu_and_mul` BF16 control matched an independent FP32 reference.
- `gpu-results.json` contains post-checkout gfx942 results for contiguous and masked BF16-to-FP8 calls, native dispatch names, private JIT module paths, sentinel checks, address checks, clear-refusal errors, and a bounded five-iteration timing matrix.
- The committed test covers the previously uncovered contiguous path: numerical comparison, unchanged output addresses, sentinel-protected storage, zero-token no-op, unsupported FP16 refusal, and unsupported launch-width refusal.

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-4ee6a6a5d890/jit
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/moe/test_silu_and_mul_contig_post_quant.py
```

The focused run passed all four tests on one AMD Instinct MI300X (`gfx942`) with Torch 2.9.1 and ROCm 7.2. No model weights or node-wide state were used.
