# MI300X FP8 memory-scaling evidence

This report records a bounded, one-GPU follow-up to amdpilot-org/sglang issue 391. It uses the supported installed `torch._scaled_mm` FP8 dense-linear kernel with synthetic FP8 weights and per-tensor scales, compares each real block forward against an independent dequantized BF16 matmul plus explicit attention, and measures live allocator bytes for a finite batch/context matrix.

## Reproduction

```bash
cd /job/sglang
/opt/venv/bin/python reports/j-263e9f0b57c0/benchmark_memory_scaling.py
```

The benchmark uses six cases: `(batch, context) = (1,1), (1,32), (2,64), (4,128), (8,256), (16,512)`, with `K=N=2048`, eight attention heads, and `M=batch*context`. It generates only the two FP8 linear weights locally; their combined storage is far below 4 GB. Each case performs one correctness forward, one independent reference forward, three warmup forwards, and ten timed forwards. No checkpoint is downloaded.

Predicted tensor workspace counts the unique underlying storages of the retained BF16 input, FP8 activation, QKV output, attention output, contiguous attention projection input, FP8 attention activation, and final BF16 output. Predicted KV workspace counts the two BF16 KV caches. Each case first performs a real warmup forward, releases its case tensors, records a baseline that includes the synthetic weights and persistent native GEMM workspace, and then performs the measured real forward. Measured live tensor/KV workspace is `torch.cuda.memory_allocated()` minus that case baseline; measured peak tensor/KV workspace uses `torch.cuda.max_memory_allocated()` with the same baseline. The persistent native GEMM workspace is measured separately and reported, not hidden inside the tensor/KV comparison. The script refuses to continue if any case exceeds the 48 GiB live-allocation limit.

The numerical gate is unchanged across cases: `torch.testing.assert_close(..., rtol=0.02, atol=1.0)`, finite final output, and exact output shape. Timing uses CUDA events only; there are no unbounded loops, sleep loops, or synthetic GPU burn loops.

The installed `sgl_kernel.fp8_scaled_mm` wrapper is not a usable ROCm native op in this image (`AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'fp8_scaled_mm'`). That boundary is recorded rather than replaced with a fabricated engine result. The working control is the installed Torch/ROCm `torch._scaled_mm` path using `float8_e4m3fnuz`.
