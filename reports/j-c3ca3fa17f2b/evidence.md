# Router GEMM FP32 evidence

Environment: AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, Torch
`2.11.0+rocm7.2`, HIP `7.2.26015`. Tests used the prepared interpreter at
`/tmp/amdpilot-repo-j-c3ca3fa17f2b/venv/bin/python` and private caches under
`/tmp/amdpilot-repo-j-c3ca3fa17f2b/cache`.

## Failing baseline

Before the patch, a `[5,7168] x [7168,256]` BF16 persistent GEMM followed by
the `aten::mm.dtype` compatibility handler reported:

```text
old dtype torch.bfloat16 compat dtype torch.float32
compat exactly bf16 widened True
non-bf16-representable compat values 0
old max abs vs fp64-rounded-fp32 0.832183837890625
compat max abs vs fp64-rounded-fp32 0.832183837890625
reference non-bf16-representable 1280
```

The reference was computed independently on CPU as BF16 inputs converted to
FP64, multiplied in FP64, then rounded once to FP32.

## Patched router-shape numerical probe

For `[17,7168] x [7168,256]`, with sub-batches spanning both sides of the
normal router threshold:

```text
gpu AMD Instinct MI350X gfx950:sramecc+:xnack-
dtype torch.float32 shape (17, 256)
non_bf16_values 4351 of 4352
max_abs_error_fp32_store 0.00030517578125
max_abs_error_bf16_rounded 0.966156005859375
mean_abs_error_fp32_store 3.0767176212975755e-05
mean_abs_error_bf16_rounded 0.09471047669649124
repeated_equal True
sub_batch 1 equal True dtype torch.float32
sub_batch 4 equal True dtype torch.float32
sub_batch 5 equal True dtype torch.float32
sub_batch 16 equal True dtype torch.float32
sub_batch 17 equal True dtype torch.float32
```

The actual deterministic `MoEGate.forward` branch reported:

```text
gate dtype torch.float32 shape (17, 256) non_bf16_values 4352
gate repeated_equal True
gate sub_batch 1 equal True
gate sub_batch 4 equal True
gate sub_batch 5 equal True
gate sub_batch 16 equal True
```

The near-tie regression constructs scores `100.0` and `100.0625`. Direct FP32
storage selects expert 1; BF16 storage rounds both to `100.0` and selects
expert 0.

## Compiler evidence

Generated artifacts:

- FP32 specialization:
  `/tmp/amdpilot-repo-j-c3ca3fa17f2b/cache/triton/2P54KICCZ542CVCKUQPPJ4HALOHH3U5K73KDJBKHZOUP7VFXTXQQ/matmul_kernel_persistent.ttgir`
- BF16 specialization:
  `/tmp/amdpilot-repo-j-c3ca3fa17f2b/cache/triton/SFVBDMWWIFR66EFQJHD3WXTO72I4IIHPU4I7CHKAFM7EJCOBF3LQ/matmul_kernel_persistent.ttgir`

The FP32 artifact has BF16 input pointers, an FP32 output pointer, the same
`#ttg.amd_mfma` reduction instruction shape, and direct stores such as:

```text
amdg.buffer_store %accumulator_92, ... : tensor<128x128xf32, #mma>
```

The BF16 artifact instead has a BF16 output pointer and stores:

```text
amdg.buffer_store %c_125, ... : tensor<128x128xbf16, #linear>
```

This is compiler evidence that the FP32 path reaches the existing direct FP32
store without an intermediate BF16 output allocation.

## Unsupported paths

CUDA and DeepGEMM execution remain unverified on this ROCm-only assigned GPU.
The implementation routes FP32 requests to Triton before DeepGEMM selection,
and a mocked dispatch regression proves the BF16-only DeepGEMM function is not
called for an FP32 request.
