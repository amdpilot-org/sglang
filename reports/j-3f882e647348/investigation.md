# Gemma3 RMSNorm investigation

The base implementation at `358c163250ad3b1f62939b01ce1314a0a31a0365`
still contained the reported `x.dim() == 2` restriction and passed
`self.weight.data` to both CUDA fused paths without a dtype/device guard.
Upstream PR https://github.com/sgl-project/sglang/pull/32670 remains open and
contains the same narrow correction; it was inspected before implementation.

The assigned device was an AMD Instinct MI355X (`gfx950`) with Torch
`2.11.0+rocm7.2`. On this backend `Gemma3RMSNorm` uses `forward_hip`, and the
CUDA Gemma symbols are intentionally not imported into `layernorm.py`.
The wheel also has no executable `torch.ops.sgl_kernel.gemma_rmsnorm` ROCm op.
Consequently, the CUDA dispatch defect was reproduced against the actual
`forward_cuda` method with kernel spies, while GPU numerical validation used an
independent explicit RMSNorm formula. This does not qualify CUDA kernel output,
the reported NVIDIA NaNs, Gemma-3 model semantics, serving, or performance.

Raw commands and outputs are retained under `reports/j-3f882e647348/raw/`.
