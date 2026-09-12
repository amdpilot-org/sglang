# LM-head FP32-output review

The implementation already present at base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`
is a viable opt-in candidate. Dense CUDA/ROCm BF16 or FP16 heads use
`torch.mm(..., out_dtype=torch.float32)` before any TP gather. FP32 logits take
the generic collective fallback because the multimem implementation explicitly
accepts BF16 only. Attention-TP gathering and TP all-to-all allocate output from
the input dtype, so they also preserve FP32.

The default remains disabled. The issue supplies useful B300 measurements, but
this job had only one MI355X and no model weights; that is insufficient evidence
to impose the doubled FP32 collective traffic on all deployments by default.

## Evidence

- `gpu_check_output.json` compares direct FP32 GEMM output with both an
  independent explicit-FP32-input reference and post-BF16 casting. Across
  synthetic 8192x4096 cases, direct output max error was 6.9e-5 to 7.9e-4;
  post-cast max error was 0.50 to 0.98.
- The constructed ranking case has reference/direct logits `[64, 64.0078125]`
  and post-cast logits `[64, 64]`; direct output preserves token 1 while the
  BF16-output path selects token 0.
- Four independently computed vocabulary shards had the same top-1 token as the
  full-width GEMM for all tested rows. They were not always bitwise identical
  (max delta 3.1e-4), presumably because different GEMM shapes can select
  different algorithms. This is single-device shard arithmetic, not TP4.
- Regression tests assert that the FP32 dtype reaches the logits gather, that
  the BF16-only multimem wrapper selects the generic gather without casting,
  and that the constructed ranking difference is retained.

## Unverified

- No live TP2/TP4 NCCL/RCCL or multimem run (only one assigned GPU).
- No B300/CUDA execution and no compiler/ISA claim.
- No GLM-5.2, DeepSeek V4, or other real-model accuracy/performance benchmark.
- No distributed all-gather timing; the recorded event timings cover GEMM plus
  the post-cast kernel only.
- Quantized and architecture-specific LM-head implementations remain governed
  by their own quantization methods; this review qualifies the dense path.
