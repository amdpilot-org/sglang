# OLMo-2 QK norm dispatch investigation

Source issue: https://github.com/sgl-project/sglang/issues/33415

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3399

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding and change

The prepared implementation imported from
`/job/repo/python/sglang/srt/models/olmo2.py`. Its non-capture branch called
`RMSNorm.forward_native` directly, while its graph-capture branch called the
module and therefore used normal fused-op dispatch. The non-capture branch now
uses the same public dispatch, retaining the existing dual-stream overlap only
during graph capture.

The focused regression fails on the base revision because eager execution
calls `forward_native`, and passes after the change. It covers the eager branch
and the capture-only dual-stream branch.

## AMD GPU evidence

Tests used the assigned AMD Instinct MI350X with Torch 2.11.0+rocm7.2 and HIP
7.2.26015. Repository imports resolved to:

- model: `/job/repo/python/sglang/srt/models/olmo2.py`
- RMSNorm: `/job/repo/python/sglang/srt/layers/layernorm.py`
- AITER core: `/tmp/amdpilot-repo-j-29840765b7c7/cache/aiter/module_aiter_core.so`
- AITER RMSNorm: `/tmp/amdpilot-repo-j-29840765b7c7/cache/aiter/module_rmsnorm.so`

With `SGLANG_USE_AITER=1`, public RMSNorm dispatch resolved to
`forward_aiter`. GPU results matched an independent FP64 formula for BF16,
FP16, and FP32 at width 2048, for both contiguous tensors and noncontiguous
views made by splitting a packed QKV tensor. The maximum absolute differences
from the rounded FP64 reference were 0, 0.00048828125, and 4.76837158203125e-7
respectively.

A real ROCm graph captured the alternate-stream branch. Capture entered each
Python norm wrapper once; replay after replacing the static input did not
re-enter Python and matched the independent reference exactly. The output
buffer immediately after capture is not treated as a numerical result because
ROCm graph capture records work for replay; replay is the measured result.

Focused timing of two standalone BF16 width-2048 Q/K norm calls measured
2.28x at 1 token, 1.91x at 128 tokens, and 5.24x at 16,384 tokens. These are
AMD kernel microbenchmarks, not full-model prefill/decode measurements.

## Limitations

- Model weights were not available, so OLMo-2 full-model serving, semantic
  accuracy, and end-to-end prefill/decode timing were not run. The tiny Llama
  transport fixture is not an OLMo-2 substitute and was intentionally unused.
- No NVIDIA H200 was assigned. The H200 figures in the source issue remain
  source evidence only and were not reproduced or presented as AMD results.
- Tensor-parallel execution was not exercised; the tested TP=1 widths and
  packed-QKV layout match OLMo-2-0425-1B-Instruct's width 2048 configuration.
- The unrelated FA3 illegal-address report is CUDA-specific and was not
  investigated on this AMD system.
- No native SGLang/FlyDSL source changed and no prepared native rebuild target
  existed. AITER compiled/loaded its real RMSNorm extension from the private
  runtime cache.

Raw logs are retained under `reports/j-29840765b7c7/raw/`.
