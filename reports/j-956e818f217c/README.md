# Investigation report: Vision Triton window attention and sinks

Upstream issue: https://github.com/sgl-project/sglang/issues/37983

Mirror issue: https://github.com/amdpilot-org/sglang/issues/831

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, `VisionTritonAttention`
accepted `window_size` and `s_aux` through `**kwargs` but did not forward either
value. The underlying `context_attention_fwd` kernel had no corresponding
parameters or masking/denominator logic. A focused baseline run on the assigned
MI350X/gfx950 showed that the legacy full-attention case agreed with a PyTorch
reference, while the backend's window-and-sink result did not.

The correction forwards the two arguments, applies FlashAttention-compatible
left/right window bounds, and treats each per-query-head sink as a value-less
virtual softmax logit. The online-softmax state is initialized from the sink and
guards fully masked window blocks against NaNs. Default arguments retain the
existing behavior for callers that do not request either feature.

The regression uses an independent FP32 PyTorch implementation plus a small
hand-computed example. It covers full attention, windows, sinks, their
combination, ragged batches, GQA, causal masking, one-sided and very small
windows over multiple Triton blocks, backend propagation, and invalid sink
shape. Raw logs and the structured claims are retained beside this report.

The MiMo-V2.5 checkpoint was not available, and the assigned hardware is AMD
gfx950 rather than the reported NVIDIA B40/H20. Therefore this work does not
claim a full-model output-quality reproduction or NVIDIA-specific validation.
