# Vision FA4 window/sink verification

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The base implementation passed `window_size` and `s_aux` from
`VisionAttention.forward()` into the selected backend, but
`VisionFlash4Attention.forward()` omitted both at its call to `flash_attn_func`.
FA3 already preserved the same values. Upstream PR
https://github.com/sgl-project/sglang/pull/38872 contained the same forwarding
direction, so the production change here intentionally stays aligned with it.

`test_vision_fa4_attention.py` instruments the external FA4 function boundary.
It covers the default full-window/no-sink call, window-only, sink-only, and
combined arguments, always requiring `ver=4`. Its end-to-end unit case runs the
ordinary `VisionAttention.forward()` data path through `VisionFlash4Attention`
and compares the returned tensor with a separate float32 implementation of the
equal-length non-causal local mask and learned sink denominator.

The baseline run was made by retaining the new regression while restoring the
recorded-base implementation of `vision.py`. All five cases failed. The
implementation fix was then restored and the focused suite passed. The raw
pytest JUnit documents are `baseline.xml.gz` and `fixed.xml.gz`.

Environment evidence from the prepared interpreter:

- source: `/job/repo`
- Python: `/tmp/amdpilot-repo-j-9ad1f4f7fdea/venv/bin/python`
- native library: not rebuilt; no native source changed
- Torch: `2.11.0+rocm7.2`
- HIP: `7.2.26015`
- visible device with `HIP_VISIBLE_DEVICES=0`: one AMD Instinct MI355X

This host cannot run the CUDA-only FA4 CUTE kernel or qualify the reported
Blackwell model path. The mocks establish the Python argument contract and the
independent reference establishes the intended small-tensor semantics, but
neither is reported as GPU execution.
