# Investigation result: upstream issue 35345

Upstream issue: https://github.com/sgl-project/sglang/issues/35345

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1391

The prepared base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains a direct solution to the reported defect. No production source
change is proposed by this report.

## Evidence

The fused kernel now accepts either `[T]` positions or `[3, T]` mRoPE
positions. For mRoPE it loads an axis for each rotary lane from
`MRotaryEmbedding.axis_map`, then indexes the temporal, height, or width row
with the real row stride. `Qwen3_5AttentionDecoderLayer` passes that axis map
when positions are two-dimensional. Thus the current source no longer treats
a contiguous `(3, T)` tensor as a flat pointer to row zero.

The existing registered kernel test covers:

- ordinary one-dimensional positions against an independent PyTorch reference;
- distinct-axis mRoPE for sectioned and interleaved layouts against
  `MRotaryEmbedding.forward_native`;
- the Qwen3.5 `[11, 11, 10]` geometry;
- a full-rotary boundary (`[24, 20, 20]`); and
- rejection when two-dimensional positions and the axis map are not supplied
  together.

On the assigned AMD Instinct MI350X (`gfx950`), the unmodified test initially
failed during compilation because `_pdl_supported()` uses
`torch.cuda.get_device_capability()` without excluding HIP. In this ROCm build
that enables the CUDA-only `griddepcontrol.launch_dependents` instruction.
This is a separate backend-detection problem, not evidence about the reported
NVIDIA mRoPE defect. The raw failure is retained in
`raw/current_gpu_test.log`.

After disabling only that optional PDL specialization in the test process, the
existing suite passed (`3 passed`, including five subtests). The focused
regression script also compared the current fused output with native mRoPE on
distinct temporal/height/width positions:

```text
fixed_distinct_axis_max_abs_error=0.03125000
old_temporal_only_max_abs_error=10.50000000
equal_axis_vs_1d_max_abs_error=0.00000000
```

The current result passes the suite's BF16 tolerance. Replaying the old bug by
calling the one-dimensional specialization with only `positions[0]` produces
the large independent mismatch. Equal-axis mRoPE is exactly equal to the 1D
control.

## Limitations

The assigned GPU is AMD gfx950, while the issue specifically reports NVIDIA
CUDA dispatch. No NVIDIA kernel, Qwen3.6/Qwen3.8 weights, 27B server, image
request, tensor-parallel run, or end-to-end visual-semantic accuracy was
executed. The evidence qualifies the current Triton mRoPE calculation on one
gfx950 and the source-level dispatch contract only. It does not qualify model
quality or a multi-node workload.

Upstream PR https://github.com/sgl-project/sglang/pull/35744 is still open and
describes a related implementation. The prepared base was inspected directly;
this report does not duplicate that PR or modify upstream.
