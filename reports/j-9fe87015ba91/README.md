# Investigation result: Qwen-VL video double sampling

Upstream issue: https://github.com/sgl-project/sglang/issues/31200

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2284

The reported defect is already fixed in the prepared `main` checkout at
`358c163250ad3b1f62939b01ce1314a0a31a0365`. No source correction was added.

The issue was filed on 2026-07-14. Upstream PR
https://github.com/sgl-project/sglang/pull/30260 merged on 2026-07-28 and added the
relevant Qwen-specific routing:

- `preprocess_video` returns the original video metadata alongside the sampled
  tensor.
- `_get_processor_video_config` removes SGLang-owned sampling and resize keys
  (`fps`, frame limits, and pixel sizing) before already-preprocessed frames are
  sent to the Hugging Face processor.
- `process_mm_data_async` passes the metadata and `do_sample_frames=False` for
  Qwen3-VL-family models.

This removes the issue's trigger: an already sampled eight-frame tensor is no
longer accompanied by `videos_kwargs.fps` or `videos_kwargs.max_frames` at the
Hugging Face boundary. Non-sampling processor options remain intact. The
prepared Transformers 5.12.1 additionally routes the top-level
`do_sample_frames=False` into its merged video kwargs.

## Evidence

`raw/routing_reproduction.log` records a deterministic exercise of the actual
checked-out helpers and `BaseMultimodalProcessor.process_mm_data`. Given the
issue's `fps=2` and `max_frames=30` configuration plus preprocessed-video
metadata, the actual processor call contained:

```text
actual_processor_call_do_sample_frames= False
actual_processor_call_videos_kwargs= {'do_normalize': False}
```

It also directly exercised Transformers 5.12.1 `_merge_kwargs`, which produced
`{'fps': 2, 'do_sample_frames': False}` for the issue's proposed boundary case.

`raw/focused_tests.log` records four passing existing regressions. They cover
sampling-key removal for preprocessed Qwen videos, the independent unprocessed
video boundary, ordinary video kwargs injection, and single-call filtered
config routing.

## Limitations

The reported Qwen3-VL/Qwen3.5-VL model weights and original NVIDIA H20 were not
available, so no full model/server token-count comparison was claimed. The
qualified tiny Llama fixture is not a Qwen-VL architecture and could only test
transport, so using it would not validate this issue. GPU execution was not
relevant to the deterministic CPU preprocessing/routing defect and was not
performed. No native code changed or was rebuilt.
