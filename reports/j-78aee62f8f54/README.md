# DSpark DeepSeek-V4 compress-state investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32038

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2157

## Result

The prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the
compress-state write-plan correction implicated by the B300 follow-up on the
source issue. No additional source correction is justified.

The current planner in
`python/sglang/kernels/jit/csrc/deepseek_v4/c_plan.cuh` derives the trailing
write pad from ring capacity (`ring_size - window_size + 2`) and applies it to
both the GPU-input and CPU-input planner paths. The configuration guard in
`python/sglang/srt/model_executor/pool_configurator.py` rejects speculative
draft counts that exceed that capacity.

The existing regression
`test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py`
independently covers:

- the original c4 boundary at five planner draft tokens (six target verify rows
  in the issue configuration), plus widths below and above it;
- every relevant tested `seq_len % compress_ratio` residue;
- c4 and c128 rings;
- agreement between the CPU-input and GPU-input planner implementations;
- unchanged non-speculative prefill write sets; and
- behavior beyond ring capacity, paired with the startup guard.

## Failing-before / passing-after evidence

For a controlled failing-before check, only the current ring-derived expression
was temporarily replaced with the historical hard-coded pad of four. The c4
residency regression then failed at `D=5, prefix=515`, reporting that committed
position 515 was omitted from `plan_w`. Wider independent cases failed as well.
The checked-in expression was restored, and the identical focused test passed
all 28 subtests. The complete checked-in regression passed 5 tests and 82
subtests on the assigned gfx950 GPU.

Raw logs:

- `raw/historical_pad_failing_before.log`
- `raw/current_pad_passing_after.log`
- `raw/compress_plan_regression.log`

## Scope and limitations

GPU execution used one AMD Instinct MI355X (`gfx950`) with PyTorch
`2.11.0+rocm7.2`; the test exercised the actual JIT-compiled GPU planner and
compared it with the host planner. This is kernel/planner invariant evidence,
not a full serving or model-quality reproduction.

DeepSeek-V4-Flash weights were not present. The assigned hardware is one AMD
gfx950, while the report used four NVIDIA B300 GPUs with TP=4. Therefore the
reported AIME25 score change, the B300-specific serving stack, TP=4, MXFP4 MoE,
and end-to-end DSpark semantic accuracy remain unverified here. The tiny Llama
transport fixture cannot qualify the DeepSeek-V4 compressed-attention model
architecture and was consequently not substituted for the original issue.
