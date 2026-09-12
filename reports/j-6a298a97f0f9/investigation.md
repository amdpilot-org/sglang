# Investigation: sglang#35437

Upstream issue: https://github.com/sgl-project/sglang/issues/35437

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1383

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The upstream reporter's follow-up isolates the breakable-graph startup crash to
VRAM exhaustion during the default prefill bucket sweep. At that boundary,
PyTorch 2.13 CUDA capture bookkeeping is lost and `capture_end()` reports the
secondary `markCaptureEnd called with no captures in progress` assertion. The
reporter measured that stopping before another capture below 0.3 GiB avoided
the corrupted-capture boundary.

The checked-out implementation still entered every configured prefill capture
unconditionally. It sampled available memory only for progress display. The
change stops before starting another bucket below the reported boundary,
preserves all successfully captured buckets for replay selection, and raises a
clean actionable error if there is insufficient headroom before the first
bucket. Unit coverage exercises 0.29 GiB after one successful capture, the
exact 0.30 GiB boundary, and 0.29 GiB before any capture.

The report's second failure (`state_indices_list[bs - 1]` with the Full backend)
could not be reproduced with the unavailable Qwen3.8 NVFP4 target and DFlash2
draft weights. The current tree has substantially changed since PR #35371, but
there is not enough issue-specific runtime evidence here to claim that failure
fixed or to make a speculative state-index change.

Hardware execution used the assigned AMD Instinct MI355X (`gfx950`), Torch
2.11.0+rocm7.2. A warmed HIP graph performing FP32 matrix multiplication was
replayed and compared to an independently computed CPU result. This validates
the available graph runtime only. It does not reproduce the CUDA 13 / SM120
allocator behavior, NVFP4 model execution, DFLASH semantic accuracy, or a
distributed workload. The raw pytest result is retained in
`evidence/gfx950_graph_smoke.junit.xml`.
