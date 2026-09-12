# Issue 28312 investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/28312

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2470

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The prepared `main` source already contains the issue-specific fix merged by
upstream PR https://github.com/sgl-project/sglang/pull/27998. In
`set_mamba_track_indices_from_reqs`, a request whose ping-pong buffer has not
yet been allocated can have `req.kv.mamba_next_track_idx is None`. The current
implementation converts that value to slot `0` before constructing the int64
tensor. Without the guard, `torch.tensor` raises the same exception reported in
the issue.

This change adds direct regression coverage; it does not alter the already
fixed runtime implementation.

## Evidence

- `raw/failing_before.log`: with only the existing `None` guard temporarily
  reverted, the all-unallocated and mixed-position tests fail with
  `TypeError: 'NoneType' object cannot be interpreted as an integer`. The
  explicit-position test passes, isolating the failing default path. The
  source guard was restored immediately after this controlled run.
- `raw/passing_after.log`: all three tests pass against the prepared source.
- `raw/gfx950_check.log`: the actual helper ran on the assigned AMD Instinct
  MI355X (`gfx950`) and selected `[10, 21, 30]` from a mixed `[None, 1, 0]`
  request state, exactly matching the independent expected tensor.

## Limitations

The reported Qwen3.5-397B-A17B-FP8 weights, NVIDIA B300 hardware, CUDA/TensorRT
backend, two-way tensor parallelism, and production traffic pattern were not
available. Therefore this does not claim a full serving-path or model-level
reproduction. The deterministic fixture validates the precise failing helper
and its device tensor selection only. No native library was changed or rebuilt.
