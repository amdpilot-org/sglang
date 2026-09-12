# Issue 37936 investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/37936

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3323

## Finding

The prepared base already contains a directly relevant candidate correction from
upstream commit `3c9cea8f` / PR https://github.com/sgl-project/sglang/pull/35546.
`EagleDraftExtendInputBuffers.select_index` is derived from the captured request
width but has the same `[max_bs]` shape for every adaptive EAGLE runner. The
process-wide input pool otherwise aliases tensors by field name, size, dtype,
and device. Pooling `select_index` therefore allows one width's indices to be
used by another width's graph; a wide value such as `7` is invalid for a
four-row narrow graph output and produces the reported gather-OOB class of
failure.

The base already excluded `select_index` at the runner call site. This change
makes that safety property intrinsic to `EagleDraftExtendInputBuffers`, so a
future or alternate caller cannot accidentally omit it, and adds a regression
that models narrow and wide capture widths. With the exclusion removed, the
regression fails because both runners obtain the same `select_index` storage;
with it present, both independent valid gathers match their exact references.

This is candidate verification, not a reproduction of the original serving
workload. The assigned hardware is one AMD Instinct MI355X (`gfx950`), not four
Tesla V100 GPUs, and the reported Qwen3.8 Flash-Next NVFP4 model weights and
V100-specific CUDA image were unavailable. Consequently CUDA, NCCL, TP=4,
`tilelang_fa_v100`, the exact concurrent HTTP/tool request, and Xid 43 remain
unverified.

## Evidence

- `raw/failing-before.log`: regression fails when the specialized buffer stops
  excluding `select_index` (`data_ptr` equality proves aliasing).
- `raw/unit-tests.log`: 46 relevant unit tests pass after the correction.
- `raw/gpu-gather-check.log`: the assigned gfx950 produced the exact safe gather
  reference `[11, 13]`; an intentionally invalid boundary dispatch then raised
  an asynchronous ROCm hardware exception. It was not repeated.
- `raw/gpu-health-after-boundary.log`: a fresh process completed a deterministic
  GPU calculation and synchronization afterward, showing the assigned device
  remained usable.

No native C++ library was changed or rebuilt.
