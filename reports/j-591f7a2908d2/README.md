# Investigation of sglang issue 30550

Upstream issue: https://github.com/sgl-project/sglang/issues/30550

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2375

## Outcome

The prepared `main` source at `358c163250ad3b1f62939b01ce1314a0a31a0365`
already contains the solution. No product-code correction is justified.

The reported failure was a test-mock defect, not a failure of
`SchedulerProfilerManager._start_profile`: the manager selects the MPS capture
strategy, `torch.mps.profiler.metal_capture`, while the old test mocked the
independent MLX strategy, `mx.metal.start_capture`. The real MPS capture was
therefore left active and failed when Metal capture was not enabled.

Upstream commit `2969ab3d4147e2ec76ec0c9b2b40bd32454f45f5` in PR
https://github.com/sgl-project/sglang/pull/34166 changed both scheduler tests to
mock `torch.mps.profiler.metal_capture`, asserted context entry/exit, and checked
the injected failure message. The exact upstream diff is retained in
`raw/pr34166_profiler_test_diff.json`. The current checkout contains those
changes and additionally pins `profiler.use_mlx` to `False` for this MPS-specific
test class, preventing ambient `SGLANG_USE_MLX` from selecting the other
strategy.

The checked-in regression covers the reported success case plus independent
failure and cleanup boundaries. A platform-neutral harness was added to this
report so those current implementation boundaries can be executed on the
prepared host even though pytest correctly skips the Apple-Silicon-only class.

## Evidence

- `raw/pytest_mlx_profiler.txt`: the actual checked-in test file collected, but
  all 10 tests skipped because this host is not Apple Silicon and has no MLX.
- `raw/verify_current_mps_capture.txt`: the current `MetalCaptureProfiler.start_mps`
  succeeds with a mocked context, enters it once, stops through `__exit__`, and
  converts an independent injected `RuntimeError` into a failed result.
- `raw/pr34166_profiler_test_diff.json`: failing-before/passing-after regression
  change from the fixing upstream PR, showing the incorrect `mx.metal` mocks
  replaced by the MPS context mock.
- `raw/fix_commit.json`: fixing commit identity and parent.
- `raw/upstream_issue_comments.txt`: current issue discussion and Apple-Silicon
  verification already posted upstream; this investigation did not notify or
  modify upstream.
- `raw/host_capabilities.txt`: Linux x86_64, ROCm PyTorch, one AMD Instinct
  MI355X, and no `mlx` module.

## Limitations

This host cannot execute Apple Metal or the MLX runtime, so it cannot independently
repeat the full Apple-Silicon run. The available AMD GPU is irrelevant to the
reported MPS/Metal path and was not used. The retained mocked-path evidence
validates dispatch/capture lifecycle behavior only, not a real Metal trace.
No native library was changed or rebuilt.
