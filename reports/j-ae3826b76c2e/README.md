# Investigation report: sglang#32924

Upstream issue: https://github.com/sgl-project/sglang/issues/32924

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1977

## Finding

The reported asynchronous launch failure was not reproduced. The prepared
environment has one AMD MI350X (`gfx950`) GPU, whereas the report requires the
CUDA-only Kimi-K3 fused route/pack/MXFP8-quant kernel on 8 NVIDIA B300 GPUs,
TP8/DCP8, the Kimi-K3 MXFP4 weights, CUDA graphs, DSPARK speculative decode,
and a long-context traffic soak. No source issue comment or later related
change identifies a root cause. In particular, PR #33764 corrects router GEMM
accuracy and is not evidence for an illegal-access fix.

Current main still had the fusion from PR #32699 and no independent way to
disable it without also changing the FlashInfer MXFP4 precision. This change
adds `SGLANG_DISABLE_KIMI_K3_ROUTE_QUANT_FUSION=1`. The check occurs before
coverage or JIT availability is probed, so the suspect combined kernel is not
compiled or launched. The existing three-operation fallback (route, packed
top-k IDs, and MXFP8 activation quantization) remains selected with the same
runner and precision.

This is a diagnostic/mitigation candidate, not a claimed fix for the unknown
cause of the B300 launch failure.

## Evidence

- `raw/failing-before.txt`: the three regression cases fail before the source
  change because the dedicated switch does not exist.
- `raw/unit-after.txt`: 8 dispatch and fallback tests pass after the change.
- `raw/environ-and-handoff.txt`: 16 environment/handoff tests plus 2 subtests
  pass.
- `raw/gpu-environment.txt` and `raw/rocm-agents.txt`: actual assigned GPU and
  portable HIP execution evidence; they also document the architecture gap.
- `raw/ruff.txt`: Ruff could not run because it is absent from the prepared
  interpreter (`No module named ruff`).

## Remaining limitations

- No NVIDIA B300/CUDA hardware, Kimi-K3 weights, eight-GPU topology, or
  production traffic was available.
- The original asynchronous failure was not reproduced or root-caused.
- The new toggle's control flow is regression-tested, but the CUDA fused and
  standalone kernels cannot be executed on gfx950.
- No native library was changed or rebuilt.
