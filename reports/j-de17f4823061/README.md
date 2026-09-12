# Independent review of amdpilot-org/sglang PR 891

Reviewed candidate commit `d2d2311e20870d102ed6f8cfbfd1328c8d0fe05d` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the contract in:

- Upstream issue: https://github.com/sgl-project/sglang/issues/37983
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/926
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/891

## Recommendation

**Accept.** The candidate is a source fix, not merely test hardening. It forwards `window_size` and `s_aux` from `VisionTritonAttention`, implements the corresponding local-window mask and value-less per-head sink denominator in the Triton prefill kernel, and preserves the old unbounded/no-sink behavior.

On the exact base, the backend regression demonstrated that changing the window or sinks produced identical output, while direct kernel calls rejected the new arguments. At the exact candidate commit, all 13 runnable candidate tests passed on the assigned GPU; the FA3 cross-check skipped because FA3 is unavailable on ROCm. Five independently selected numerical cases also passed against a separate PyTorch reference, covering sequence lengths around Triton block boundaries, GQA, head dimensions 32/64/80/128, asymmetric and zero-width windows, causal composition, BF16/FP32, and sink logits from approximately -81 through +81.

## Source and environment verification

- Prepared checkout initially matched the required base exactly and was clean.
- Candidate parent and merge base are both the required base commit.
- During candidate testing, imports resolved to `/job/repo/python/sglang/kernels/ops/attention/prefill_attention.py` and `/job/repo/python/sglang/srt/layers/attention/vision.py`, not an installed SGLang copy.
- Interpreter: `/tmp/amdpilot-repo-j-de17f4823061/venv/bin/python`.
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`.
- GPU execution was real on one AMD Instinct MI355X, `gfx950:sramecc+:xnack-`.
- No C++ or other separately built native-extension source changed. The modified kernel is Triton JIT code and compiled/executed during the GPU tests, so a native extension rebuild was not applicable.
- The checkout was returned to `amdpilot/j-de17f4823061` before this report was committed. The candidate itself was not modified or merged.

## Limitations

No MiMo weights were available, so this review does not claim a full model/server or semantic-quality reproduction. The available accelerator was AMD gfx950/ROCm, not the NVIDIA B40/H20 CUDA environments in the report, so CUDA compilation and NVIDIA-specific behavior remain unexecuted here. FA3 could not provide an on-device cross-backend oracle on ROCm. These limitations do not leave a known source-level counterexample: the independent PyTorch oracle directly checks the mathematical window-and-sink contract and the actual vision backend forwarding path.

Raw logs and the reviewed source diff are retained under `raw/`.
