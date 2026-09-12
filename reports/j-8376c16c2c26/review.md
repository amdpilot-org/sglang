# Independent review of PR 3370

Reviewed exact candidate commit `e3b9d9055cb4d5cf348e3cc28313ba8fea266beb` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue https://github.com/sgl-project/sglang/issues/32807.

The base reproduces the source-level contract failures under controlled kernel spies: rank-3 input bypasses `gemma_rmsnorm`, while BF16 activations with FP32 weights enter both fused normal and residual calls. The candidate's five focused regressions fail 4/5 on the base and pass 5/5 at the exact candidate commit.

Independent adversarial checks on the assigned AMD Instinct MI350X covered rank-1, rank-3, rank-4, zero-sized leading dimensions, and a weight/device mismatch. Flattened paths preserved shape and matched the native reference exactly in the spy-backed GPU checks, including empty tensors. The device mismatch did not enter the fused call and then raised from the native operation, as expected for tensors on incompatible devices.

The change is Python-only, so no native rebuild was required. Imports resolve `sglang` and `layernorm.py` from `/job/repo/python`, but `sgl_kernel` resolves from the prepared ROCm environment under `/opt/venv`. This ROCm host cannot execute the issue's CUDA-only Gemma kernel. Consequently, the original NVIDIA mixed-dtype NaN, real fused CUDA numerics, and H200 performance improvement were not independently verified. The recommendation is therefore `unverified`, not a rejection based on a found code counterexample.

No candidate source was modified or duplicated. The review report was committed from the prepared `amdpilot/j-8376c16c2c26` branch.
