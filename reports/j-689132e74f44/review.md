# Independent review of amdpilot-org/sglang PR 1266

Reviewed exact candidate commit `2f3dfee039131afc685e53b42d34d2b8fb412b45` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

## Recommendation

Accept. The candidate is a source fix, not test-only hardening. It removes compile-time ABI selection from the dynamically resolved `cudaMemcpyBatchAsync` call and selects the CUDA 12.8 nine-argument or CUDA 13 eight-argument ABI using `cudaRuntimeGetVersion()` from the loaded runtime.

The base behavior was reproduced deterministically with a native host harness: compiling the base-equivalent call as CUDA 12.8 and invoking a mocked CUDA 13 entry point causes the callee to receive the `failIdx` pointer as its stream argument. This demonstrates the reported ABI displacement without claiming the unavailable NVIDIA SIGSEGV reproduction.

The candidate regression passed. An independent adversarial native harness also passed with `CUDA_VERSION=13000` while exercising both a 12.8 runtime ABI and a 13.0 runtime ABI, covering the mirror-image mismatch omitted from the submitted test.

## Scope and limitations

The prepared interpreter imports SGLang from `/job/repo/python/sglang` and Torch from `/opt/venv/lib/python3.12/site-packages/torch`. The available device is one AMD Instinct MI355X, gfx950, with Torch 2.11.0+rocm7.2. There is no CUDA compiler/runtime or NVIDIA GPU, so the original mixed nvcc 12.8/libcudart 13 configuration, actual CUDA symbol resolution, NVIDIA ISA, the 27B AWQ model, and TP2 serving were not executable here.

A fresh private-cache rebuild compiled the staged HiCache JIT module and 12 GPU round-trip tests passed on gfx950. That validates the unaffected ROCm path only; it is not used as proof of the CUDA ABI fix. No AOT or FlyDSL native source changed.

Raw review evidence is preserved outside the checkout at `/job/review-evidence-j-689132e74f44/`.
