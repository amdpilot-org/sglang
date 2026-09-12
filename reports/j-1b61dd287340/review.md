# Independent review of amdpilot-org/sglang PR 1204

Candidate reviewed: `83e7f0d2ee8f2d71620185f81c8cba6bc84960ad`

Upstream issue: https://github.com/sgl-project/sglang/issues/36533

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1238

## Recommendation

Accept, with the qualification that this environment cannot establish a full end-to-end reproduction on CUDA 13. The implementation is narrowly aligned with the reported venv layouts and passed an independent dynamic-loader subprocess test. No counterexample was found within the original CUDA 13 pip/venv contract.

`fully_resolves_original` is recorded as false because the assigned system is ROCm/gfx950, the prepared interpreter has no `torch_memory_saver`, and Qwen weights were unavailable. Consequently, the actual CUDA 13 hook, SGLang scheduler startup, and model-serving path remain unverified here.

## Evidence

- The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image-prepared revision difference.
- On the base, a separately compiled preload object with `NEEDED libcudart.so.13` and no RPATH/RUNPATH caused the prepared Python executable to fail before startup with the reported loader error and exit code 127.
- The exact candidate imports from `/job/repo/python/sglang/srt/utils/torch_memory_saver_adapter.py`. Torch imports from `/opt/venv/lib/python3.12/site-packages/torch/__init__.py` and is `2.11.0+rocm7.2`; `torch.version.cuda` is `None`, `torch.version.hip` is `7.2.26015`, and `torch_memory_saver` is not installed.
- The candidate's focused regression passed: 3 tests.
- An independent test created both supported pip layouts and used the adapter context around an actual child exec. For `nvidia/cu13/lib`, the child printed `child-started` and exited 0; an existing library path was retained, an existing exact runtime entry was not duplicated, and the parent value was restored. The `nvidia/cuda_runtime/lib` layout was also discovered, and an initially absent variable was removed after the context.
- The candidate changes Python only. There are no native/C++ changes, so a native rebuild is not applicable.
- A tensor check ran on the assigned AMD Instinct MI350X (`gfx950`) and matched `[2.0, 5.0, 10.0]`. This confirms GPU access only and is not evidence for CUDA 13 or the serving fix.

## Scope and limitations

The review verifies the dynamic-loader mechanism and adapter environment contract, not a real CUDA 13 scheduler/model launch. It does not validate the NVIDIA driver, the actual `torch_memory_saver==0.0.9.post1` cu13 hook, Qwen architecture/weights, semantic model output, or distributed execution. A user-site-only CUDA runtime installation outside `site.getsitepackages()` is not discovered, but that is outside the issue's stated venv reproduction.
