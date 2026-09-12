# Independent review of amdpilot-org/sglang PR 2763

- Upstream issue: https://github.com/sgl-project/sglang/issues/35332
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2701
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2797
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Exact candidate: `f9d58a04741052affdf8b9d5d90d7538b40eb0e4`
- Recommendation: **request changes**
- Fully resolves original issue: **false**

## Finding

The candidate adds a useful Python-only controller skeleton and its focused tests pass, but it does not safely preserve the last real prediction across repeated scheduler consumption. `after_real_forward()` stores a detached clone, while `reused_prediction()` returns that stored tensor itself. A scheduler or other consumer that performs an in-place operation therefore mutates the cache. The next skipped step reuses the modified value instead of the prediction produced by the last real forward.

The independent adversarial test reproduced this on CPU and on the assigned AMD Instinct MI355X. On GPU, the real prediction was `3.25`; after the first reused value was multiplied in place by its consumer, the following reuse returned `6.5` and shared the same storage. This contradicts the stated contract and the implementation comment that the reusable observation is independently owned for the scope.

The candidate also remains a framework-only partial implementation: no architecture adapter is enabled, no real stateful diffusion model validates cache-off/on quality or persistent session/KV equality, multi-rank agreement was not exercised on real distributed hardware, and model-specific denoising overrides do not all use the shared reuse path. Those limitations are disclosed by the candidate and mean it cannot be classified as a full original-issue fix.

## Reproduction and validation

The prepared checkout was exactly the recorded base. Importing `sglang.multimodal_gen.runtime.pipelines_core.step_reuse` there failed with `ModuleNotFoundError`, confirming that the requested framework contract was absent before the candidate.

At the exact candidate SHA, imports resolved to `/job/repo/python/sglang/...`, confirming the checked-out source was exercised. The candidate's focused suite passed: 16 tests. Compatibility tests passed: 68 tests and 30 subtests.

Independent adversarial pseudocode:

```python
controller.after_real_forward(torch.tensor([3.25], device="cuda"), 0)
first_reuse = controller.reused_prediction()
first_reuse.mul_(2)  # scheduler/consumer mutates model_output
later_reuse = controller.reused_prediction()
assert later_reuse.item() == 3.25  # fails: 6.5
```

No C++, HIP, CUDA, FlyDSL, or other native source changed. `repository-environment.json` records `native: null`, so a native rebuild was not applicable. The environment used Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and one AMD Instinct MI355X. No model weights were available; the tiny Llama fixture is not capable of qualifying diffusion semantics, stateful world-model behavior, or output quality.

