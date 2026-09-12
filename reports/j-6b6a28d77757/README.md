# Investigation evidence

The source at base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`
still used full-width `Vectorized::loadu` and `store` operations in both sigmoid
overloads. The dispatch explicitly accepts 1, 2, 4, and 8 experts, so those
operations crossed the input and stack-buffer boundaries.

The original one-expert float32 reproducer was run against an AddressSanitizer
build of the checkout's CPU extension. It failed with a 64-byte read from a
4-byte allocation in `sigmoid<float, 1>` at `topk.cpp:181`. After the change,
the same ASan native library was rebuilt and 32 combinations passed: float32
and bfloat16 inputs, 1/2/4/8 experts, with and without renormalization. Results
were checked against independent PyTorch sigmoid, top-k, gather, and sum
normalization operations.

Native build and raw execution logs are retained outside the worktree at:

- `/tmp/amdpilot-repo-j-6b6a28d77757/cpu-asan-before`
- `/tmp/amdpilot-repo-j-6b6a28d77757/cpu-release`
- `/tmp/amdpilot-repo-j-6b6a28d77757/evidence/asan-before-repro.log`
- `/tmp/amdpilot-repo-j-6b6a28d77757/evidence/asan-after-small-cases.log`
- `/tmp/amdpilot-repo-j-6b6a28d77757/evidence/regression-test-normal.log`
- `/tmp/amdpilot-repo-j-6b6a28d77757/evidence/existing-topk-tests.log`

This is a CPU-kernel defect. No GPU execution was used or claimed, and no full
model or serving-path behavior was inferred from the operator fixture.
