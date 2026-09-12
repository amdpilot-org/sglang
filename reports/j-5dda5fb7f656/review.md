# Independent review of PR 1466

- Upstream issue: https://github.com/sgl-project/sglang/issues/35150
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1501
- Candidate: https://github.com/amdpilot-org/sglang/pull/1466
- Exact candidate commit: `5da4789fed6b1a27c129f69d73fff3ee051675f1`
- Recorded base and candidate parent: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept as a partial fix**
- Fully resolves original issue: **no**

## Finding

The candidate is a real, narrowly scoped correction to GDN `TARGET_VERIFY` beta precision. It makes the width-one verify transition use the BF16 beta materialization boundary used by packed decode. The candidate regression fails with the recorded base source and passes at the exact candidate. An independent 96-round randomized test with a nonzero initial FP32 recurrent state also found bitwise-identical outputs and states between packed decode and repeated width-one verify transitions on gfx950.

This does not fully resolve the original issue. The issue's supplied evidence explicitly says beta alignment only moves the first BF16-state model divergence from output index 13 to index 23, and that FP32 persistent state still diverges at index 38. The candidate changes only the already-identified beta contributor and does not address or reproduce the remaining accumulated model-level drift. Its PR prose correctly calls the change partial.

## Failing-before / passing-after evidence

The candidate test file was retained while the kernel source was restored exactly to the recorded base semantics. On the assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), the four-case regression produced two failures: `b=-20` differed in all 128 output elements with maximum absolute difference `9.094947017729282e-13`, and `b=-0.5` differed in all 128 output elements with maximum absolute difference `0.000244140625`. The process exit code was 1.

At exact candidate commit `5da4789fed6b1a27c129f69d73fff3ee051675f1`, the same regression passed all four cases. The full candidate GDN test file passed 17 tests. The non-contiguous GDN and KDA boundary suites passed 23 tests and 3 subtests.

The independent adversarial script is preserved outside the checkout at `/job/review-evidence-j-5dda5fb7f656/adversarial_repeated_transition.py`. It used randomized BF16 `q`, `k`, `v`, `a`, and `b`, a randomized nonzero FP32 state, and 96 sequential width-one transitions. Each verify intermediate state was committed as the following round's state. It reported:

```text
first_output_mismatch None
first_state_mismatch None
final_state_max_abs 0.0
```

Raw logs were preserved outside the revision-switched checkout under `/job/review-evidence-j-5dda5fb7f656/`.

## Source and build validation

The prepared interpreter was `/tmp/amdpilot-repo-j-5dda5fb7f656/venv/bin/python`. Imports resolved to the checkout, including:

```text
sglang=/job/repo/python/sglang/__init__.py
kernel_module=/job/repo/python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py
```

The change is Triton/Python source only. It changes no C++, FlyDSL compiler, extension, wheel, or other native source, so no native rebuild was applicable. A private Triton cache under `/tmp/amdpilot-repo-j-5dda5fb7f656/cache/triton` was used, ensuring candidate kernels were compiled from the checked-out source rather than taken from an unrelated wheel.

## Limitations and remaining counterexamples

The assigned GPU is AMD gfx950 with ROCm 7.2 and PyTorch `2.11.0+rocm7.2`, not the issue's NVIDIA RTX 5090 / CUDA 13 environment. The exact Qwen3.8-27B-NVFP4 target and Qwen3.8-27B-DSpark weights were unavailable. Therefore the original 96-token serving failure could not be independently reproduced here, and no full-model, NVFP4, CUDA, or semantic-accuracy claim is made.

The remaining counterexample is the original reporter's beta-aligned forced-rejection run: repeated GDN `TARGET_VERIFY` still diverges from Base at output index 23 with BF16 persistent state, and at index 38 with FP32 persistent state. A fresh prefill of the same prefix restores the Base continuation. The candidate's local transition parity and synthetic repeated kernel test do not eliminate or explain that model-level accumulated-state counterexample.

