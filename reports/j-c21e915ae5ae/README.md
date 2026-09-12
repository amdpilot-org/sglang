# DSPARK probability recovery correction

This correction investigates candidate PR https://github.com/amdpilot-org/sglang/pull/1465 at exact commit `1dd7ea6afbbe9cdac3c31585526b779869bb9e3b` and its independent review PR https://github.com/amdpilot-org/sglang/pull/1545 against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

The exact candidate test file passed on the base (`3 passed`), so it did not provide a failing-before/passing-after regression. A new boundary regression then reproduced all four review counterexamples on the base: NaN outside column 0, positive infinity, a negative value, and a zero-sum row were left unchanged by `_one_hot_token0`; all four subtests failed before the correction.

The production correction validates each complete probability row against `torch.multinomial`'s relevant preconditions: every element must be finite and nonnegative, and the finite row sum must be positive. Any invalid row uses the existing token-0 one-hot fallback. Valid rows remain unchanged.

After the correction, the focused suite passed on CPU and the assigned gfx950 (`4 passed, 8 subtests passed`). A separate gfx950 probe recovered all four malformed rows, preserved a valid row, synchronized successfully after `torch.multinomial`, and is retained in `raw/gpu_probability_boundary_after.txt`.

## Limitations

DeepSeek-V4-Flash-0731 weights and the original eight-H100 CUDA environment were unavailable. The rare producer of invalid probabilities in that TP=8 workload remains unidentified. This change establishes containment at the concrete eager DSPARK multinomial boundary; it does not claim a full model, CUDA, TP=8, fast native-kernel, or week-long production reproduction. No native source changed, so no native rebuild was applicable.
