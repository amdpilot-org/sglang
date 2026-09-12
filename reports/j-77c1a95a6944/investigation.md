# Qwen3.8 gateway reasoning parser investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35148

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1431

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared checkout pinned `reasoning-parser = 1.0.0`. Its Qwen3 implementation explicitly used `initial_in_reasoning: false`, while the gateway response processors did not initialize parser state from the tokenizer's thinking-prefill metadata. An issue-shaped regression using `Compute 17 * 23. </think>\n\n391` therefore failed before the fix: `reasoning_text` was empty and the whole string remained normal content.

The correction upgrades the parser dependency to 1.3.0, resets and initializes the non-streaming pooled parser for each request, and initializes each streaming choice parser from the tokenizer's `ThinkingToggle`, `ThinkingKeyName`, and `think_in_prefill` values. Explicit request overrides take precedence over template defaults. Boundary tests cover default-on disabled explicitly, default-off enabled explicitly, templates with no recognized toggle, and selection between `enable_thinking` and `thinking` keys.

Raw evidence is retained outside the worktree in `/tmp/amdpilot-repo-j-77c1a95a6944/`: `failing-before.log`, `passing-after-prefill.log`, `cargo-test-lib.log`, `cargo-fmt.log`, and the reviewed candidate patches `pr35249.patch` and `pr35346.patch`.

No GPU or full model was used. This verifies gateway parsing for the reported generated-output shape, not Qwen3.8-27B-FP8 inference quality or an end-to-end deployment on the reporter's NVIDIA hardware.
