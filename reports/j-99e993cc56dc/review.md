# Independent review of amdpilot-org/sglang PR 1213

Candidate reviewed: `9c8a73c0c3ad979a58fdf7b2be486435f5862e39`

Upstream issue: https://github.com/sgl-project/sglang/issues/36528

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1246

Recommendation: **accept**. The candidate fully resolves the original startup-contract bug for ordinary Qwen/Llama targets: EAGLE3 without a separate draft is rejected during argument resolution, before scheduler creation or duplicate target-weight loading. Existing explicit-draft and bundled-draft routes remain accepted.

## Evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a focused configuration probe accepted ordinary `LlamaForCausalLM` plus EAGLE3 with no draft (`NO_ERROR draft=None`).
- On that base, the issue's dummy-weight Llama launch ran on the assigned AMD Instinct MI350X (`gfx950`), loaded `LlamaForCausalLM` twice, and crashed at `draft_runner.model.hot_token_id` with `AttributeError`; the launcher exited 137.
- At the exact candidate, the same Llama launch exited 1 during argument resolution with the requested clear `ValueError`, before scheduler creation or weight loading.
- At the exact candidate, the public Qwen2.5 launch also exited 1 at the same early validation point.
- The candidate's focused speculative suites passed: 36 tests.
- Independent adversarial probes passed for lowercase `eagle3`, explicit draft paths, bundled `DeepseekV3ForCausalLM`, bundled `Qwen4ExpForConditionalGeneration`, unchanged EAGLE without a draft, and no speculative algorithm.

## Scope and limitations

No native/C++ source changed, `repository-environment.json` records no native build target, and no native rebuild was applicable. Python imported from `/job/repo/python/sglang`; AITER loaded from `/tmp/amdpilot-repo-j-99e993cc56dc/cache/aiter/module_aiter_core.so`; the interpreter was `/tmp/amdpilot-repo-j-99e993cc56dc/venv/bin/python` with Torch `2.11.0+rocm7.2`.

The assigned device was one AMD Instinct MI350X, gfx950, with ROCm 7.2. The base failure used actual GPU initialization and dummy Llama weights. The corrected path intentionally exits before GPU worker initialization, so this review does not claim EAGLE3 numerical correctness, full target/draft inference, semantic accuracy, multi-GPU, or multi-node coverage. Qwen weights were not loaded because early rejection is the behavior under review.

The candidate also commits raw pytest logs containing trailing whitespace, so a whole-commit `git diff --check` reports those evidence-file lines. The source and test patch itself has no demonstrated functional defect, and this does not leave an original-issue counterexample.
