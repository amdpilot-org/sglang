# Independent review of PR 2102

Upstream issue: https://github.com/sgl-project/sglang/issues/32572

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2023

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2142

Candidate: https://github.com/amdpilot-org/sglang/pull/2102 at exact commit `6f9ba912ed5a5a9ff2434f7505e86326bff8c20d`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365` (the prepared checkout matched it exactly).

## Recommendation

Request changes. The candidate is a substantive, passing partial fix for the reported Qwen3-VL adapter path, but it does not fully resolve the original report's explicit MOSS-VL scope. It also adds an unbounded `deepstack_merger_list` regex alternative which accepts names such as `notdeepstack_merger_list.4` as layer 4.

## Evidence

On the recorded base, an issue-specific fixture reproduced the failure class and preceding loss of adapter weights:

- `get_hidden_dim("linear_fc2", ...)` raised `NotImplementedError: get_hidden_dim not implemented for linear_fc2`.
- `get_layer_id("...visual.deepstack_merger_list.2.linear_fc2.lora_A.weight")` returned `None`, so those tensors cannot enter the indexed adapter layers.
- The prepared interpreter imported `sglang` and `sglang.srt.lora.utils` from `/job/repo/python`, not from an unrelated installed package.

At the exact candidate commit:

- The candidate's Qwen3-VL regression plus the existing Laguna hidden-dimension suite passed: 17 tests.
- Independent checks confirmed Qwen3-VL merger dimensions `(4608, 4608)` and `(4608, 4096)`, `linear_fc2` row-parallel classification, target normalization, indices 0-2, rejection of unindexed/malformed deep-stack names, and preservation of ordinary `model.layers.N` parsing.
- The candidate's BF16 rank-32 GPU arithmetic fixture ran on the assigned AMD Instinct MI350X (`gfx950`) for both reported tensor shapes and compared against CPU FP32 references. This validates arithmetic at those shapes, not a full model load or semantic output.
- No native source changed. `repository-environment.json` declares no native rebuild target, so no native rebuild was applicable.

## Remaining counterexamples and limitations

1. MOSS-VL remains unsupported by this patch. `MossVLForConditionalGeneration` has no `get_hidden_dim` override. Its merger uses `input_hidden_size = hidden_size * spatial_merge_size**2 * (1 + num_deepstack_features)`, which is architecture-specific and differs from the Qwen3-VL override added by the candidate. Admitting `linear_fc1/2` globally therefore does not supply the MOSS-specific buffer dimensions and the fallback still fails for these names.
2. `get_layer_id("model.notdeepstack_merger_list.4.linear_fc2.weight")` returns `4`. The new regex should require a path-segment boundary. The older `layers` alternative has the analogous pre-existing behavior (`otherlayers.9` returns 9), but the candidate newly extends it to the deep-stack namespace.
3. The full Qwen3-VL-8B and EditScore adapter weights were not available locally, so no full server/model-load reproduction was performed. The deterministic tiny Llama fixture would not qualify Qwen3-VL or MOSS-VL architecture behavior and was therefore not substituted.
4. The available GPU is AMD gfx950/ROCm 7.2, whereas the report used NVIDIA H100/CUDA. The Python control-path evidence is architecture-independent; backend parity and full serving remain unverified.

Raw logs are retained in `reports/j-693ea22a5021/raw/`.
