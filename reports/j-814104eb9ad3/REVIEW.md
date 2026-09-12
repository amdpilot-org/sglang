# Independent review of PR 2130 at `8b5ab59b905622ea414746c75e1f70dc4065c928`

Upstream issue: https://github.com/sgl-project/sglang/issues/32204

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2109

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2165

## Recommendation

Request changes. The candidate is a useful partial fix for the ordinary Qwen3.5/Qwen3-VL path, but it does not fully resolve the original issue and has an uncovered Qwen3.5 counterexample.

## Evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, configuring a 32-layer `Qwen3_5ForCausalLM` with shifted capture positions `[4, 12, 20, 28, 32]` raises `IndexError: index 32 is out of range` in `set_dflash_layers_to_capture`.
- At the exact candidate commit, its focused test passes: 5 tests passed.
- The candidate accepts terminal position 32 and its extracted terminal-capture arithmetic exactly matches an independent CPU reference on the assigned AMD Instinct MI350X (`gfx950`).
- The original report also identifies Qwen2/Qwen3's silent terminal-drop variant. Minimal real `Qwen2Model.forward` and inherited `Qwen3Model.forward` executions with three layers and `layers_to_capture=[3]` return only the final tensor, not an auxiliary-state tuple. Those files are unchanged by the candidate.
- Qwen3.5 has an existing FlashInfer MNNVL deferred-finalization path where `hidden_states` is `Qwen35MoeFinalizeHandoff` after the decoder loop. The candidate calls `_capture_final_decoder_output` before that handoff is finalized; with a non-null residual this raises `TypeError` from `Qwen35MoeFinalizeHandoff + Tensor`. Thus terminal capture is not valid across the model's supported forward branches.
- Calling `set_dflash_layers_to_capture([1])` and then `[2]` leaves `_is_layer_to_capture` set on both layers, because old flags are not cleared. This is independent hardening rather than the central original failure, but demonstrates missing state-reset coverage.
- The candidate PR says `git diff --check` passed, but the exact candidate diff reports trailing whitespace in `reports/j-ba3a9a857a8f/raw/passing_after.log` and exits 2.

## Scope and limitations

The prepared interpreter imports SGLang from `/job/repo/python/sglang` and Torch 2.11.0+rocm7.2. No native C++/FlyDSL source is changed by the candidate, so no native rebuild applies. One assigned AMD Instinct MI350X GPU was available. Qwen3.5-4B weights and Ascend hardware were unavailable, so this review does not claim a full server/model or Ascend reproduction. The deterministic structural reproducers exercise the exact setup and forward contracts involved; the GPU check only validates the terminal residual-stream arithmetic.
