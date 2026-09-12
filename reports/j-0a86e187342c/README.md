# Investigation: sglang#35415

The prepared base already contains the reported fix. Upstream PR
[`sgl-project/sglang#33875`](https://github.com/sgl-project/sglang/pull/33875),
merged as `914644e81c9b6fc31d60a1ab5327a4884514c074` on 2026-08-07, explicitly
identifies MiniMax-H3 Turbo's fused 2-D `lora_B` merge crash and adds
per-logical-section tensor-parallel slicing. The source issue was opened on
2026-08-19.

The current implementation in
`python/sglang/multimodal_gen/runtime/layers/lora/linear.py` preserves the old
3-D stacked-adapter path and handles a fused 2-D matrix by slicing every
logical output section at the current TP rank before concatenation. The
existing `test_composed_pair_shards_correctly_under_mock_tp` regression in
`python/sglang/multimodal_gen/test/unit/test_fused_lora_compose.py` compares a
two-rank fused Q/K/V result against an independent unfused numerical reference.

`reproduce_lora_sharding.py` additionally records the old implementation's
exact `IndexError` on a 2-D tensor, checks both TP rank boundaries for unequal
Q/K/V sections, checks exact GPU values, and verifies that both TP boundaries
of the pre-existing 3-D path remain unchanged.

No production source correction was made because duplicating the already
merged fix would not be justified. Full MiniMax-H3 serving was not attempted:
the checkpoint and Turbo LoRA weights are unavailable, the assigned host has
one AMD gfx950 GPU rather than the report's two NVIDIA sm_120 GPUs, and a tiny
Llama serving fixture cannot qualify this diffusion architecture or its
multi-GPU TP path.

Raw outputs and upstream metadata are retained under `raw/`.
