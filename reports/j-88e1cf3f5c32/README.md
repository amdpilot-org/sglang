# Investigation result

The prepared `main` snapshot already contains the narrow correction for the
reported Qwen3.5/3.6 hybrid GDN + MTP sizing defect. In
`ModelConfig._config_draft_model`, the Qwen draft rewrite sets
`num_nextn_predict_layers = 1` on both the outer wrapper and
`hf_text_config`. Later shape derivation reads the nested value, so the EAGLE
pool configurator prices one draft layer rather than falling back to all 64
target layers.

For the reported geometry (16 full-attention layers, four KV heads, 256-byte
head dimensions for K and V, and one-byte FP8 elements), current sizing is:

`16 * 4 * (256 + 256) + 1 * 4 * (256 + 256) = 34,816 bytes/token`.

The broken fallback prices 64 draft layers:

`16 * 2,048 * (1 + 64 / 16) = 163,840 bytes/token`, exactly 5x the target
cell and matching the issue analysis.

To establish failing-before evidence, the existing nested assignment was
temporarily removed. The new wrapper regression then failed because
`hf_text_config.num_nextn_predict_layers` was absent. The assignment was
restored, and the complete focused suite passed. See
`pytest-failing-before.log` and `pytest-focused.log`.

No production source was changed. The added tests cover the reported wrapper,
the causal-LM sibling, the non-draft boundary, and the exact pool arithmetic.

Hardware inventory is retained in `gpu-inventory.log`. The assigned GPU is an
AMD Instinct MI350X (gfx950), not the reported NVIDIA RTX PRO 6000 Blackwell,
and the Qwen3.6-27B-NVFP4 weights were unavailable. Consequently this is not a
full-model or NVIDIA runtime reproduction.
