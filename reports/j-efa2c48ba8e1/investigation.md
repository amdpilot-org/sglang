# Qwen3-VL `linear_fc` LoRA investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32572

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2023

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The reported adapter configuration was fetched without downloading model
weights. Its target list includes `0.linear_fc1`, `0.linear_fc2` through index
2. The safetensors header shows that these names belong exclusively to
`model.visual.deepstack_merger_list.{0,1,2}` and have these shapes:

- `linear_fc1`: A `[32, 4608]`, B `[4608, 32]`
- `linear_fc2`: A `[32, 4608]`, B `[4096, 32]`

At the base commit, `get_hidden_dim("linear_fc2", ...)` cannot allocate the
buffer, reproducing the reported failure class. A second defect appears after
that point: `get_layer_id` recognizes only `layers.N`, so every indexed
`deepstack_merger_list.N` adapter tensor is dropped and its base module is not
wrapped.

The open upstream PR https://github.com/sgl-project/sglang/pull/32573 was
reviewed before implementation. It proposes `intermediate_size -> hidden_size`
for both projections. For the reported Qwen3-VL configuration those are text
dimensions `12288 -> 4096`, which conflict with the adapter's measured vision
shapes above. This change instead derives the merger width from
`vision_config.hidden_size * spatial_merge_size**2`, uses
`vision_config.out_hidden_size` for `linear_fc2`, recognizes only explicitly
indexed deep-stack merger layers, and applies row-parallel sharding to
`linear_fc2`.

Raw evidence is retained in `reports/j-efa2c48ba8e1/raw/`. Downloaded issue,
PR, model-config, adapter-config, and safetensors-header evidence remains in
`/tmp/amdpilot-repo-j-efa2c48ba8e1/`; no model weights were downloaded.
