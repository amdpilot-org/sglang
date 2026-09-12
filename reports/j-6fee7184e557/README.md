# Kimi-VL DP-attention reduction investigation

The prepared base still constructed two MoonViT row-parallel projections without
`use_dp_attention_reduce`: the MLP `fc1` projection and the vision attention
output projection. Both are sharded with `attn_tp_size`, so under DP attention
their reduction communicator must be the corresponding attention TP group.

The regression in `test_kimi_vl.py` constructs the actual encoder layer with
`tp_size=4` and `attn_tp_size=2`. Before the source correction, its DP-enabled
case observed `use_dp_attention_reduce=False`; after the correction both affected
projections observe `True`. The independent DP-disabled boundary case continues
to observe `False`.

An upstream search found open PR https://github.com/sgl-project/sglang/pull/35831,
which independently proposes the same two-call-site correction. It was open and
review-required at inspection time, so the prepared main checkout did not already
contain the solution.

The assigned machine exposes one AMD Instinct MI355X gfx950 GPU. The original
TP4/DP2 HTTP-serving deadlock and Kimi-VL model execution remain unverified because
four GPUs and model weights were not available. Raw test and inventory output is
retained under `raw/`.
