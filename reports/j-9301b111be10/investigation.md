# Investigation notes

Upstream issue: https://github.com/sgl-project/sglang/issues/31890

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2184

The recorded base still omitted four `BatchEmbeddingOutput` fields when routing a
batch item to its tokenizer worker. The scheduler populates `retraction_counts`,
while `TokenizerManager` indexes it unconditionally, confirming the reported
crash path from the checked-out implementation.

Two open upstream proposals were inspected before implementation:

- https://github.com/sgl-project/sglang/pull/29203
- https://github.com/sgl-project/sglang/pull/31892

Both carry the four reported fields. Their proposed pooled-hidden-state
extraction indexes the outer list directly. That works for non-stacked states,
but the current scheduler intentionally sends same-shaped states as
`[tensor(batch, ...)]`; request indices above zero would therefore index past
the one-element wrapper. This patch handles that documented representation by
selecting the request row from the stacked tensor, and retains ordinary list
partitioning for non-stacked/mixed states.

Raw failing-before, passing-after, and GPU numerical evidence is retained in
`reports/j-9301b111be10/raw/`.
