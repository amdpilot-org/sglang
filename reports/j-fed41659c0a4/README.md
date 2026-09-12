# DSA EAGLE context-boundary investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/30570

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2373

## Finding

The reported defect is present in the v0.5.14 source but already corrected in
the prepared `main` checkout. In v0.5.14,
`DeepseekSparseAttnBackend.init_cuda_graph_state` allocated the CUDA-graph page
table with `model_config.context_len + speculative_num_draft_tokens` columns.
For the reported settings that is `614400 + 6 = 614406`, while EAGLE's
`req_to_token` row includes additional speculative reserve and reached 614408
columns. `_apply_cuda_graph_metadata` then attempted to copy the 614408-column
source into the 614406-column graph buffer.

Current source allocates the graph buffer from `self.req_to_token.shape[1]` and
`_graph_page_table_width` uses that allocated width for the replay source and
destination. The focused regression uses the actual current allocator and
width helper, performs the target-verify-shaped GPU copy at 614408 columns, and
checks exact equality. It also covers no-reserve, exact legacy-width, and
reserve-beyond-legacy-width boundaries.

The relevant upstream change is https://github.com/sgl-project/sglang/pull/30274,
merged on 2026-07-07 and included starting in v0.5.15. No production source
change is justified on the prepared checkout; this PR adds missing regression
coverage and records the investigation.

## Reproduction evidence

The v0.5.14 file was retrieved without modifying upstream from:

`https://github.com/sgl-project/sglang/blob/v0.5.14/python/sglang/srt/layers/attention/dsa_backend.py`

Its allocation uses `self.max_context_len + speculative_num_draft_tokens`, and
its target-verify replay derives `max_seqlen_k` from sequence length plus draft
tokens before copying `req_to_token[..., :max_seqlen_k]` into that buffer.

Replaying the historical tensor shapes on the assigned GPU:

```text
RuntimeError: The size of tensor a (614406) must match the size of tensor b (614408) at non-singleton dimension 1
```

The passing test output is retained in `pytest.xml`.

## Limitations

The assigned device is one AMD Instinct MI350X (`gfx950`) under ROCm 7.2. The
reported environment requires eight Blackwell GPUs, CUDA graphs, TP=8, and
ZhipuAI/GLM-5.2-FP8 weights, none of which were available. Therefore this is a
GPU tensor-shape and current-code invariant verification, not a full model,
semantic-accuracy, CUDA-graph, or distributed reproduction. No native library
was changed or rebuilt.
