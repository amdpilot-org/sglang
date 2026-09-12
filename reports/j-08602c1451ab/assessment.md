# KVPress integration assessment

Upstream issue: https://github.com/sgl-project/sglang/issues/10585

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2925

## Result

Candidate rejected; the requested end-to-end feature is not implemented in this
change. The prepared base contains neither KVPress configuration nor the
`kvpress` dependency. Two earlier upstream attempts were inspected:

- sgl-project/sglang#11829 attempted serving integration but used one shared
  request token map for independently selected per-layer token sets. That makes
  at least one layer read the wrong slots whenever layer selections differ. It
  also required radix caching and CUDA graphs to be disabled and did not cover
  chunked prefill, speculative decoding, paged allocation, quantized caches,
  transfer backends, or distributed execution.
- sgl-project/sglang#12933 only copied a SnapKV press implementation and a unit
  test; it explicitly had no serving effect and was closed.

The current SGLang cache contract has one `ReqToTokenPool.req_to_token` row per
request, shared by all attention layers. Decode allocation and attention
metadata use the logical sequence length to index that row. Generic KVPress
presses may return a different, shorter sequence for every layer and sometimes
every head. Consequently, a correct integration needs a new layer-aware cache
index/length abstraction (and corresponding backend kernel metadata), or it
must explicitly restrict the contract to one global token selection shared by
all layers and heads. The latter is not the generic `BasePress.compress`
contract requested by the issue.

The GPU reproducer in `test_shared_mapping_incompatibility.py` independently
computes attention for two different layer selections and demonstrates the
incorrect result obtained when the last layer's selection is published through
the shared request map. This is the central correctness failure in the prior
serving MVP, not merely a missing optimization.

## Unimplemented and unverified scope

- No one-node KVPress serving path is claimed.
- No supported press registry or external KVPress API compatibility is added.
- Llama, Mistral, Phi, Qwen 2/3, and Gemma 3 model behavior is unverified.
- Radix/prompt caching, chunked prefill, CUDA graphs, speculative decoding,
  sliding-window/hybrid models, quantized KV caches, and prefix sharing remain
  unsupported for this proposal.
- Tensor, pipeline, and data parallelism are unverified.
- PD disaggregation and KV transfer backends are unverified.
- The available deterministic tiny Llama fixture can validate transport and
  engine execution only; it cannot resolve the cache-indexing contract or
  qualify semantic accuracy, so it was not used as evidence of completion.
- No native source was changed, so no native rebuild applies.

## Required design work

Before implementation, the feature needs an accepted contract for layer/head
specific physical lengths and indices, allocation/free semantics after
compression, prefix-cache identity after token removal, and decode position
handling. Every attention backend and transfer path that consumes the shared
mapping must either support that metadata or reject KVPress at configuration
time. Only then can individual presses be integrated and checked against
independent dense-attention references and end-to-end model quality tests.
