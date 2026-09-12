# Investigation result for sglang#38022

Upstream issue: https://github.com/sgl-project/sglang/issues/38022

Mirror issue: https://github.com/amdpilot-org/sglang/issues/823

The prepared base already contains the root-cause correction from upstream PR
https://github.com/sgl-project/sglang/pull/37971 (merge commit
`e980c1a2f133a8f7936a88cf6bd572d0fffde506`). The production reporter linked
the crashes to mixed image/video GLM requests. GLM uses the same post-tokenizer
ID for image spans and video-frame spans. Before PR #37971, the generic lookup
assigned every such span to both modalities, so chunked prefill could expect
more image tokens than the image embedding contained. The current
`Glm4vImageProcessor.get_mm_item_offsets` partitions the shared-ID spans using
the begin/end-video boundary tokens.

## Evidence

`reproduce_mixed_offsets.py` runs the legacy generic lookup and the current
implementation on the same mixed-media token stream. On the assigned gfx950 it
observed:

```text
legacy image offsets: [(1, 2), (4, 6), (10, 11), (13, 15)]
legacy video offsets: [(1, 2), (4, 6), (10, 11), (13, 15)]
current image offsets: [(1, 2), (4, 6)]
current video offsets: [(10, 11), (13, 15)]
```

This is failing-before/passing-after evidence for the issue-specific mixed
image/video trigger. The registered regression independently covers multiple
images followed by video, interleaved image/video content, and the boundary
case where image and video token IDs are distinct. The focused test run passed
all 11 tests, including the existing short/long embedding behavior tests.

Raw logs are retained in `raw/focused_pytest.log` and
`raw/gpu_mixed_offsets.log`.

## Related change review

Upstream PR https://github.com/sgl-project/sglang/pull/38155 remains open and
proposes zero-padding missing embeddings. That keeps the scheduler alive but
substitutes fabricated vectors into a model forward pass and provides no
semantic-accuracy evidence. This investigation did not adopt it.

## Limitations

GLM-5.3-Flash weights and the reported 8xH20, TP8/DP8/EP8 configuration were
not available. Therefore this is not a full-model, HTTP-serving, multi-GPU, or
multi-node reproduction. The deterministic fixture validates the preprocessing
offset calculation that caused the reported mismatch, not model semantics.

The generic `_adjust_embedding_length` helper still raises when handed a
synthetically short embedding. Thus this result verifies the correction for
the reported mixed image/video GLM trigger; it does not prove that every future
source of malformed multimodal embeddings is isolated to one request.
