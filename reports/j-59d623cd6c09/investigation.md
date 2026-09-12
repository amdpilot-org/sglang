# GLM-5.3 vision candidate correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/38821

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1103

Candidate parent: https://github.com/amdpilot-org/sglang/pull/972

Independent-review parent: https://github.com/amdpilot-org/sglang/pull/1066

## Outcome

PR 972 at exact commit `7764c2a5b91022ed2ad96e9f60bad81d6276dfae`
contains valid GLM-5.3 processor registration, image preprocessing, JPEG data-URL,
resize, and single-expanded-span fixes. Those changes are preserved here.

The independent review's adjacent-placeholder counterexample was reproduced
against that exact candidate. Given pretokenized input `[1, 10, 99, 99, 11, 2]`
and two `image_data` items, the candidate passed `[1, 10, 99, 11, 2]` to
`load_mm_data`, silently reducing two adjacent placeholders to one. See
`raw/candidate_adjacent_failure.txt`.

The correction makes GLM-5.3 normalization aware of request image cardinality.
It retains at least one placeholder per separated run and additional adjacent
placeholders up to the number of supplied images, without inventing placeholders.
The same counterexample passes after the change, as do the candidate's prior
single-run/separated-run cases, a no-invention boundary, the non-GLM boundary,
synthetic JPEG preprocessing, resize boundaries, and neighboring GLM mixed-offset
tests.

## Semantic limitation

This source correction does not claim to reproduce or explain the original
Statue of Liberty-to-bird semantic response. The exact JPEG with SHA-256
`ff13fd6f991b37253d3745dc6b9ef8e7a92f17cee7c4c8bc84735000a668fcd7`, the
private `GLM-5.3-Flash-sglang0907-128k-0907t3` weights, the reported 0907
deployment, and eight NVIDIA H20 GPUs were unavailable. The available single AMD
MI355X/gfx950 cannot qualify that architecture or its landmark semantics. The
reported deployment already produced visual bird semantics, so missing processor
registration and token normalization alone do not establish a causal explanation
for that observation.

No native source changed, so a native rebuild was not applicable. No GPU model
execution was performed.
