# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/38360

Mirror issue: https://github.com/amdpilot-org/sglang/issues/760

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the non-generation branch of
`TokenizerManagerScoreMixin._process_single_item_scoring_results` appended a
scalar classification result directly. `ScoringResponse.scores` declares one
list of floats per item, so the issue values became a flat list and failed
Pydantic validation.

The preserved failing-before test demonstrates the exact reported shape with
`5.875` and `-10.109375`. The correction wraps only scalar classification
outputs after optional softmax. Existing multi-label vectors are unchanged.

The assigned gfx950 check exercised the actual `CrossEncodingPooler`, confirmed
that its final `squeeze(-1)` emits a rank-1 batch for a single-label head, and
then validated the corrected rows through `ScoringResponse`. It used synthetic
deterministic inputs with an independent tensor reference because the reported
BGE weights were not available. This is not a full BGE server or semantic
accuracy reproduction.

Raw command output is retained under `raw/`. No native code changed and no
native rebuild was applicable.
