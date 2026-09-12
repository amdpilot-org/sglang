# Independent review of PR 816

Reviewed `https://github.com/amdpilot-org/sglang/pull/816` at exact commit
`fc221c29a238457b74ee4e2b5ee26e32cb299982` against upstream issue
`https://github.com/sgl-project/sglang/issues/38360` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/861`.

Recommendation: **accept**. The candidate is a full fix for the reported response-shape
defect. On the recorded base, issue-valued scalar outputs become a flat list and fail
`ScoringResponse` validation. On the candidate, scheduler-transported Python scalar
outputs become one-element rows, while existing one-label rows and multi-label vectors
remain unchanged.

The independent GPU probe ran on one AMD Instinct MI350X (`gfx950`) and exercised the
actual `CrossEncodingPooler`, the scheduler's `.tolist()` transport representation, the
candidate score processor, and `ScoringResponse`. The pooler emitted `(2,)` values
`[5.875, -10.109375]`, exactly equal to an independent tensor reference, and the final
response validated as `[[5.875], [-10.109375]]`.

The prepared source imports resolved to `/job/repo/python/sglang`. No native files or
native build inputs changed, and `repository-environment.json` declares no separate
native component, so a native rebuild was not applicable.

Limitations: the BGE weights were unavailable, so the original full model server and
HTTP curl were not run. The deterministic probe validates the defect's complete
pooler-to-schema shape contract, but not BGE semantic accuracy or NVIDIA execution.
A directly injected 0-D tensor is still rejected, but it is not a production-path
counterexample because `BatchResultProcessor._convert_embeddings` converts tensors via
`.tolist()` before tokenizer-manager processing.
