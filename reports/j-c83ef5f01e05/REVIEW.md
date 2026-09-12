# Independent review of PR 3348

Candidate: `89326ddaae9db3f6ce59041bb6db3df362318389`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**

The candidate is a narrow source-level fix for the original HYV4 contract
mismatch. On the recorded base, both HYV4 architectures return no
`disable_attn_tp_gather` override. The model sends its full-width hidden state
directly to `DeepseekV2MoE`, while the scheduler's A2A gather predicate derives
a rank-local non-padded count. On an 8-row fixture, that count of 1 causes the
real GPU MoE mask helper to invalidate rows 1 through 7.

At the exact candidate commit, the model override declares
`disable_attn_tp_gather=True` for both HYV4 architectures and makes the field
eligible for the model-resolution pipeline. The focused candidate test and an
independent configuration matrix pass. A non-HYV4 DeepSeek control remains
unchanged. Because current HYV4 never consumes the scheduler gathered buffer,
the model-wide opt-out is appropriate even when the current invocation uses no
A2A backend or has attention TP size one; in those configurations it is inert
and prevents the same mismatch if parallel settings change.

No native source changed. Imports were explicitly measured from
`/job/repo/python/sglang/...`, so the tested Python was the candidate checkout,
not an installed copy. No native rebuild was applicable.

## Scope and limitations

The assigned device was one AMD Instinct MI355X, gfx950, with ROCm 7.2 and
PyTorch 2.11.0+rocm7.2. It executed the repository's actual padded-row masking
helper and matched an independent tensor reference for 0, 1, 5, and 8 valid
rows. Only one GPU was available. The Hunyuan-V4-preview checkpoint and a TP8 /
EP8 DeepEP deployment were unavailable, so this review does not claim a
full-model logprob, throughput, multi-rank communication, or semantic-accuracy
reproduction. The source-level contract and its concrete masking consequence
are nevertheless directly reproduced and corrected.

The complete model-override module produced 91 passes and four failures. The
same four tests fail on the recorded base in this ROCm environment; they depend
on CUDA/SM100 behavior or `modelopt_fp4`, which is rejected on ROCm. They are
not introduced by this candidate.

No remaining counterexample was found for the original contract. A future HYV4
implementation that adds and consumes a real `LayerCommunicator` would need to
revisit the model-wide opt-out, but that is not the implementation reviewed.

Raw logs are retained in `reports/j-c83ef5f01e05/raw/`.
