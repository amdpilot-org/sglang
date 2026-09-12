# DeepSeek-V4 DP gather regression correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33360

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2079

Candidate parent: https://github.com/amdpilot-org/sglang/pull/1958 at
`152efc1aaa4da7cf62be370847166446667d21e2`

Review parent: https://github.com/amdpilot-org/sglang/pull/2045

The prepared base already contains the issue-specific production correction:
the synchronous post-attention MoE path gathers replicated hidden states with
`dp_gather_replicate`, and the main and NextN token-ID gathers clone the local
`input_ids[:, None]` view. This change preserves those fixes and corrects only
the candidate's regression oracle.

## Independently reproduced review counterexamples

At the exact candidate commit, the original regression passed (`3 passed`),
then also passed (`3 passed`) against a temporary source fixture containing all
three corruptions from the review:

- both token-ID local operands were changed to `input_ids_global.clone()`;
- an extra `dp_gather_partial(hidden_states, local_hidden_states, ...)` was
  inserted beside the required replicate call.

The corrected regression failed that same combined fixture with two failures:
one for the forbidden partial gather and one for the wrong token-ID expression.
It passed the unmodified prepared source with all three tests passing. The
token assertion now compares the complete AST expression to
`input_ids[:, None].clone()`, and the synchronous path explicitly forbids any
partial gather while retaining the independent TBO partial-gather boundary.

## GPU mechanism evidence

On the assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), summing a
replicated tensor scaled it exactly by widths 2, 4, and 8, while selecting a
single replica preserved it. Zeroing `input_ids[:, None]` mutated the original
IDs; zeroing its clone preserved them. This validates the arithmetic and
aliasing mechanisms only, not the distributed SGLang collective.

## Remaining limitation

The original eight-H800 TP8/DP4 DeepSeek-V4-Flash-0731 Marlin serving case was
not executable: this job has one AMD gfx950 GPU, no NVIDIA H800/CUDA/Marlin,
and no model weights. Its generated-text semantic accuracy therefore remains
unverified. No speculative production change is made from that limitation.
