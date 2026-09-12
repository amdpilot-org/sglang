# Correction review for PR 736

Candidate: https://github.com/amdpilot-org/sglang/pull/736 at
`2aea7cad31ab4ef5547ee7a23c6425220e41a65f`

Independent review: https://github.com/amdpilot-org/sglang/pull/842

The candidate correctly changes the 42-token, width-6 capture from eight rows
`[6, 6, 5, 5, 5, 5, 5, 5]` to seven uniform rows. It also fixes the separately
reproduced DSV4 replay copy failure when source and destination are overlapping
views of the same device allocation.

It does not resolve the Engram contract described by the issue for genuinely
ragged tiers. Running the actual layout helpers gives:

- 41 tokens: `[6, 6, 6, 6, 6, 6, 5]`
- 43 tokens: `[6, 6, 6, 5, 5, 5, 5, 5]`
- 47 tokens: `[6, 6, 6, 6, 6, 6, 6, 5]`

Each remains incompatible with an assertion requiring
`num_tokens == request_count * 6`. The prepared base and candidate contain no
`EngramHasher` or `python/sglang/srt/layers/engram.py`, so a layout-aware Engram
change and real per-request hash-output comparison cannot be implemented or
validated from this source tree. DeepSeek-V4.1-Flash weights and the reported
TP4/EP4 NVIDIA deployment are also unavailable.

Accordingly, this correction preserves the two source changes that are
demonstrably valid, expands the tier regression to record the remaining
counterexamples, and reports the candidate as rejected rather than claiming
the upstream issue is fully fixed.
