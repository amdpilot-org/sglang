# Independent review of EAGLE greedy TP synchronization

Candidate: https://github.com/amdpilot-org/sglang/pull/2531 at
`eed79ea7bf564e3c5975f68d68b8c258fd1acf6c`

Upstream issue: https://github.com/sgl-project/sglang/issues/31071

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2545

## Recommendation

Accept. The candidate fully addresses the source-level contract described by
the original issue: every multi-rank greedy EAGLE verify path now broadcasts
the finalized `predict`, `accept_index`, and `num_correct_drafts` values from
rank 0. This removes the backend-dependent `_is_hip` guard that left non-HIP
greedy execution able to retain different accepted-token counts.

This is a source-level and deterministic-regression verification, not a claim
that the production deadlock was reproduced end to end. Only one GPU was
assigned, so a real two-rank ROCm process group and subsequent collective hang
could not be exercised.

## Evidence

The prepared checkout was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The candidate's regression was
copied outside the checkout before revision switching. On the base, its four
cases produced `1 failed, 3 passed`: the non-HIP TP=2 near-tie case retained
accepted counts 3 and 2 on the two logical ranks. The HIP case already passed
because the prepared base contained a narrower HIP-only broadcast.

At the exact candidate commit, the candidate suite and two independent
multi-request adversarial cases produced `6 passed`. The independent cases
used different per-rank token decisions and different per-request accepted
counts, and checked both the ordinary TP group and DP-attention TP group. All
three finalized outputs on the second logical rank matched rank 0 after the
broadcast replay.

The tests used real GPU tensors and `torch.argmax` on an AMD Instinct MI350X,
`gfx950:sramecc+:xnack-`, with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`.
Imports resolved to `/job/repo/python/sglang/srt/speculative/eagle_utils.py`.
No native source changed, so a native rebuild was not applicable.

## Limitations and remaining counterexamples

- The host exposed one GPU. No real two-process/two-GPU TP process group was
  run, and the reported downstream collective deadlock was not reproduced.
- AITER all-reduce fusion, a full serving deployment, model weights, semantic
  generation accuracy, and multi-node behavior were not tested.
- The regression uses a replay group to model rank-0 broadcast semantics. It
  verifies the call site, tensors, group selection, and resulting decisions,
  but it does not validate ROCm collective transport.

Raw logs and the independent test are retained outside the checkout at
`/job/review-evidence-j-bf9f8e9e88b8/`.
