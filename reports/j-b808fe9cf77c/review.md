# Independent review of amdpilot-org/sglang PR 1887

Upstream issue: https://github.com/sgl-project/sglang/issues/33493

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1924

Candidate: https://github.com/amdpilot-org/sglang/pull/1887 at
`4fc5ad8a7df70cd4eb4306b7d08acf190ce7a6d9`

## Recommendation

Request changes. The candidate is a verified partial fix, not a full resolution
of the original `min_new_tokens` contract.

The recorded base reads the nonexistent `acc_linear_penalties` attribute in
both the verify adjustment helper and DSPARK's no-adjustment predicate. The
candidate correctly changes both reads to `acc_additive_penalties`. Its focused
regression fails on the base (four failures) and passes at the exact candidate
(five passes). An independent gfx950 test also confirmed that additive
penalties and logit bias are broadcast across all verify positions, preserve
`-inf`, convert dtype correctly, and match an explicit CPU reference exactly.

However, DFLASH-family decode preparation in
`dflash_info_v2.py::DFlashDraftInputV2.prepare_for_decode` still does not call
`ScheduleBatch.cumulate_penalty_output_tokens()`. EAGLE explicitly performs
that update. The real `BatchedMinNewTokensPenalizer` advances its
`len_output_tokens` only through that call. The retained lifecycle test shows
that its EOS adjustment remains `-inf` after repeated applications without
cumulation and becomes zero only after the configured number of cumulations.
Therefore the candidate prevents stopping before the minimum, but can keep
stop tokens suppressed after the minimum and force generation toward
`max_tokens`. That is a remaining counterexample to full min-token semantics.

The candidate's own regression never advances the real penalizer lifecycle; it
only injects a precomputed tensor. It proves the stale-field correction but not
the complete serving behavior reported by the issue.

The candidate PR body and committed investigation also cite mirror issue
`https://github.com/amdpilot-org/sglang/issues/1860`, not the assigned mirror
issue `https://github.com/amdpilot-org/sglang/issues/1924`.

## Environment and scope

The prepared interpreter imported both edited modules directly from
`/job/repo/python`. Torch was `2.11.0+rocm7.2`. GPU execution used the single
assigned AMD Instinct MI355X (`gfx950`). No C++ or other native source changed,
so no native rebuild was applicable. The installed AITER extension loaded from
the job-private runtime cache.

The reported DeepSeek-V4-Flash command requires TP4 and qualified model
weights. This environment had one GPU and no qualified weights, so the exact
full-model HTTP reproduction was not possible. The tiny Llama fixture was not
substituted because it cannot validate DSPARK/DeepSeek-V4 architecture or
distributed behavior.

## Evidence summary

- Base plus candidate regression file: four failed and one passed, demonstrating
  omission of accumulated penalties and the incorrect DSPARK fast-path result.
- Exact candidate focused regression: five passed.
- Exact candidate existing DFlash logits suite: ten passed.
- Independent gfx950 adversarial script: passed, including two batch rows, four
  verify positions, fp32-to-fp16 conversion, `-inf`, logit-bias composition,
  invalid row count, and zero draft-token validation.
- Real min-new-tokens penalizer lifecycle: `-inf` initially and without
  cumulation; still `-inf` after one of two required cumulations; zero after the
  second.

Raw commands, outputs, import paths, issue/PR snapshots, and the exact candidate
patch are retained under `reports/j-b808fe9cf77c/evidence/`.
