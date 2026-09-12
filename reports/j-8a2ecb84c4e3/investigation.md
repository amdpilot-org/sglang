# DSPARK/DFLASH `min_new_tokens` correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33493

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2011

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1887 at exact commit
`4fc5ad8a7df70cd4eb4306b7d08acf190ce7a6d9`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1974

## Finding

The candidate's stale-field correction is valid: overlap-mode sampling stores
precomputed additive penalties in `acc_additive_penalties`, and DFLASH/DSPARK
verification must consume that field and disable DSPARK's no-adjustment fast path.

The review's remaining lifecycle counterexample also reproduces. Speculative
decode returns from `ScheduleBatch.prepare_for_decode` before the ordinary decode
cumulation site. EAGLE performs the equivalent update in
`eagle_prepare_for_decode`, while `DFlashDraftInputV2.prepare_for_decode`, shared by
DFLASH and DSPARK, did not. With a real `BatchedMinNewTokensPenalizer` configured
for two tokens, the candidate's adjustment tests passed but the EOS penalty was
still `-inf` after two DFLASH preparation calls.

The consolidated correction preserves the candidate adjustment fix and adds the
missing per-step cumulation to the DFLASH-family prepare owner. The lifecycle
regression creates the real penalizer, observes its initial stop mask, advances it
through two actual DFLASH preparation calls, refreshes `acc_additive_penalties`
after each step, verifies that EOS becomes available exactly at the minimum, and
then disables and removes the penalty.

## Evidence

- `raw/candidate_before.log`: five candidate adjustment checks pass and the real
  lifecycle check fails because EOS remains `-inf` after the second preparation.
- `raw/passing_after.log`: all six focused checks pass after the cumulation fix.
- `raw/existing_tests.log`: existing DFLASH logits and decode bookkeeping tests
  pass (12 tests).
- `raw/gfx950_adjustment.log`: on the assigned AMD Instinct MI350X
  (`gfx950:sramecc+:xnack-`), the adjustment operation matches an explicit CPU
  reference exactly and preserves negative infinity.

## Limitation

The exact DeepSeek-V4-Flash-0731 TP4 HTTP reproduction was not run. The environment
has one gfx950 GPU and no qualified model weights. The tiny Llama transport fixture
cannot validate DSPARK's DeepSeek-V4 architecture, semantic behavior, or a TP4
workload, so it was not substituted as proof.
