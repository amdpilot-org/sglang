# Independent review of PR 2376

Reviewed candidate commit `cad9443d635cf595832da6a7e4d6f82ce81adcf4`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and
the original issue contract.

## Recommendation

Request changes. The candidate is useful test-only hardening for the existing
ROCm-specific fix, but it does not establish a full fix for the original
backend-independent greedy TP-consistency contract and labels the result
`fixed`. The implementation at both the recorded base and candidate only
broadcasts finalized greedy decisions when `_is_hip` is true.

The candidate's focused test passed at the exact candidate commit on the
assigned gfx950 GPU. Removing only the existing HIP broadcast block made two
of its three tests fail, so the regression is sensitive to the ROCm fix. The
same test also passed unchanged on the recorded base, confirming that the
candidate does not introduce the implementation fix.

An independent adversarial variant used the same two logical ranks and
opposed near-tied GPU logits but selected the non-HIP greedy branch. With
`world_size=2`, no broadcasts occurred and the returned accepted-draft counts
remained different (`3` on logical rank 0 and `2` on logical rank 1). This is
a remaining counterexample to the issue's stated requirement that greedy
accept decisions be TP-consistent regardless of per-rank logit differences.

## Scope and limitations

Only one AMD Instinct MI350X (`gfx950`) was assigned. GPU tensor and argmax
execution were real, but the tests emulate two logical ranks and replay rank
0 broadcast values. They do not exercise a real TP=2 process group, reproduce
the subsequent collective deadlock, run AITER all-reduce fusion, or validate
full EAGLE serving/model semantics. No model weights were available. No
native source changed in the candidate, so a native rebuild was not
applicable.

The prepared interpreter imported
`/job/repo/python/sglang/srt/speculative/eagle_utils.py`, confirming that the
checked-out repository source—not an installed copy—was tested. Raw command
outputs are stored alongside this report.
