# Independent review of PR 1035

Upstream issue: https://github.com/sgl-project/sglang/issues/37393

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1068

Candidate: https://github.com/amdpilot-org/sglang/pull/1035 at
`5247b13d80bb3e7549c5517a7da17ca469ac8cd0`

## Recommendation

Request changes. The candidate is test-only hardening around a plausible fix
already present in the recorded base; it is not an independently verified full
resolution of the original production failure.

The candidate adds no runtime source changes. Its five-test regression passes
unchanged on the required recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, so it is not a
failing-before/passing-after candidate regression. A source comparison does
confirm that the reported v0.5.18 build lacks the current
`DSPARK_TARGET` synchronization call and that the prepared base contains it.
That makes the pre-existing synchronization a credible mitigation for one
failure chain, but it does not prove that this was the cause of the observed
unequal `VocabParallelEmbedding` row counts or that every source of TP state
divergence is closed.

## Review findings

1. The regression uses a fake TP group whose `broadcast` is a local
   same-shaped `copy_`. It does not execute a distributed collective and does
   not reproduce the original contract violation: different ranks entering a
   `VocabParallelEmbedding` all-reduce with 628 versus 1140 rows (or 6098 versus
   9682 rows).
2. The AST assertion proves only that one synchronization call textually
   precedes one sequence-length assignment. It does not run chunked prefill,
   multimodal embedding, DSpark acceptance/draft state transitions, request
   retirement, or concurrent batching.
3. The candidate's regression passes 5/5 on both the recorded base and the
   exact candidate. Therefore the candidate adds coverage, not a runtime fix.
4. The synchronization is enabled by the default `SGLANG_SPEC_TP_SYNC=all`,
   and the assigned-GPU tensor overwrite behaves numerically as expected.
   Explicit `SGLANG_SPEC_TP_SYNC=off`, however, disables this protection; no
   guard or original-row-count invariant was tested.
5. The candidate report and PR body cite mirror issue `/975`, not the assigned
   mirror issue `/1068`, and use `candidate_verified` even though they expressly
   lack real multi-rank, full-model, B300, NCCL, and sustained-traffic evidence.

## Evidence and limitations

The prepared checkout exactly matched the recorded base. The exact candidate
was checked out detached for its tests, then the checkout was returned to
`amdpilot/j-1e9005587aeb` before this report was committed. Source imports
resolved to `/job/repo/python/sglang`, including the DSpark worker. No native
files differ between the base and candidate, so no native rebuild was
applicable.

Available hardware was one AMD Instinct MI355X (`gfx950`) with PyTorch
2.11.0+rocm7.2. The reported environment was eight NVIDIA B300 GPUs with CUDA
13.0 and NCCL 2.28.3. Kimi-K3/DSpark weights were unavailable. Consequently,
this review does not claim a full-model, TP=8, NCCL, multimodal serving,
semantic-accuracy, or sustained-concurrency reproduction.

Raw logs and compared source snapshots were preserved outside the checkout in
`/job/review-evidence/` while revisions were switched.
