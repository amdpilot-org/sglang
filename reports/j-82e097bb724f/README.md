# Independent review of PR 872

Upstream issue: https://github.com/sgl-project/sglang/issues/38009

Mirror issue: https://github.com/amdpilot-org/sglang/issues/905

Candidate: https://github.com/amdpilot-org/sglang/pull/872 at
`3c8680cd17302b3a17b8de8463944e67a000000d`.

## Recommendation

**Accept as test-only hardening, not as a fully verified resolution of the
original issue.** `fully_resolves_original` is false.

The candidate changes no production source or native code. It adds a
parameterized unit test which checks that the current DFlash acceptance helper
calls the TP synchronization boundary for greedy, sampling, and selector
verification. The focused file passes at the exact candidate (13 tests). The
same file at the prepared base already passes its pre-existing 10 tests, and
the production synchronization behavior is already present there. Thus PR 872
does not itself fix the defect; it hardens tests around an earlier source fix.

The earlier production fix is upstream commit
`f60bc73c5836d45457575c17f4722c6bd60f06b0`, which is an ancestor of the
recorded base. Its pre-fix parent computes/commits rank-local DFlash acceptance
without a TP broadcast. The fixed implementation broadcasts greedy target
predictions, or sampled accept length and bonus, before `_commit_accept`.

An independent adversarial script outside the candidate checkout emulated a
nonzero rank whose local decisions disagree with rank 0. Both greedy and
sampling paths consumed the broadcast replacement before committing output.
This validates ordering and data flow more strongly than merely recording the
number of `sync` calls.

## What remains unverified

The original report's decisive contract is exact greedy token equality between
target-only Qwen3.8-27B and Qwen3.8-27B-DFlash2 with thinking enabled, with the
reported no-thinking control. That run could not be performed: the assigned
host has one AMD Instinct MI350X (`gfx950`, ROCm 7.2), not four NVIDIA RTX 3090
GPUs with CUDA/NCCL and TP=4, and neither 27B checkpoint is available. The
token-30 divergence was therefore not reproduced on the recorded base and the
full semantic equivalence claim remains unverified.

The candidate test uses a recording mock for `sync`; it does not launch
multiple ranks or exercise an actual collective. The independent adversarial
case verifies that an in-place broadcast result is consumed before commit, but
also remains a single-process unit fixture. This is sufficient to accept the
test hardening, not to claim the original model-level issue is fully resolved.

## Evidence

- `evidence/base-tests.log`: 10 pre-existing tests passed on recorded base.
- `evidence/candidate-tests.log`: 13 tests passed at the exact candidate.
- `evidence/adversarial-accept-sync.log`: forced rank-disagreement data-flow
  checks passed for greedy and sampling acceptance.
- `evidence/base-import-gpu.txt`: actual source import paths and gfx950 tensor
  execution.
- `evidence/candidate-import.txt`: exact candidate imported the checkout's
  DFlash worker source.
- `evidence/pre-fix-accept-source.txt`: pre-fix acceptance implementation,
  showing no TP synchronization before commit.

No native files changed in PR 872, so no native rebuild was applicable.
