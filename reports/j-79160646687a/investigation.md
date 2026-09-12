# Independent review of PR 2497

Reviewed exact candidate: `454114868ff60985549c5602da2e440e41b78d3d`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Original candidate: https://github.com/amdpilot-org/sglang/pull/2345 at `12863b77a051754759334e8752894f633b3bc5fd`

Prior independent review: https://github.com/amdpilot-org/sglang/pull/2397

Candidate under review: https://github.com/amdpilot-org/sglang/pull/2497

## Finding

Recommendation: **accept** the bounded correction. It fixes both concrete counterexamples left by the prior review. It must not be described as a fully reproduced end-to-end fix for the original H20/Qwen incident.

The recorded base has the earlier `None -> 0` fallback, so it no longer raises the literal `torch.tensor([None])` `TypeError`. However, an independently constructed freed overlap-lagged row on the assigned GPU returned `[101, 202]`: the fallback selected stale mapping column 0 for the freed row. Safe behavior is `[-1, 202]`, where `-1` is the existing skip/null sentinel.

At the exact candidate commit, the same case returned `[-1, 202]`. The candidate's 18 focused tests passed. Independent cases also passed for an explicit lazy track-position override, a freed row in either batch position, two freed rows, unchanged live rows, and downstream masking of `last_correct_step_indices` before recurrent-state scatter.

Source imports were confirmed from `/job/repo/python/sglang/srt/managers/schedule_batch.py` and `/job/repo/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` while detached at the candidate. The commit changes no native source, so rebuilding native code was not applicable.

## Classification

- Original base behavior: **partial fix**. It prevents the `NoneType` construction failure but can gather stale state for a freed request.
- Candidate behavior at the approved deterministic boundary: **verified correction**. No remaining counterexample was found for freed-row slot selection or non-track commit masking.
- Full original issue: **not fully verified**. The available environment cannot reproduce the reporter's NVIDIA H20, Qwen3.6-35B-A3B-FP8, EAGLE, FA3, or sustained serving configuration.

Raw command output is retained in this report's `evidence/` directory. More complete transient review material was preserved outside the revision-switching checkout at `/job/review-evidence-j-79160646687a/`.
