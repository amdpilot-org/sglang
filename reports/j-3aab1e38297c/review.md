# Independent review of amdpilot-org/sglang PR 1639

Candidate reviewed: `aae252eb457f1f0358ae4b98897f3a8fa6eda50d`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate is a meaningful partial fix, but it does not fully resolve the original umbrella issue.

## Findings

1. A repeated sequence with a different immutable-decision payload is silently accepted as a duplicate. `_pp_apply_bootstrap_decision` returns whenever `decision.sequence < expected` without retaining or comparing the previously applied `(good_rids, bad_rids)`. The original contract explicitly says payload divergence must raise immediately. The independent adversarial test applies sequence 0 as good and then repeats sequence 0 as bad; no exception is raised. Evidence: `evidence/candidate_adversarial.txt`.

2. The original issue tracks three fixes, including router replay prevention in upstream PR 34570. Candidate `aae252e` changes only the scheduler/bootstrap protocol and HiCache readiness paths; it has no `sgl-model-gateway` change. It therefore cannot fully resolve the original issue's replay-amplification component. Evidence: `evidence/upstream_pr_34570.json` and `evidence/candidate-source.diff`.

## What was verified

- On the recorded base, the candidate regression cannot collect because `PPBootstrapDecision` is absent. Source inspection also shows the vulnerable local failure and mutable return-ring implementation. Evidence: `evidence/base_candidate_regression.txt` and `evidence/base_related_paths.txt`.
- At the exact candidate commit, all candidate protocol tests pass: 24 tests plus 4 parameterized subtests. They cover local KV-failure proposals, abort fencing/deferment, sequence gaps, exactly-once application, metadata reservation, and HiCache readiness. Evidence: `evidence/candidate_pytest.txt`.
- Modified Python modules compile. There are no C++ or native-source changes, so no native rebuild applies.
- Imports resolve to `/job/repo/python/sglang` and `/job/repo/python/sglang/srt/managers/scheduler_pp_mixin.py`, not an unrelated installed SGLang package. Evidence: `evidence/candidate_import_paths.txt`.
- The candidate does fix the demonstrated unilateral `KVPoll.Failed` queue mutation and replaces per-stage mutable return values with a sequenced frozen decision. This is substantive source behavior, not test-only hardening.

## Architecture and environment limits

The assigned environment exposes one gfx950 GPU. The reported production topology requires PP=8 disaggregated prefill, Mooncake, HiCache L2-only, 32K-token shared-prefix requests, and cancellation storms. That topology, model weights, and multi-node workload were not available. No full NCCL deadlock reproduction, full model run, or soak was performed. A single-GPU serving smoke would not validate this PP control-plane contract, so none is claimed.

The independent deterministic tests establish a concrete remaining protocol counterexample, but do not measure model semantics or production NCCL scheduling.
