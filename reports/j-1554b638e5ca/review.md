# Independent review of PR 3422

Upstream issue: https://github.com/sgl-project/sglang/issues/38478

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3401

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3426

Candidate reviewed at exact commit `e7dff9af317bbfd5701e4c43f3b0b94f40c63a8b` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. The image-prepared checkout matched the recorded base.

## Recommendation

Accept. The candidate fully implements the original opt-in contract in the source and contract-testable environment: default-off lookahead after a queue-head `NO_TOKEN`, ordinary `PrefillAdder.add_one_req` gates for candidates, priority-scheduling disable, bounded same-head aging, receipt-balanced prefix locking across rematch/head change/removal/admission, and a post-pin capacity reserve. I found no remaining functional counterexample within the exercised scheduler/cache contract.

The base was reproduced from the actual prepared checkout: its waiting-queue loop unconditionally reaches `break` after the first non-`CONTINUE` result, including a head `NO_TOKEN`, and it has no lookahead implementation. This establishes the failing-before behavior, though it is a deterministic source/control-flow reproduction rather than the unavailable H100 TP8 deployment.

The exact candidate imports resolved to `/job/repo/python/sglang/...`, not an installed wheel. It changes only Python and test/report files, so no native rebuild was applicable. Candidate tests passed (40 tests), and an independent adversarial script passed lock lifetime, head-change/admission balance, bounded aging, and refusal/unwind when a pin would violate reserve.

## Scope limits

The available device was one AMD Instinct MI350X (`gfx950`) with ROCm 7.2 and PyTorch 2.11.0+rocm7.2. No model weights were needed for this scheduler/cache control-flow review. This environment cannot reproduce the reported prefill-only H100 TP=8 hierarchical-cache deployment, NVIDIA behavior, distributed timing/races, or the claimed throughput, TTFT, idle-round, cache-hit, and production lock-count percentages. Those performance claims remain deployment-unverified and are not used as acceptance evidence.

## Commands

- Base control-flow reproduction: prepared checkout at `358c163250ad3b1f62939b01ce1314a0a31a0365`, inspected and asserted the actual admission loop's unconditional break after `NO_TOKEN`.
- Candidate tests: `/tmp/amdpilot-repo-j-1554b638e5ca/venv/bin/python -m pytest -q test/registered/unit/managers/test_prefill_lookahead.py test/registered/unit/managers/test_prefill_adder.py test/registered/unit/managers/test_scheduler_decision_batch_params.py` (exit 0, 40 passed).
- Independent adversarial checks: `/tmp/amdpilot-repo-j-1554b638e5ca/venv/bin/python /tmp/amdpilot-repo-j-1554b638e5ca/independent_adversarial.py` (exit 0).
- Source import check at the candidate commit confirmed both `sglang` and `prefill_lookahead` loaded from `/job/repo/python/sglang`.
- GPU architecture inventory: `rocm-smi --showproductname --showmeminfo vram --showuse` (exit 0; MI350X/gfx950; no candidate GPU workload executed).

