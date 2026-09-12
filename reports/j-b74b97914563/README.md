# Independent review of PR 2220

Reviewed exact candidate commit `3c6d8d65e5445df04576fb1bec215ea28d044b27` against upstream issue https://github.com/sgl-project/sglang/issues/32158 and candidate mirror issue https://github.com/amdpilot-org/sglang/issues/2123.

Recommendation: **accept**. The recorded base commit `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduces the accounting defect through the real `SchedulerLoadInquirer.get_loads()` method: with four unique request IDs distributed across PP slots, the base reports only the one request in the currently selected batch. The exact candidate reports all four and passes independent deduplication and boundary cases. The implementation changes the scheduler callback to expose every `running_mbs` slot only when PP is enabled and counts unique `rid` values; the non-PP path remains a singleton tuple containing `running_batch`.

The candidate test also fails on the base and passes on the candidate, but that result alone is weaker because the base lacks the candidate's new field/method. The independent `full_snapshot_contract.py` fixture is therefore the primary failing-before/passing-after evidence.

Architecture limitation: this host provides one AMD Instinct MI350X (`gfx950`) GPU. The report used CPU scheduler bookkeeping and did not execute GPU kernels. It cannot reproduce the original NVIDIA H20 TP=2, PP=4, eight-GPU HTTP workload, model execution, or pipeline communication. No native files changed, so no native rebuild was required. Source imports were verified from `/job/repo/python` at both revisions.

Raw logs, exact source diff, issue snapshots, import paths, and executable independent fixtures are retained in `evidence/`.
