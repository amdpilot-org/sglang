# Independent review of amdpilot-org/sglang PR 827

Candidate: `0b653c280b4e96ffa31571a4cff170c95f6322e8`

Recommendation: **request changes**. The patch is a partial fix and does not fully resolve the original issue.

The patch correctly moves the load-burst hand-over out of prefill batch formation and stages pure-prefill `input_ids`, extend metadata, and DP token-count metadata before `ready_to_load_host_cache()`. Its regression suite passes, as does the broader focused suite, and the staging primitive completed a real H2D copy on the assigned gfx950 GPU.

The unresolved case is a mixed prefill+decode batch. `pre_upload_forward_inputs()` explicitly skips the `input_ids` upload whenever `mix_running_indices` is present. `_handover_hicache_load()` then hands over the burst. At forward entry, `resolve_forward_inputs()` uploads `prefill_input_ids_cpu` and combines it with the decode tokens. The independent probe recorded:

```text
['burst_handover', 'input_ids_h2d']
```

That is the ordering the original issue asks to eliminate. The candidate's own `test_mixed_batch_keeps_the_deferred_path` test confirms the behavior but treats it as expected, so the suite does not enforce the full original contract.

The original source-level failure was present at the recorded base: `ready_to_load_host_cache()` ran before `prepare_for_extend()`. The candidate fixes that general placement and several metadata uploads, but the mixed-batch `input_ids` counterexample prevents accepting it as a complete fix.

Environment limitation: only one AMD Instinct MI355X (`gfx950`, ROCm 7.2) was available. The reported 8x H100 TP=8 multi-GiB HiCache timing behavior and all-reduce consequences were not reproduced or profiled. No native source changed, so a native rebuild was not applicable.

Raw review evidence was preserved outside the checkout at `/job/review-evidence/` while revisions were switched.
