# Independent review of PR 1608

Reviewed exact candidate `c0d54de7cd2aafe30f38d9b98c4c8e9c034ec70c` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue https://github.com/sgl-project/sglang/issues/35201.

Recommendation: **request changes**. This is a partial fix, not a full original-issue fix.

The base schedules/allocates the full FP8 paged or nonpaged logits rectangle. For the issue dimensions, `4096 * align256(92992) * 4` is `1526726656` bytes (`1.421875 GiB`). The candidate adds row budgeting to those paths, correctly chunks the eager case, makes the previously excluded graph modes use a static budget, and rejects a budget smaller than one aligned row. Its focused regression suite passes.

The remaining counterexample is graph-backed execution. The candidate deliberately avoids `mem_get_info` there and derives its budget from configured static headroom. At the issue settings on an 80 GiB device, `mem_fraction_static=0.9` and the default logits fraction of `0.2` produce a 1.6 GiB graph budget. Since 1.421875 GiB is below that budget, `rows_per_chunk` remains `None`. If live free memory is the issue-reported 1.17 GiB, the unchanged full allocation can still OOM. Thus the patch removes unbounded context growth in graph mode but does not guarantee that the issue-sized allocation fits the memory actually available.

Evidence retained outside the revision-switched checkout is under `/job/review-evidence/`. The candidate tests passed (`20 passed`, `29 subtests passed`); the same regression at parent candidate `ace5d612...` reproduced five failures. An independent GPU test on the assigned MI350X/gfx950 verified identical selected top-k pages for unchunked execution and row chunks of 1, 3, 7, 16, and 40.

Environment limitations: this host has one AMD Instinct MI350X with ROCm 7.2, not the reported four H100/CUDA 12.9 system. DeepSeek-V4-Flash-0731 weights were unavailable. Consequently CUDA DeepGEMM, TP4, and the full 800K-token serving reproduction were not executed. No native files changed, so no native rebuild was applicable.
