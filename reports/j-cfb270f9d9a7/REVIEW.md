# Independent review of PR 982

Reviewed exact candidate commit `956321de134fde54ee75b71aad60be1546163183` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue https://github.com/sgl-project/sglang/issues/38156.

Recommendation: **request changes**. The candidate is a substantive partial fix, not test-only hardening: a world-CPU-group minimum snapshot plus locally accumulated reservations removes the allocation-order double charge for equal-sized TP/PP/DP ranks. Its four-process Gloo regression passed with primary budgets of 22.5 GiB on all ranks and 2.5 GiB remaining for sidecars; all 22 focused host-pool tests passed.

It does not fully meet the issue's stated aggregate-fit contract for heterogeneous ranks. With 100 GiB initially available, a 10 GiB reserve, four local ranks, and requests of 30+20+20+20 GiB, the 90 GiB aggregate fits exactly. The retained equal-share calculation nevertheless gives every rank 22.5 GiB and rejects the 30 GiB pipeline stage. The candidate itself acknowledges this limitation, but reports outcome `fixed`. This is a concrete remaining PP topology counterexample, not an architecture-only concern.

The candidate's `candidate_regression.py` is not a valid recorded-base reproduction: on the base it errors while trying to patch the absent `host_memory_sync_group` symbol. The independent base reproduction instead calls the actual base implementation with deterministic pre/post-allocation readings and obtains 22.5 then 12.5 GiB for a fitting 20 GiB request.

No native source changed, so no native rebuild was applicable. The imported source was `/job/repo/python/sglang/srt/mem_cache/pool_host/base.py`. The environment had one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), Torch 2.11.0+rocm7.2, and ROCm 7.2. The accounting regressions are CPU/Gloo tests. The exact 8xH200 model workload, model weights, multi-node execution, and a real heterogeneous PP serving job were unavailable and are not claimed.

