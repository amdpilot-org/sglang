# Independent review of amdpilot-org/sglang PR 1398

Recommendation: **request changes**. Candidate commit `2ffc47a4c31856746c82020ea6e55e50cc2c8709` is a partial fix, not a full resolution of the original issue.

The recorded base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`, with no difference from the image-prepared checkout. On that base, the candidate regression failed because `HiCacheStorageConfig` had no `dp_rank`, confirming that DP identity could not reach Mooncake's SSD path selection. On the exact candidate, its three-test regression passed. An independent test mapped 12 distinct `(dp_rank, tp_rank, pp_rank)` tuples to 12 distinct existing directories and confirmed that the path passed to Mooncake remains unchanged when `storage_config is None`.

The blocking counterexample is `MooncakeDirectLinker`, selected through the registered unified-cache path. It constructs `HiCacheStorageConfig` without `dp_rank`. The candidate's new field silently defaults to zero, so two DP-attention clients with equal attention-local TP rank and PP rank still select the same `rank_0_<tp>_<pp>` directory. Only `HiCacheController` was updated to propagate its DP rank. The issue's requested documentation of the one-client-per-directory rule is also absent.

Imports were confirmed from `/job/repo/python/sglang/...`, not an installed SGLang copy. No native source changed and no native rebuild was applicable. The host exposed one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) with Torch `2.11.0+rocm7.2`; GPU execution was intentionally not claimed because this is Python path-selection logic. The production DeepSeek-V4-Flash TP8 topology, multiple ranks/nodes, live Mooncake bucket collision, and restart recovery were not available and remain unverified.

Raw outputs and the full candidate diff against the recorded base are under `raw/`.
