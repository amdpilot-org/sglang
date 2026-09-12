# Independent review of PR 611

Candidate: https://github.com/amdpilot-org/sglang/pull/611  
Exact commit: `2286e15f1b108f52426d58a71034b76048f6316a`  
Upstream issue: https://github.com/sgl-project/sglang/issues/39147  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/613

## Finding

Recommendation: **accept**. The candidate fully resolves the original issue's `HiCacheFile.batch_exists_v2()` existence-query contract.

On prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the unmodified source implementation reproduced the exact failure: `[3, 2, 2]` versus expected `[3, 0, 1]`. At the pinned candidate commit, the identical reproduction returned `[3, 0, 1]`.

The implementation replaces the minimum of unrelated per-pool maxima with an intersection of each pool's legal endpoints. This directly addresses sparse `TRAILING_PAGES` endpoints and `ALL_PAGES` truncation. It also returns the complete common endpoint list in `restorable_prefix_pages`, consistent with the existing Mooncake implementation.

Independent testing compared the candidate against a separately written endpoint oracle over 20,480 combinations of contiguous KV lengths, every page-presence subset for two auxiliary pools, all combinations of `ALL_PAGES` and `TRAILING_PAGES`, and trailing window sizes one through three. No mismatch or remaining counterexample was found.

## Environment and scope

The tested module loaded from `/job/repo/python/sglang/srt/mem_cache/hicache_storage.py` on both revisions. The host is x86_64 Linux with Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015, and one AMD Instinct MI350X (`gfx950`). No GPU kernel was run: the reported behavior is a CPU filesystem query and does not use accelerator data. No native file changed in the candidate, so no native rebuild was applicable.

This review does not establish model-output correctness, an end-to-end cache load, or scheduler use of L3. Those are outside the explicitly scoped original issue.

Raw outputs and source-path evidence are retained in this directory.
