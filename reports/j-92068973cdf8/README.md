# Investigation of sglang#35811

Upstream issue: https://github.com/sgl-project/sglang/issues/35811

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1298

## Outcome

Environment blocked. No source correction is included because the reported failure could not be reproduced or attributed narrowly with the available inputs.

The report requires DeepSeek-V4-Flash-0731 weights, NVIDIA B200 execution, tensor parallelism across two GPUs, the FlashInfer MXFP4 MoE backend, the FP4 indexer, DSpark, and the reporter's multi-round workload. This job provides one AMD MI355X (`gfx950`) GPU, no reported model weights, and no benchmark dataset. The upstream issue also contains no workload after a maintainer requested one. A tiny Llama transport fixture would not exercise the DeepSeek-V4 architecture, FP4 indexer, FlashInfer backend, DSpark, TP2, or the implicated cache layout, so it was intentionally not presented as a reproduction.

## Current-source evidence

The prepared checkout is `358c163250ad3b1f62939b01ce1314a0a31a0365`; the report named `880ab72f34ead8f39544c7c7e08edbd443964a1f`. Current source contains substantial later DeepSeek-V4/HiCache coverage, including:

- `test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_dsv4.py`, with DeepSeek-V4 Flash L2/L3, multi-turn/cache-hit, EAGLE, and DSpark configurations.
- `test/registered/unit/mem_cache/test_unified_radix_hicache_dispatch.py`, which asserts DeepSeek-V4 selects its specialized HiCache stack before the generic SWA stack.
- `test/registered/unit/mem_cache/test_dsv4_c4_state_lifecycle.py`, which checks request-scoped C4 state sizing, reset isolation, and new-versus-reused request-slot boundaries.
- `test/registered/unit/model_executor/test_pool_configurator.py`, which checks invalid one-page SWA pools and the first valid boundary.

Related merged upstream PR https://github.com/sgl-project/sglang/pull/38957 states that it fixed DeepSeek-V4.1 encoder SWA replay and preserved DeepSeek-V4 unified-KV HiCache dispatch. That is evidence of a related post-report correction, but it is not enough to claim the original B200 DeepSeek-V4-Flash DSpark failure fixed: the issue has no stack trace in text, no benchmark workload, and no confirmation tying that PR to issue 35811.

## Local validation

The focused CPU suite passed 76 tests (one skipped, seven subtests), covering current DSV4 HiCache dispatch, C4 state lifecycle, staged-transfer dispatch, and pool-sizing boundaries. See `raw/focused_cpu_tests.log`.

On the assigned gfx950, 59 HiCache kernel cases passed against byte-exact tensor references for MHA/MLA round trips, staged page-first write-back, alignment boundaries, and page-count boundaries. The two selected DSV4 paged HiSparse cases were skipped by their explicit `is_hip()` guard because that implementation is CUDA-only. See `raw/gfx950_hicache_kernel_tests.log`.

These results show that the tested current-source paths do not expose a neighboring generic transfer defect on gfx950. They do not reproduce or clear the original NVIDIA B200 failure.

## Remaining work needed to resolve the source issue

Run the reporter's actual multi-round workload (or a supplied deterministic reproducer) using the named DeepSeek-V4-Flash checkpoint on at least two B200 GPUs and the exact serving flags. Capture the first CUDA fault with synchronous launch diagnostics and isolate whether it occurs in HiCache transfer, DeepSeek-V4 compressed attention/indexer state, speculative KV handling, or another kernel. Only then can a failing-before/passing-after regression be tied to the original defect.
