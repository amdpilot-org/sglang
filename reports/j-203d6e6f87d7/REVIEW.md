# Independent review of candidate b12ae4a

Recommendation: accept. The candidate fully resolves the original reported default dLLM radix-cache ownership/accounting failure in the classic and current default unified implementations.

On base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the preserved candidate regression failed with the issue-specific signature: after two different token keys reused one volatile dLLM KV page, both cache implementations accounted for 128 tokens but referenced only 96 tokens of distinct physical pages. At exact candidate `b12ae4a2681706a9ab161b2d432808f8f9f9ce86`, the focused suite passed (54 tests, 61 subtests), and an independent matrix covering incomplete lengths 1, 17, 31, and 32 plus final resolution passed.

The fix is causal rather than test-only: `ReqDllmMixin.get_cacheable_fill_ids()` excludes the in-flight incomplete denoise block, and both `RadixCache.cache_unfinished_req()` and `UnifiedRadixCache.cache_unfinished_req()` use that stable prefix. Once the block resolves, the complete fill IDs become cacheable again.

No native source changed, so no native rebuild was applicable. Imports were verified from `/job/repo/python/sglang/srt/...`. The host is x86_64 with one AMD Instinct MI350X (`gfx950`); a small Torch GPU computation matched its CPU reference. LLaDA2.1 weights and Ascend 910B3 were unavailable, so the original full HTTP/model workload remains unrerun. The deterministic allocator/tree fixture directly tests the reported ownership and invariant mechanism; the tiny Llama transport fixture would not exercise dLLM and was intentionally not treated as proof.

Experimental C++ radix and legacy standalone SWA/Mamba paths were not qualified. They are outside the reported/default LLaDA configuration; the current default unified cache is covered.

Upstream issue: https://github.com/sgl-project/sglang/issues/35270

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1905
