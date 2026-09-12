# Independent review of PR 1028

Reviewed `amdpilot-org/sglang` PR 1028 at exact commit
`b747fb9ee118b07112396a58cb3873ee30145cfa` against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the original issue's
stated contract. On the base, `ResponsesRequest` accepted but silently removed
`prompt_cache_key`. At the candidate, the field survives validation and the
Responses adapter forwards it as `GenerateReqInput.cache_salt` when the native
`cache_salt` is absent. A non-empty native `cache_salt` retains precedence.

The candidate's focused tests passed (61 tests plus 2 subtests). An independent
single-GPU HTTP run used the qualified deterministic tiny-Llama fixture from
amdpilot-org/sglang PR 649 at exact commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Four identical prompts with cache
keys A/A/B/B returned HTTP 200 and cached-token counts 0/47/0/47, demonstrating
same-key reuse and different-key isolation through the real Responses serving
path and radix cache.

The imported source paths were the checkout files under
`/job/repo/python/sglang/`, not an installed SGLang wheel. No C++, HIP, or
FlyDSL source changed, so a native rebuild was not applicable. The GPU was one
AMD Instinct MI350X (`gfx950`) with ROCm 7.2 and Torch 2.11.0+rocm7.2. This does
not reproduce the report's NVIDIA RTX 5090, Qwen3.8-27B hybrid GDN, 36,022-token
latency, semantic-isolation, or distributed workload claims.

Upstream issue: https://github.com/sgl-project/sglang/issues/37263

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1064
