# Independent review of PR 2214

Recommendation: **accept** at exact commit `fc7f4447bf3957533ad13f882953b2f7b5cda0b8`.

The candidate is test-only hardening, not a duplicate production fix. The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the upstream PR 31982 correction: both reported `Tensor.item()` invariants are gated by `SGLANG_MAMBA_DEBUG_ASSERTS`, off by default and enabled only by the exact value `1`. The candidate adds focused regression coverage and a GPU probe without changing either production file.

The exact candidate tests passed. On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), the actual `HybridReqToTokenPool.donate_mamba_ping_pong_slot` method emitted zero `local_scalar_dense` profiler operations with debugging disabled and one with debugging enabled; the enabled case preserved the invalid-slot assertion. Independent checks covered both source locations and additional noncanonical flag values. The candidate regression also failed against source from the real upstream pre-fix parent `0eae9423d86ddf0c820dd47c97b97fb779467b91`.

No counterexample to the issue's source-level contract was found. `fully_resolves_original` is therefore true for removing the two reported default-path scalar reads. This does not claim a full hybrid-Mamba serving reproduction: no suitable model weights were provided, and scheduler starvation, detokenizer heartbeat behavior, sustained-load throughput, and model semantics remain unverified. The second `mamba_radix_cache.cache_finished_req` location was structurally verified but not dynamically exercised end-to-end. No native source changed, so no native rebuild was applicable.

The required recorded base is identical to the image-prepared checkout and is already fixed. Consequently, the original failure cannot be reproduced honestly at that revision; historical failing-before evidence is retained separately and is not represented as a base failure.

Upstream issue: https://github.com/sgl-project/sglang/issues/31970

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2159

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2250
