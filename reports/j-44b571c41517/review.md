# Independent review of PR 2635

Candidate: https://github.com/amdpilot-org/sglang/pull/2635 at `f917979aeb38db42c20d164042cb24e2d7a1be65`

Upstream issue: https://github.com/sgl-project/sglang/issues/30314

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2636

## Verdict

Request changes. The candidate is a credible partial fix and regression hardening, but it does not fully resolve the original issue's scheduler-liveness and serving-responsiveness contract.

The recorded-base comparison independently reproduced the two narrow failures claimed by the candidate: decode-prefix Mamba eviction is not enabled by default, and the opt-in lazy-buffer ratio is only 3 rather than the admission-safe 4. At the exact candidate, all 10 focused tests pass, as do 17 related unified-cache allocation/eviction tests. Imports resolve to the checked-out source.

The decisive remaining counterexample is also encoded in the candidate test: with `N` request-owned states plus `N` admission-locked prefix states in a `2N` pool, donated-state allocation performs one eviction attempt and raises `AssertionError: Can not alloc mamba cache`. This establishes bounded allocator behavior in the fixture, but not graceful scheduler behavior. There is no regression showing request rejection/rollback, continued scheduling, completion processing, or `/health` responsiveness after that failure.

The candidate therefore qualifies as a partial resource-headroom fix plus test hardening. It is not proof that the original intermittent server hang, direct HiCache I/O stall, 222-second TTFT, or zero-running post-flush exhaustion is fixed.

## Environment and architecture limits

The prepared interpreter is `/tmp/amdpilot-repo-j-44b571c41517/venv/bin/python`, using Torch `2.11.0+rocm7.2`. The host exposes one AMD Instinct MI350X/gfx950 GPU. The reported Qwen3.5-397B-A17B-FP8 weights, eight H100 80GB GPUs, CUDA, TP=8, EAGLE configuration, 100K+ contexts, and production traffic were unavailable. The tiny Llama transport fixture cannot exercise hybrid-Mamba allocation and was not used as substitute proof.

No native files changed, so a native rebuild was not applicable.
