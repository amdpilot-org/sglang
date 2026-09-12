# Independent review of PR 2589

Candidate: https://github.com/amdpilot-org/sglang/pull/2589 at exact commit `49a136764025c3a0ac76b2105884bbf1215e1e0b`

Upstream issue: https://github.com/sgl-project/sglang/issues/30314

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2590

## Verdict

Request changes. The candidate is a valid partial correction for two deterministic Mamba-slot accounting defects, but it does not fully resolve or verify the original event-loop hang contract.

On the recorded base, the candidate regression independently failed because decode-prefix unlock was disabled by default and enabling it reduced the lazy/overlap pool ratios below the admission-time peak. At the exact candidate commit all 10 focused tests passed. The fixture directly demonstrates that a `2N` pool with `N` request-owned slots and `N` admission-locked prefix slots cannot allocate a donated slot and raises `Can not alloc mamba cache`; `3N` succeeds, and `2N` succeeds only when prefix slots are evictable. The correction therefore preserves useful work from the parent candidate while restoring admission headroom.

This does not establish bounded scheduler progress when allocation still cannot succeed. The allocation path still raises synchronously after eviction cannot reclaim a slot, and no candidate regression drives scheduler/HTTP handling concurrently with an unreclaimable allocation. The candidate also does not reproduce or explain the zero-running post-flush case, direct hierarchical-cache I/O blocking, the 222-second TTFT, or TP=8 behavior. Its request-boundary suite measures configured token/request limits rather than responsiveness under Mamba admission pressure.

## Environment and source verification

The prepared checkout was exactly base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Candidate imports resolved to `/job/repo/python/sglang/...`, including `environ.py`, `kv_cache_configurator.py`, and `unified_radix_cache.py`. The candidate changes Python only; no native source changed and no native rebuild was applicable.

The available environment is Torch `2.11.0+rocm7.2` on ROCm 7.2, not the reported eight H100 80GB CUDA system. Qwen3.5-397B-A17B-FP8 weights, TP=8, EAGLE, 100K+ traffic, and the production direct-I/O setup were unavailable. No GPU/model execution is claimed. The deterministic tests qualify CPU-side allocation/accounting behavior only.

## Evidence

- `base-regression.log`: candidate regression run against the recorded base; 10 tests ran with 2 failures.
- `candidate-regression.log`: exact candidate; 10 focused tests passed.
- `candidate-allocation-eviction.log`: exact candidate; 17 allocation-aware eviction tests passed.
- `candidate-request-bounds.log`: exact candidate; 8 request-boundary tests passed.
- `candidate.patch`: exact base-to-candidate diff preserved outside revision switches.

## Remaining counterexamples

1. No bounded-progress or fail-fast scheduler behavior is demonstrated for an admission whose Mamba allocation remains unreclaimable; the focused fixture expects the allocator assertion rather than proving server responsiveness.
2. No `/health` or completion-handling regression runs while Mamba admission cannot make progress.
3. The reported post-flush state with zero running requests is not reproduced or explained.
4. Direct hierarchical-cache I/O blocking and the 222-second TTFT are untested.
5. The reported Qwen3.5 hybrid-Mamba, TP=8, H100/CUDA workload is untested in this ROCm single-GPU environment.
