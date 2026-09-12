# Independent review of PR 2342

Upstream issue: https://github.com/sgl-project/sglang/issues/33268

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2374

Candidate: https://github.com/amdpilot-org/sglang/pull/2342 at exact commit
`c8135ea13ade6c245ab99fef8be25e19f54998ca`.

## Recommendation

Request changes. The candidate is a substantive partial fix, not test-only
hardening: it adds the logical KV-cache dtype to `HiCacheStorageConfig`, fixes
the core file backend and several other in-tree key paths, and fixes the
specific Aibrix E4M3/E5M2 collision. Its focused regression suite passes.

It does not fully resolve the original all-backend contract. On the exact
candidate, independent interface-level probes demonstrated that NPU MemCache
still maps `fp8_e4m3` and `fp8_e5m2` to the identical key `same_page`, while
SiMM maps both to `same_page_0_k` and `same_page_0_v`. Neither backend refuses
a dtype-bearing configuration. FlexKV and LMCache remain unverified because
their external packages/services are absent; neither integration was changed
by the candidate.

## Evidence

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: an independent probe confirmed
  `HiCacheStorageConfig` has no dtype field and bf16/fp8 runs both derive
  `page_model_0_1` (`raw/base-original-collision.txt`).
- Exact candidate: its five focused test files passed, 62 tests total
  (`raw/candidate-regression.txt`).
- Exact candidate: an independent service-free probe invoked the production
  NPU MemCache and SiMM key transformations and confirmed identical keys for
  logical `fp8_e4m3` and `fp8_e5m2`
  (`raw/candidate-adversarial-remaining.txt`).
- The candidate changed Python only. No C/C++/HIP/native source changed, so no
  native rebuild was required. Imports resolved from `/job/repo/python`; Torch
  resolved from `/opt/venv/lib/python3.12/site-packages/torch`.

## Environment limitations

The host exposed one AMD Instinct MI355X (`gfx950`) through ROCm 7.2 and Torch
2.11.0+rocm7.2. GPU execution was not needed for string/key derivation and was
not used as evidence. `aibrix_kvcache`, `flexkv`, `lmcache`, `memcache_hybrid`,
and `simm` were not installed, and no corresponding services, Ascend NPU, model
weights, or multi-node environment were available. Aibrix, NPU MemCache, and
SiMM were therefore checked at their in-tree client boundaries with minimal
stubs; no end-to-end backend claim is made. FlexKV and LMCache remain
unverified rather than presumed fixed or broken.
