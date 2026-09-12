# Independent review of PR 2203

Upstream issue: https://github.com/sgl-project/sglang/issues/33268

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2238

Candidate: https://github.com/amdpilot-org/sglang/pull/2203 at exact commit `8665dbfc8eac3fd74113ba2365e7409ce80f1c44`

Recommendation: **request changes**. The candidate is a useful partial fix but does not fully resolve the original all-backend contract.

## Findings

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the independent contract script reproduced the original collision: otherwise identical bf16 and fp8 configurations produced the same file key. The base also did not carry a dtype through the production storage config or UMBP prefix.

At the exact candidate commit, independent checks confirm that file keys differ, explicit `fp8_e4m3` and `fp8_e5m2` remain distinct even when the runtime/native storage dtype is shared, and UMBP receives the logical dtype. The candidate's focused suite passed 60 tests.

However, `AibrixKVCacheStorage` ignores `storage_config.kv_cache_dtype`. Its get/set/exists paths pass the original page keys directly to the external client's `BlockHashes`, and initialization does not reject dtype-unsafe operation. A stubbed AIBrix client captured identical `(('same_page',), 64)` hash inputs for `fp8_e4m3` and `fp8_e5m2`; the adversarial assertion failed as expected. This violates the original issue's requirement that every backend isolate dtype or refuse loudly.

FlexKV and LMCache are separate radix-cache implementations rather than `HiCacheStorageConfig` consumers in this checkout, but their external packages/services were absent and the candidate did not change their key derivation. Their coverage therefore remains unverified. NPU memcache and SiMM are also registered persistent HiCache backends whose visible component-key construction does not consume the new dtype field; the required services/architectures were unavailable.

## Evidence

- `raw/base-independent-contract.txt`: failing-before reproduction on the recorded base.
- `raw/candidate-independent-contract.txt`: passing independent checks for the candidate's corrected paths.
- `raw/candidate-regression.txt`: 60 candidate tests passed.
- `raw/candidate-adversarial-aibrix.txt`: executable remaining counterexample and assertion failure.
- `raw/candidate-import-paths.txt`: confirms imports came from `/job/repo/python`, not an installed SGLang wheel.
- `raw/environment.txt`: Torch/ROCm and assigned GPU details.
- `raw/dependency-availability.txt`: unavailable external client packages.
- `raw/independent_contract.py` and `raw/adversarial_aibrix.py`: review harnesses.

## Environment and native status

The host exposed one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. No GPU execution was needed for string/key identity and none is claimed. No native source changed in the candidate, so a native rebuild was not applicable. External services, model weights, serving, and multi-node behavior were not exercised.
