# Independent review of PR 3500

Upstream issue: https://github.com/sgl-project/sglang/issues/31207

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3489

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3501

Candidate: https://github.com/amdpilot-org/sglang/pull/3500 at `fb34ada0b8d8b9db3ee04ecd4c82c563db4c0cdd`

Recorded base and prepared checkout: `358c163250ad3b1f62939b01ce1314a0a31a0365` (no difference).

## Recommendation

Request changes. The candidate is a partial decode-offload fix and LMCache diagnostic hardening, not a full resolution of the original composite-pool contract.

The base failure was reproduced from the prepared source: `DecodeKVCacheOffloadManager` rejected a `DeepSeekV4TokenToKVPool` with `ValueError: Unsupported KV cache type for decode offload`. The candidate's focused suite passes, and its synthetic GPU test confirms exact D2H bytes for the selected paged/state buffers.

The independent adversarial test executes the candidate constructor with a composite host group that exposes `PoolName.DEEPSEEK_V4_C128_STATE`. The manager's hard-coded sidecar enumeration drops that entry. This means the candidate still does not preserve every available V4 state sub-pool and is not the requested capability-driven enumeration.

The LMCache change does not add hybrid-pool support. It resolves a `DevicePoolGroup`, inspects the external connector signature, and raises `NotImplementedError` because current LMCache connectors accept only `k_pool`/`v_pool`. That is a useful diagnostic, but the original LMCache failure remains functionally unresolved. LMCache is not installed in the prepared environment, preventing an end-to-end connector/daemon check.

## Evidence

- `base_decode_repro.txt`: failing-before constructor traceback from the recorded base.
- `candidate-pytest.txt`: candidate regression result, 26 passed and 8 subtests passed.
- `candidate-gpu.txt`: MI355X/ROCm 7.2 synthetic exact-byte D2H output.
- `adversarial_c128_state.txt`: failing independent state-preservation assertion.
- `candidate.patch`: exact base-to-candidate patch preserved outside revision switching and copied here.

No native files changed, the environment metadata declares no native build target, and no rebuild was applicable. Source imports for the tested decode module resolved under `/job/repo/python/sglang`; the LMCache module also resolved from that tree but stopped at the missing external `lmcache` package. The GPU is one AMD Instinct MI355X, not the reported 8xB300 configuration. No V4 weights were available, so full-model attention correctness, distributed PD decode, unified-KV/HiSparse variants, storage restore, and multi-rank execution remain unverified.
