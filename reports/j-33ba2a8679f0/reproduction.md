# Correction-generation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/31207

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3518

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3500 at `fb34ada0b8d8b9db3ee04ecd4c82c563db4c0cdd`

Independent-review parent: https://github.com/amdpilot-org/sglang/pull/3514

The prepared base was `358c163250ad3b1f62939b01ce1314a0a31a0365`. The exact candidate commit was cherry-picked without modification before reproduction.

## Failing before

`candidate_c128_before.txt` is the raw output from constructing the candidate's actual `DecodeKVCacheOffloadManager` with a `DeepSeekV4TokenToKVPool` instance shell and a mocked result from `build_deepseek_v4_hicache_stack`. The returned host group contained `PoolName.DEEPSEEK_V4_C128_STATE`. The candidate produced no matching sidecar and the assertion exited 1.

Observed values:

```text
entry_present= True
transfer_present= False
transfers= []
AssertionError: C128 state entry was dropped
```

## Passing after

The same constructor reproduction is in `candidate_c128_after.txt`. It exits 0 and reports:

```text
entry_present= True
transfer_present= True
transfers= ['deepseek_v4_c128_state']
```

The focused suite in `focused_pytest_after.txt` reports 27 passed and 8 subtests passed. It includes a constructor-level regression verifying that C128 state derives indices from SWA and uses trailing-page storage matching.

The retained GPU primitive check in `gpu_composite_transfer_after.txt` reports exact D2H byte equality for three paged buffers and two ring-state buffers on one AMD Instinct MI350X with HIP 7.2.26015. This checks transfer primitives only, not DeepSeek-V4 model semantics.

## Remaining external and environment limitations

The candidate's LMCache path still intentionally raises `NotImplementedError` when the installed external connector lacks a `pool_group` parameter. This checkout does not contain an LMCache grouped connector or end-to-end grouped transfer protocol, so LMCache DeepSeek-V4 operation is not claimed fixed.

No DeepSeek-V4 weights or B300 distributed PD environment were available. Attention correctness, request-lifetime behavior under real serving, storage restore, unified-KV, HiSparse, and multi-rank behavior remain unverified. No native source changed, so a native rebuild was not applicable.
