# Independent review of PR 930

Candidate: https://github.com/amdpilot-org/sglang/pull/930
Exact commit: `a3fa16067a307d43108209a4235e91e0b7a3611c`
Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The patch is a substantial partial fix, but it does not fully implement the issue's requirement to calculate memory from all physical tensors owned by the active layout.

## Finding

`HiSparseC4DevicePool` allocates `self.data_ptrs` as a GPU tensor after its base `kv_buffer` allocation. The candidate adds `DeepSeekV4SingleKVPool.allocated_tensors()`, but that method returns only `self.kv_buffer`. Consequently, `DeepSeekV4TokenToKVPool._finalize_mem_usage()` omits the HiSparse pointer table.

On the exact candidate, an actual gfx950 allocation produced:

```text
hisparse: expected=3594952 reported=3594944 delta=8 mem_usage=0.003348052502
```

The eight-byte difference is one `uint64` pointer for the reduced fixture's single C4 layer; it scales with the number of HiSparse compressed layers. The submitted verifier independently sums only `child.kv_buffer`, so it encodes the same omission and cannot prove the full original contract.

## What is fixed

The recorded base reproduced the original zero-value defect for every exercised layout. At the candidate commit, independently counted bytes matched `mem_usage` exactly for:

- non-unified FP8 storage;
- the HIP/AITER FP4 indexer layout; and
- unified KV storage.

The candidate's complete focused test file also passed. This is functional source correction, not merely test hardening, but it remains partial because of the HiSparse counterexample.

## Environment and scope

Tests used `/tmp/amdpilot-repo-j-da6f828b96d2/venv/bin/python`. The imported SGLang source was `/job/repo/python/sglang`; Torch was `/opt/venv/lib/python3.12/site-packages/torch`. Hardware execution used one AMD Instinct MI355X (`gfx950`) with PyTorch `2.11.0+rocm7.2`.

DeepSeek-V4 weights and the original 8×H100 CUDA environment were unavailable, so this review does not claim a full server/HTTP, CUDA, multi-rank, or model-semantic reproduction. NPU-specific accounting was inspected in source but could not be executed. The candidate changes Python only; no native rebuild was applicable.

Raw commands and outputs are retained under `reports/j-da6f828b96d2/evidence/`.
