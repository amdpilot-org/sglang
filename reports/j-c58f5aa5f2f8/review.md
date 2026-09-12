# Independent review of candidate PR 576

Reviewed exact candidate commit `3893cff8924af7d59701ab5a2861272fb3d366ed` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: request changes. The patch is a useful partial fix, not a full resolution of the original issue.

The candidate correctly brings `cal_padded_tokens()` into line with the attention-TP alignment performed by `ForwardBatch.prepare_mlp_sync_batch()`. Its test failed on base (`3 != 4`) and passed at the exact candidate commit. Independent GPU cases for several non-divisible token/TP combinations also produced the expected zero-padded row metadata.

However, the original invariant remains unenforced. On the candidate, a `ForwardBatch` marked as preplanned and non-replannable at three rows can be changed to four physical rows, after which `needs_forward_metadata_init()` still returns `False`. There is no common mismatch rejection and no central explicit compact extent restricting attention, KV writes, and row-indexed metadata to the real rows. The candidate therefore addresses only the smaller DSA padding-prediction consistency issue called out near the end of the report.

Validation ran on AMD Instinct MI355X (`gfx950`) with PyTorch 2.11.0+rocm7.2 and HIP 7.2.26015. No native source changed, so no FlyDSL/native rebuild applied. No model-weight end-to-end speculative DSA run was performed, and CUDA-only behavior was not tested.
