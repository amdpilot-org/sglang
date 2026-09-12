# HiCache KV-cache dtype key isolation

Upstream issue: https://github.com/sgl-project/sglang/issues/33268

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1921

The prepared base reproduced the reported collision: `HiCacheStorageConfig` had
no dtype field, so bf16 and fp8 runs generated the identical file-backend key
`page-1_DeepSeek-V4-Flash`. The failing output is retained in
`raw/reproduction-before.txt`.

This change threads the host pool's actual runtime dtype through
`HiCacheStorageConfig` and scopes the file, NIXL, EIC, Mooncake, and hf3fs key
paths. The hf3fs path and metadata key are both scoped, and its write, read,
existence, and deletion operations use the same prefix. The regression includes
an in-process hf3fs mock-client round trip, cross-dtype miss, and legacy
`None`-dtype round trip, plus file-backend TP/PP/CP/component-key boundaries.

The implementation was compared with the current open upstream candidate PR
https://github.com/sgl-project/sglang/pull/36420 and applies its current commit
`7f9a5702378b493ccfe09d78e756668d1792abf8` to the prepared base. The tests here
were executed independently in the prepared interpreter.

No GPU execution was used: storage-key derivation and the deterministic mock
hf3fs round trip are CPU behavior, and running an unrelated GPU smoke would not
add evidence for this defect. No model weights, external hf3fs deployment,
Mooncake service, NIXL service, EIC service, multi-node setup, or full serving
run was available or claimed. FlexKV and LMCache use separate connector/client
key paths rather than `HiCacheStorageConfig`; Aibrix supplies dtype in its
external library's `KVCacheBlockSpec`. Those external systems were inspected
but not integration-tested here.

The attempted direct pytest collection of
`python/sglang/srt/mem_cache/storage/mooncake_store/test_mooncake_store.py` is
retained in `raw/related-tests.txt`; that standalone script requires Mooncake
environment configuration and includes a function argument that is not a
pytest fixture. The registered unit tests were rerun separately and passed.
