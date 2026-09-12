# HiCache dtype key correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/33268

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2299

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2203

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2265

## Result

The independent Aibrix counterexample was reproduced at exact candidate commit
`8665dbfc8eac3fd74113ba2365e7409ce80f1c44`. With a shared native
`torch.float8_e4m3fnuz` dtype, configurations for logical `fp8_e4m3` and
`fp8_e5m2` both passed `("same_page",)` unchanged to `BlockHashes`; the
reproduction exited 1.

This correction preserves the candidate's valid changes and scopes every
Aibrix `BlockHashes` input with `HiCacheStorageConfig.kv_cache_dtype`. A missing
dtype retains the legacy unscoped input. The same reproduction now observes
`dtype_fp8_e4m3_same_page` and `dtype_fp8_e5m2_same_page` and exits 0.

## Commands

```bash
git switch --detach 8665dbfc8eac3fd74113ba2365e7409ce80f1c44
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-2fcb53cc45eb/venv/bin/python \
  /tmp/amdpilot-repo-j-2fcb53cc45eb/adversarial_aibrix.py
# AssertionError: AIBrix key collision across logical FP8 formats

PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-2fcb53cc45eb/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_aibrix_kv_cache_dtype_key.py \
  test/registered/unit/mem_cache/test_hicache_kv_cache_dtype_key.py \
  test/registered/unit/mem_cache/test_mem_pool_host.py \
  test/registered/unit/mem_cache/test_mooncake_tenant_config.py \
  test/registered/unit/mem_cache/test_umbp_store.py
# 62 passed
```

## Limitations

The installed environment has no `aibrix_kvcache`, `lmcache`, or `flexkv`
package/service. Aibrix key construction was therefore validated with the
independent review's stubbed client and a registered regression, not end to
end. FlexKV and LMCache key derivation remains unverified.

NPU memcache and SiMM source still derives component keys without the new dtype
field, but their required architecture/services are unavailable. No speculative
change was made for those paths. No model weights, serving process, multi-node
workload, or GPU execution was needed or claimed. No native source changed, so
no native rebuild was applicable.
