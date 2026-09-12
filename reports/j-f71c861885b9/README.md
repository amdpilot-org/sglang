# Consolidated HiCache dtype-key correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33268

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2117

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1993

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2081

## Result

The two remaining review counterexamples reproduce at exact candidate commit
`d38922c9de6874789c758a6931895eb142332a77` on base
`358c163250ad3b1f62939b01ce1314a0a31a0365`:

- `_generate_storage_config` used the host byte-storage dtype, so E4M3 and E5M2
  configurations both became `torch.uint8`.
- `UMBPStore` omitted the dtype from `config_prefix`; the direct linker also
  constructed `HiCacheStorageConfig` without a dtype.

The retained before run has two failures. The correction preserves the
candidate's valid file, hf3fs, Mooncake, NIXL, and EIC changes, while using the
explicit runtime KV format for storage namespaces. This is necessary on ROCm,
where E4M3 and E5M2 may share both the host `uint8` representation and the same
native Torch FP8 dtype. For `auto`, the resolved device dtype remains the
fallback. UMBP now includes the dtype in its prefix, and its direct linker
threads the configured format into the shared config.

## Reproduction and verification

Candidate-failing review regressions:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-f71c861885b9/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_hicache_kv_cache_dtype_key.py::TestHiCacheDtypeKeyCollision::test_production_config_preserves_logical_fp8_format \
  test/registered/unit/mem_cache/test_umbp_store.py::TestUMBPStore::test_config_prefix_isolates_kv_cache_dtype
# 2 failed
```

Consolidated after suite:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-f71c861885b9/venv/bin/python -m pytest -q \
  test/registered/unit/mem_cache/test_hicache_kv_cache_dtype_key.py \
  test/registered/unit/mem_cache/test_mem_pool_host.py \
  test/registered/unit/mem_cache/test_mooncake_tenant_config.py \
  test/registered/unit/mem_cache/test_umbp_store.py
# 60 passed
```

Raw pytest output is retained under `raw/`.

## Limitations

No external hf3fs, Mooncake, NIXL, EIC, UMBP, FlexKV, or LMCache service was
available, so those integrations were not exercised end to end. The tests use
the existing hf3fs mock client and mocked UMBP client to validate namespace
construction and local read/write symmetry. No model weights, serving run, or
multi-node workload was used or claimed. GPU execution would not strengthen
these CPU-side string/key regressions and was not used. No native source was
changed, and the prepared environment declares no native rebuild target.
