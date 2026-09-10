# skip_cache_insert device-cache validation

## Scope and candidate

- Upstream context: sgl-project/sglang issue 38069 and pull request 38738.
- Tested upstream candidate head: `d239509eb8d851c16ccf16b337e1d7f9f66b6bb0`.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c` (`origin/main`).
- The candidate commits were cherry-picked onto the mirror branch; their content is preserved, while the local commit IDs differ after the rebase.
- Added `test/registered/unit/mem_cache/test_skip_cache_insert_device_lifecycle.py` to drive the real device pools and lifecycle with synthetic KV tensors.

## Environment

- Qualified image (operator-specified): `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Container hostname `banff-cyxtera-cx57-5` is not used as image identity.
- GPU: one AMD Instinct MI300X, reported capability `(9, 4)` (`gfx942`).
- Python: `/opt/venv/bin/python`.
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, version `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm/HIP: `7.2.26015-fc0010cf6a`.
- Working SGLang source: `/job/sglang/python/sglang`.
- Native modules: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py` and `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Build and download caches were kept under `/tmp/sglang-cache-j-9c489a55391c`; no model weights were downloaded.

## Validation

The focused GPU test constructs a small `MHATokenToKVPool`, `ReqToTokenPool`, `TokenToKVPoolAllocator`, and `RadixCache`. It allocates real device slots, writes synthetic K/V tensors, drives `maybe_cache_unfinished_req` and `release_kv_cache`, and performs radix matching by tenant salt.

- Opted-in request: six slots remain owned by a reusable radix entry, the request row is released, and a PyTorch SDPA reference output is identical before and after finish-path release.
- Opted-out request: no radix entry is created, all six KV slots return to the allocator, and the request row is released.
- Interrupted opted-out request: `release_kv_cache(..., is_insert=False)` releases all allocated slots and leaves no radix entry.
- Tenant salt: the opted-in entry matches only under `tenant-a`, not `tenant-b`; the opted-out requests leave no entry under their salts.

Raw results:

```text
test/registered/unit/mem_cache/test_skip_cache_insert_device_lifecycle.py
3 passed, 3 warnings in 19.05s

adjacent candidate tests
244 passed, 45 warnings, 81 subtests passed in 21.34s

pre-commit (focused file)
all applicable hooks passed
```

Reproduction:

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-9c489a55391c/inductor
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-9c489a55391c/triton
export XDG_CACHE_HOME=/tmp/sglang-cache-j-9c489a55391c/xdg
export HF_HOME=/tmp/sglang-cache-j-9c489a55391c/hf
/opt/venv/bin/python -m pytest \
  test/registered/unit/mem_cache/test_skip_cache_insert_device_lifecycle.py \
  test/registered/unit/mem_cache/test_skip_radix_cache_insert.py \
  test/registered/unit/mem_cache/test_radix_cache_cpp_unit.py \
  test/registered/unit/managers/test_io_struct.py \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  test/registered/unit/entrypoints/openai/test_serving_completions.py \
  test/registered/unit/entrypoints/test_http_server_warmup.py \
  test/registered/unit/managers/test_priority_scheduling_disaggregation.py \
  test/registered/unit/disaggregation/test_encode_receiver.py \
  test/registered/unit/mem_cache/test_session_token_share_unit.py
```

## Boundaries

- No numerical model-output gate was changed. The only numerical control is exact equality of a small SDPA reference output before and after normal cache release.
- Host-tier and HiCache behavior is untested here because no host cache service or model-weight-backed E2E server was exercised.
- The candidate's model-weight-backed E2E and PD runner tests were not run in this bounded job.
