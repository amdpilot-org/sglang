# Independent review of candidate d38922c9

Upstream issue: https://github.com/sgl-project/sglang/issues/33268

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1921

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2028

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1993

## Verdict

Request changes. The candidate is a useful partial fix: it adds a dtype
namespace to file, NIXL, EIC, Mooncake, and hf3fs paths, and its hf3fs mock
round trips are symmetric. It does not fully satisfy the original issue's
requirement that incompatible KV formats cannot silently share a key.

The production configuration is populated from `mem_pool_host.dtype`, while
`HostKVCache` defines that value as `device_pool.store_dtype`. Both FP8 E4M3
and FP8 E5M2 use `torch.uint8` as their storage dtype. Consequently two runs
using those incompatible KV formats both receive the suffix
`_dtype_torch.uint8`. The candidate regression bypasses this path by directly
injecting distinct strings into `HiCacheStorageConfig`.

There is a second uncovered in-tree key path: `UMBPStore` constructs
`config_prefix` from the optional backend tag and model name, but not
`kv_cache_dtype`. Its hybrid component keys therefore remain dtype-agnostic.
The UMBP direct linker also constructs its own `HiCacheStorageConfig` without
the new field. External FlexKV and LMCache namespaces were not integration
tested, so no full-coverage claim can be made for them either.

## Evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the original
bf16-versus-fp8 intended-run reproducer generated the same file key and exited
1. At exact candidate `d38922c9de6874789c758a6931895eb142332a77`, the candidate's
18 focused tests and 24 related tests passed. The independent production-path
test then generated the same key for E4M3 and E5M2 and exited 1:

```text
e4m3 production key: page-1_model_0_1_dtype_torch.uint8
e5m2 production key: page-1_model_0_1_dtype_torch.uint8
threaded dtype: torch.uint8
collision: True
```

Imports resolved to `/job/repo/python/sglang/...`, not an installed SGLang
wheel. The candidate changes no C/C++/HIP sources, and the prepared environment
declares no native build target, so no native rebuild was required.

## Environment and limitations

The host exposes one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) through
PyTorch 2.11.0+rocm7.2 / HIP 7.2. The defect and tests are CPU-side namespace
construction and mocked local storage, so GPU execution would not strengthen
the relevant claim and was not used. No model weights, serving run, external
hf3fs/Mooncake/NIXL/EIC/UMBP service, multi-node workload, FlexKV service, or
LMCache service was available or claimed. The mocked hf3fs tests validate
local read/write key symmetry only.
