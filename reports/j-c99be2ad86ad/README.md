# gfx942 contiguous versus paged top-k property

## Result

No implementation mismatch was demonstrated, so this report makes no production-code change.

On one AMD Instinct MI300X (`gfx942:sramecc+:xnack-`), a finite adversarial score matrix was run through all three relevant v2 representations:

- contiguous ragged top-k,
- paged top-k with raw indices,
- paged top-k with a page-table transform.

All three produced the exact independent top-k index set and the same selected-value multiset. Dispatch recording confirmed both eligible transform entry points (`topk_transform_ragged` and `topk_transform_paged`) were reached.

## Reproduction

Environment:

- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Checkout: `/job/sglang`, branch `amdpilot/j-c99be2ad86ad86ad`, base `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Source import: `/job/sglang/python/sglang`
- Native JIT module: `/tmp/sglang-cache-j-c99be2ad86ad/jit/gfx942/sgl_kernel_jit_dpsk_v4_topk_v2/build-0b3fcfa0a1167094/deps-7c84bba947901559/sgl_kernel_jit_dpsk_v4_topk_v2.so`

Adversarial property probe:

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-c99be2ad86ad/jit \
/opt/venv/bin/python /job/evidence/probe_contiguous_paged.py
```

Result:

- `batch=4`, `seq=65537`, `k=512`, `page_size=64`
- finite, unique scores with winners on page starts, page ends, tail positions, and the final position
- contiguous, raw-paged, and page-transformed outputs all matched the independent CPU stable-sort reference
- selected-value multisets were preserved across all three representations
- dispatch calls included `topk_transform_ragged` and `topk_transform_paged`

Full affected suite:

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-c99be2ad86ad/jit \
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_topk_v2.py \
  -q --durations=15 -p no:cacheprovider
```

Result: `278 passed, 1 warning in 26.44s`.

## Early installed-source baseline

The first GPU execution used the preinstalled AITER stable per-row top-k test:

```bash
/opt/venv/bin/python /sgl-workspace/aiter/op_tests/test_topk_per_row_stable.py
```

Result: all 48 cases passed against an independent CPU reference in `12.438206347s`. The complete artifact is `/job/baseline-first.json`; it is installed-source evidence only and is not proof for this checkout.

## Limitations

- `gfx942` exercises the ROCm streaming path, not the SM120/GB300 cluster path from upstream issue 37892.
- No full model weights were downloaded and no end-to-end serving run was performed.
- The actual DSV4 paged-prefill `raw_indices` dispatch condition remains outside this kernel-level property and is already covered by open mirror PR 307; this report does not duplicate that work.
