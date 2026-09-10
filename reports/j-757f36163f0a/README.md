# gfx942 KV specialization investigation

## Result

This is a report-only change. Upstream pull request `sgl-project/sglang#31689`
already removes the demonstrated redundant `N: tl.constexpr` axis from
`masked_set_kv_buffer_kernel`, and its current head passes on the assigned
AMD Instinct MI300X (`gfx942`). Duplicating that working kernel change here
would create an unnecessary second fix, so no production code was modified.

The tested kernel commit is:

```text
adfec536845af8bcca46323344ea5c91b5a60cbe
[Kernel] Avoid batch-size specialization for masked KV writes
```

The current upstream PR head is:

```text
8645d578414c6b4e06d7db30089c8b631c20f4ec8
Merge branch 'main' into codex/fix-masked-kv-constexpr
```

The PR-head regression test was run with:

```bash
cd /job/sglang-pr-head
TRITON_CACHE_DIR=/tmp/sglang-cache-j-757f36163f0a/pr-head \
PYTHONPATH=/job/sglang-pr-head/python \
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/test_masked_set_kv_buffer.py
```

Result:

```text
1 passed, 3 warnings in 18.07s
```

## Controlled comparison

The mirror base was commit
`0084030179bfba86bfeb6d43f7997d4076329d2c`. The candidate was the exact
upstream kernel commit above. Both used the same `N=17` then `N=33` workload,
`float16` K/V tensors, `int64` locations, `int32` masks, `H=2`, `D=8`,
`CHUNK=16`, and separate job-private Triton cache directories.

Timing used `time.perf_counter()` around one cold launch followed by
`torch.cuda.synchronize()`, then ten bounded warm launches with the median
reported. The independent reference assigned selected rows with ordinary PyTorch
indexing and asserted that masked-out rows remained untouched.

| Source | N | Cache entries | Cold ms | Warm median ms | Max K/V error |
|---|---:|---:|---:|---:|---:|
| Mirror base | 17 | 1 | 770.024 | 0.045324 | 0 / 0 |
| Mirror base | 33 | 2 | 39.178 | 0.039257 | 0 / 0 |
| Candidate | 17 | 1 | 810.786 | 0.037432 | 0 / 0 |
| Candidate | 33 | 1 | 0.064277 | 0.036145 | 0 / 0 |

The base cache keys differ only by the reported nonsemantic `constexpr` values
`17` and `33`. The candidate key omits that axis and remains identical across
both grid sizes. The candidate's second launch is therefore warm rather than a
new specialization, while all selected and untouched output assertions remain
exact.

## Installed-source baseline

Before cloning or editing, the installed source at
`/sgl-workspace/sglang/python/sglang/__init__.py` was used for an early
`N=17` then `N=31` baseline. This installed-source baseline is context only
and is not proof for later checkout changes.

- Installed source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Baseline command: `/opt/venv/bin/python /tmp/sglang-baseline.py`
- Baseline cache: `/tmp/sglang-cache-j-757f36163f0a/baseline`
- `N=17`: one entry, 770.780 ms cold, 0.040091 ms warm median, exact K/V
- `N=31`: two entries, 48.147 ms cold, 0.038617 ms warm median, exact K/V

The complete early record is in `results.json` and was also written outside the
repository as `/job/baseline-first.json`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 CUs
- GPU UUID: `66373066-3466-6665-6339-613163306536`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Triton: `3.7.0`
- Torch Python package: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch ROCm libraries include `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Triton Python package: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- `sgl_kernel` Python package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- `sgl_kernel` native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Mirror checkout: `/job/sglang`
- Candidate worktree: `/job/sglang-candidate`
- PR-head worktree: `/job/sglang-pr-head`

## Architecture limitations

The issue also reports an unused `NT_BUCKET` specialization in the XPU FLA
kernel `chunk_gated_delta_rule_fwd_kernel_h_blockdim64_k_loop`. That backend is
for Intel XPU and was not executed on the assigned `gfx942` GPU. The module
imports successfully, but a meaningful XPU launch/control would require an
Intel XPU device and its qualified runtime; claiming a `gfx942` result for that
axis would be invalid. Static inspection confirms `NT_BUCKET` appears in the
autotune key, kernel signature, and call-site bucket calculation, but not in the
kernel body.

No upstream issue, pull request, or comment was posted or modified.
