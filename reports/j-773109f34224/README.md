# DSV4 paged top-k v2 MI300X investigation

## Scope and identity

- Campaign: `repo-e2e-20260909`.
- Upstream context: `sgl-project/sglang` issue `37892`; candidate PR `38016`.
- Mirror base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Tested upstream candidate head: `0041f7053addfbe154e67b47416c7e0b9d22fac3`.
- Candidate source commits: `cd8a84c932b0f9c4c0bef55e016cccadd61c40f3` and `c736638b173b4bb822de362e5d9fa0af257ddd82`.
- Delivery branch: `amdpilot/j-773109f34224`, rebased delivery head `b53707999daa6da414fe67a912da372ff5bbcaea`.
- Requested image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, `gfx942`, device ID `0x74a1`, GUID `9845`.
- Interpreter: `/opt/venv/bin/python` (Python 3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, HIP `7.2.26015-fc0010cf6a`.
- Delivery source import: `/job/sglang/python/sglang`.
- Native wheel paths: `/opt/venv/lib/python3.10/site-packages/sgl_kernel` and `/opt/venv/lib/python3.10/site-packages/tvm_ffi`.
- JIT cache: `/tmp/sglang-cache-j-773109f34224/jit`, outside `JOB_WORKDIR`.

## Early installed-source baseline

The first GPU execution started at `2026-09-10T03:19:35Z`. The complete artifact is `/job/baseline-first.json`; it is explicitly an installed-source baseline and is not evidence for later checkout changes.

The installed source was `/sgl-workspace/sglang` at `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. Its existing test was:

```bash
cd /sgl-workspace/sglang
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_topk_v2.py \
  -q --maxfail=1 --durations=5
```

This failed in 25 seconds because the v2 JIT build could not find `cooperative_groups.h` under ROCm 7.2. The installed AOT v1 op was also unavailable in the installed `sgl_kernel` wheel. The supported neighboring control was installed Torch top-k on the same MI300X:

- Fixture: 4 rows, 32768 columns, `k=2048`, float16.
- Timing: three warmups, ten timed launches, CUDA events, mean `0.04078600108623505 ms`.
- Independent reference: stable descending float32 `torch.argsort` plus gathered values.
- Result: indices equal, values equal, maximum absolute error `0.0`.

The current mirror checkout guards `cooperative_groups.h` for ROCm. A real current-checkout v2 case, `test_topk_v2[1-65537-512-identity]`, passed on MI300X in 13.52 seconds.

## Candidate and delivery validation

All commands below used:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-773109f34224/jit \
/opt/venv/bin/python -m pytest ...
```

The exact upstream candidate head passed the focused dispatch and long-row dual-output cases:

```bash
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_dsa_indexer.py::test_c4_raw_indices_keep_sgl_backend_on_topk_v2 \
  'test/registered/kernels/ops/attention/test_topk_v2.py::test_topk_v2_paged_and_raw_outputs[2-65537-512]' \
  'test/registered/kernels/ops/attention/test_topk_v2.py::test_topk_v2_paged_and_raw_outputs[2-65537-2048]' \
  'test/registered/kernels/ops/attention/test_topk_v2.py::test_topk_v2_paged_and_raw_outputs[31-131072-512]' \
  'test/registered/kernels/ops/attention/test_topk_v2.py::test_topk_v2_paged_and_raw_outputs[31-131072-2048]' \
  -q --maxfail=1 --durations=10 -p no:cacheprovider
```

Result: 5 passed in 22.52 seconds. The same five cases passed after cherry-picking onto the mirror delivery branch in 23.55 seconds.

The full affected kernel suite was:

```bash
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_topk_v2.py \
  -q --durations=15 -p no:cacheprovider
```

Results:

- Exact upstream candidate head: 297 passed in 25.77 seconds.
- Rebased delivery head: 297 passed in 27.66 seconds.

The dual-output test inverts the page-table transform and requires exact slot alignment between raw and paged outputs. It also compares raw selected indices with an independent `torch.topk` reference, permits only equal-score tie swaps, and includes a zero-length DP row and a tie-heavy row.

Python syntax checks passed for the four changed Python files. `git diff --check` passed. `ruff` is not installed in this image, so no local Ruff run was made.

## Limitations and negative controls

- The complete `test_dsa_indexer.py` run is not a usable gate on this image: 38 tests failed and 9 passed because `flashinfer` is not installed. A broader non-FlashInfer attempt also encountered missing `deep_gemm` mocks/imports. The focused dispatch regression passed.
- MI300X (`gfx942`) compiles out the CUDA cluster path. The `65537` and `131072` fixtures exercise ROCm streaming dispatch, not the GB300/SM120 cluster paths named in the upstream issue.
- No full DeepSeek-V4-Pro model weights were downloaded and no end-to-end serving benchmark was run. This investigation is bounded to the real paged top-k kernel and dispatch fixtures.
- The installed-source v1 AOT operator is absent from the installed wheel, so no v1-versus-v2 AOT comparison was possible without rebuilding the qualified stack.
- The upstream issue reports an SM120/GB300 illegal-memory-access reproduction. That exact hardware fault was not reproduced here and remains architecture-specific.

