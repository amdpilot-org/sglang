# QSA BF16-query/FP8-KV investigation on MI300X

## Result

- Ported the open upstream candidate sgl-project/sglang PR 36644 at commit `3df8e1e7dbc5807696622afe2929b6c33c185ca3` onto mirror `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- The focused candidate suite passed all 7 tests on one AMD Instinct MI300X (`gfx942`) in 17.91 seconds.
- An independent float32 dequantized reference matched the real Triton prefill and chunk-prefill paths within `7.8125e-3` maximum absolute error.
- The compact FP8-to-BF16 gather was exact, and cache-write admission passed scales `0.25` and `0.5` while preserving prefill K/V.
- The existing QSA backend suite passed 37 tests and skipped 1 in 18.16 seconds.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm: `7.2.26015-fc0010cf6a`
- GPU: AMD Instinct MI300X, capability `(9, 4)`, UUID `30643934-3739-6337-3231-353634353565`
- Checkout Python source: `/job/sglang/python/sglang`
- Native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`

## Installed-source baseline

The preinstalled source was commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` at `/sgl-workspace/sglang`. This baseline is environment context only and is not proof for later checkout changes.

The first relevant test command failed before GPU execution because its CUDA-only support check compares `torch.version.cuda` with a string:

```bash
cd /sgl-workspace/sglang
/opt/venv/bin/python -m pytest -q \
  python/sglang/kernels/aot/tests/test_flash_attn_sparse.py::test_sparse_attention \
  -k 'not backward' --tb=long
```

Exact error: `TypeError: '>=' not supported between instances of 'NoneType' and 'str'`; elapsed time was 9.464923 seconds.

Direct installed controls retained these exact native errors:

- Sparse attention: `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'fwd_sparse'`
- FA3 wrapper: `ImportError: Can not import FA3 in sgl_kernel. Please check your installation.`

The supported neighboring control was one small BF16 PyTorch SDPA call. It matched an independent float32 einsum/softmax reference with maximum absolute error `9.765625e-4` and completed in 0.004523352 seconds using synchronized `time.perf_counter`. Full details are in `/job/baseline-first.json`.

## Candidate commands

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python

/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/test_qsa_fp8_kv.py -v --tb=long

/opt/venv/bin/python -m pytest -q \
  test/registered/kernel/qsa/test_qsa.py --tb=short

/opt/venv/bin/python reports/j-1195380d7784/qsa_fp8_gpu_probe.py
```

The independent probe uses a separate float32 dequantized einsum/softmax implementation. It records one synchronized call per path:

- Prefill: maximum absolute error `7.8125e-3`, mean absolute error `4.8508745e-4`, elapsed 1.023618518 seconds.
- Chunk prefill: maximum absolute error `7.8125e-3`, mean absolute error `6.5134966e-4`, elapsed 0.024415033 seconds.
- Compact gather: exact K/V match, elapsed 0.004381541 seconds.

Raw values are retained in `reports/j-1195380d7784/qsa_fp8_gpu_results.json`.

## Architecture-specific limitations

- `_resolve_trtllm_sparse_decode()` returns `None` on gfx942 because the current gate accepts SM100/SM120, not CDNA gfx942.
- `_resolve_flash_attn_varlen_func()` raises `ImportError: QSA decode requires flash_attn (FA2) or flash-attn-4 (FA4 cute) for its packed varlen fallback.`
- Therefore, the real Triton prefill, chunk-prefill, compact-gather, and cache-write admission paths were exercised on gfx942, but an end-to-end native decode kernel was not available in this qualified stack.
- The candidate's decode admission test uses a mocked TRTLLM decode function to verify FP8 K/V and descale propagation; it is not an end-to-end gfx942 decode proof.
- The existing QSA indexer suite has 10 pre-existing failures on both mirror base and candidate. The TVM kernel rejects BF16 with `Dtype value [bfloat16] not in the allowed options: [float32]`; this is unrelated to the FP8-KV changes.

## Scope

- Read-only upstream context: sgl-project/sglang issue 36545 and PRs 36556 and 36644.
- No upstream issue, PR, or comment was posted or modified.
- No model weights were downloaded, no node-wide state was changed, and build/runtime artifacts stayed job-private.
