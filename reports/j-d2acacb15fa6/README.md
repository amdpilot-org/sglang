# DSA kpool gfx942 control

## Scope

This is a ROCm architecture control for the CUDA-reported kpool/FP8-KV issue in
upstream issue 36830. It does not change or claim SM90 support. The upstream
thread records PR 36904 as already validated on SM90; that result is context
only and was not reproduced here.

## Environment

- Mirror base: `amdpilot-org/sglang` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, gfx942, device ID `0x74a1`, serial `692440004372`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, version `2.9.1+rocm7.2.0.git7e1940d4`
- AITER: `/sgl-workspace/aiter/aiter/__init__.py`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Working source: `/job/sglang`
- Job-private JIT cache: `/tmp/sglang-cache-j-d2acacb15fa6`

## Reproduction

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-d2acacb15fa6 \
/opt/venv/bin/python -m pytest -q \
  test/registered/amd/test_dsa_kpool_rocm_control.py
```

The test uses synthetic `index_kpool=1` and `index_kpool=4` configurations. It
checks the fused kpool history/tail expansion against an independent selected-set
and page-table reference, then runs the real AITER `mla_a16w8` decode kernel with
BF16 queries and an FP8 KV cache against a gather-plus-softmax reference.

## Raw result

All three tests passed. Numerical gates remain `atol=0.05, rtol=0.05`.

- `index_kpool=1`: max absolute error `0.00390625`, RMSE `0.0007596093346364796`
- `index_kpool=4`: max absolute error `0.0078125`, RMSE `0.0009839615086093545`
- `index_kpool=4` tail token row 0: `[12]`
- `index_kpool=4` tail token row 1: `[1041]`

The `index_kpool=4` AITER probe is deliberately kernel-level. The production DSA
backend still rejects that combination before attention, with:

```text
index_kpool > 1 appends tail tokens to topk_indices and is currently only
supported by the FA3/TileLang/TRTLLM DSA decode backend.
```

The new admission test pins this exact message. No end-to-end AITER kpool route
is enabled by this change.
