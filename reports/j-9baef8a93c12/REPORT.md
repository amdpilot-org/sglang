# Packed QKV projection parity on MI300X

## Scope

This report covers the packed vision QKV projection representation only. It does not test MLA attention algebra, cache writers, or model weights.

Upstream context is sgl-project/sglang issue 16255, whose DeepSeek refactor checklist led to the exact projection-layout change in sgl-project/sglang PR 35336. That PR is already merged as commit `0be3039aa683afc150c8f385704125e70e01434f`; the mirror contains it as commit `c0c87e05477d84584cb8f1a2af667b679604675b`. No upstream issue, PR, or comment was posted or changed.

## Environment

- GPU: one AMD Instinct MI300X, gfx942 (`torch.cuda.get_device_capability() == (9, 4)`).
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python` (realpath `/usr/bin/python3.10`).
- Torch/ROCm: `torch 2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Checkout: `/job/sglang`, branch `amdpilot/j-9baef8a93c12`, base `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Native package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`.
- Build cache: `/tmp/sglang-cache-j-9baef8a93c12` (no build was required).

## Reproduction

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/layers/attention/test_vision_strided_qkv.py
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-9baef8a93c12/gpu_packed_qkv_parity.py \
  --output reports/j-9baef8a93c12/results.json
```

The harness compares one packed projection followed by `split`/`reshape` against three independent `torch.nn.functional.linear` projections using the corresponding packed-weight rows. It checks q/k/v row order, storage aliasing, last-dimension contiguity, token strides, storage offsets, MHA grouped-view equivalence, and bounded CUDA-event timing. Cases cover MHA and GQA with head dimensions 8, 64, 72, 80, and 128, including a 1023-token tail shape.

## Result

The already-merged fix is present and working. The focused checkout test passes: `14 passed`. All eight parity cases alias the same packed storage, preserve q/k/v row order, keep the head dimension contiguous, and have the expected q/k/v storage offsets. They match the independent projections under the unchanged numerical gate `torch.allclose(atol=0.05, rtol=0.05)`.

Five bf16 cases are bit-identical to the independent projections. The remaining bf16 GQA-80 case differs by at most `0.015625`; fp16 GQA-80 differs by at most `0.00390625`, while fp16 GQA-128 is bit-identical. These are GEMM-shape-dependent differences between one packed and three independent calls, not layout failures.

For GQA-128 (`1023` tokens, hidden `1024`, 16 Q heads, 4 KV heads, head dim `128`), CUDA-event means over 10 iterations were:

- bf16 packed projection + views: `0.0726995 ms`; independent projections: `0.1250201 ms`; packed projection + dense copies: `0.0919800 ms`.
- fp16 packed projection + views: `0.0511259 ms`; independent projections: `0.1113606 ms`; packed projection + dense copies: `0.1006078 ms`.

Because the fix is already in `main`, this branch makes no product-code change and does not duplicate it.

Raw machine-readable results are in `results.json`. The installed-source baseline is preserved outside the repository at `/job/baseline-first.json`; it is evidence for the pre-clone environment only, not for later checkout changes.
