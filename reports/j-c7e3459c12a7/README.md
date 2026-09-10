# Aiter sparse-MLA compiled-path reuse on MI300X

## Scope

This is the execution-representation and reuse follow-up to amdpilot-org/sglang issue 226 and sgl-project/sglang issue 34947. It does not repeat the already-covered 65,535-row trtllm guard from amdpilot-org/sglang pull request 326.

The uncovered case tested here is cold versus warm reuse of the supported Aiter sparse persistent BF16 MLA decode path across a finite shape sequence on one AMD Instinct MI300X (`gfx942`).

## Method

- Operation: `aiter.mla.mla_decode_fwd`
- Shape sequence: batch sizes `[1, 2, 4]`, 128 context tokens, 64-token top-k, 16 heads, `d_qk=576`, `d_v=512`, four KV splits.
- Each shape executes once cold and once warm.
- The output buffer is refilled with sentinel `-31337.0` before every call.
- An independent float32 PyTorch einsum/softmax reference checks every output.
- Host timing uses `time.perf_counter`; device timing uses CUDA events; both synchronize around each call.
- `torch.profiler` records actual native dispatch.
- The six persistent metadata buffers are allocated once at maximum batch size and their addresses are checked across the complete sequence.
- The output address is checked across cold and warm calls for each shape.
- The unsupported 24-head variant is required to raise before dispatch.

## Results

The first shape paid the compiled-kernel load cost:

| batch | pass | host seconds | device seconds | max abs error |
|---:|---|---:|---:|---:|
| 1 | cold | 0.0563796 | 0.0564778 | 1.45003e-4 |
| 1 | warm | 0.0004115 | 0.0004249 | 1.45003e-4 |
| 2 | cold | 0.0002596 | 0.0002717 | 1.46430e-4 |
| 2 | warm | 0.0001715 | 0.0001825 | 1.46430e-4 |
| 4 | cold | 0.0001988 | 0.0002090 | 1.49127e-4 |
| 4 | warm | 0.0001590 | 0.0001680 | 1.49127e-4 |

The later cold rows do not pay the first-shape kernel-load cost, showing compiled-path reuse across the finite shape sequence. All outputs pass `torch.testing.assert_close(atol=5e-4, rtol=5e-3)`, no sentinel remains, and the output and metadata addresses remain stable.

A second complete run reproduced the behavior: first-shape cold host/device times were 0.0574571/0.0575146 seconds, first-shape warm times were 0.0003908/0.0004058 seconds, and all numerical, sentinel, address, dtype, aliasing, dispatch, and unsupported-variant gates passed.

Actual native dispatch contains:

- `aiter::mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps`
- `kn_mla_reduce_v1<MlaReduceKernelV1Traits<512, 16, 1>, float, bfloat16>`

Contracts preserved:

- Q and KV remain BF16; output remains BF16; returned partial logits remain FP32.
- Partial logits do not alias the output storage.
- Persistent metadata addresses remain stable across all shapes.
- The unsupported 24-head variant raises `AssertionError: nhead=24 and max_seqlen_q=1 not supported` before dispatch.

## Installed-source baseline

Before cloning or editing, the installed source at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` ran the portable Triton sparse-MLA decode control:

- First GPU execution: 2.3651484539732337 seconds.
- Independent-reference maximum absolute error: `3.6734198463196485e-37`.
- Native dispatch: `_tiled_sparse_decode_kernel`.
- Full record: `/job/baseline-first.json` in the job workspace.

The relevant native sparse-MLA prefill and vertical/slash sparse-attention paths are unavailable in this ROCm build and were not forced:

- `sgl_kernel.flash_mla.flash_mla_sparse_fwd`: `ImportError: Failed to load sgl_kernel.flashmla_ops extension. Ensure CUDA Driver >= 12.4`
- `sgl_kernel.sparse_flash_attn.sparse_attn_func`: `torch.ops.sgl_kernel` has no `fwd_sparse` op.

## Reproduction

From the repository root on one MI300X with the qualified Torch/ROCm stack:

```bash
PYTHONPATH=python /opt/venv/bin/python \
  test/registered/kernels/benchmark/attention/bench_aiter_sparse_mla_reuse.py
```

The command prints the complete JSON timing matrix and native dispatch record.

## Limitations

- This is MI300X/ROCm evidence only; it does not execute or claim results for the SM100 trtllm-gen kernel from upstream issue 34947.
- Timings are one bounded run per cell, not a statistical benchmark.
- No model weights, toolchain replacement, unbounded stress, or GPU burn were used.
