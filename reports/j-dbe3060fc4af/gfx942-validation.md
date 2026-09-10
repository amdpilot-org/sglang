# gfx942 validation for issue 35096

## Scope and identities

- Upstream context: sglang issue 35096 and candidate PR 35097, tested at commit `6a7fa215d172afcd66559dcb9751dbca4a403013`.
- Mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c` (`main`).
- Cherry-picked candidate on the mirror: `1c182d3622e6892d071090ae3d91bb0262c9ee8f`.
- Source: `/job/sglang/python/sglang/kernels/ops/attention/deepseek_v4_rope.py`.
- Python: `/opt/venv/bin/python`; checkout import: `/job/sglang/python/sglang/__init__.py`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; Triton: `3.7.0+amd.rocm7.2.0.git89002410`.
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440004306`, unique ID `0xb5c590cf4c10631d`, driver `6.19.14.31400000`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. The image ID is operator-provided; this container has no Docker socket, and hostname is not used as image identity.

## Installed-source first baseline

The first GPU baseline used the preinstalled source at `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, and native extension `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`. Its registration source is `/sgl-workspace/sglang/python/sglang/kernels/aot/csrc/common_extension_rocm.cc`.

The first reference attempt failed because the installed test helper kept `cos_sin_cache` on CPU while positions were on GPU:

```text
RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu
```

The supported control moved the helper module and cache to `cuda:0`. For `head_size=128`, `rotary_dim=64`, 512 tokens, 8 heads, FP16, NeoX style, and sentinel `-12345`, the independent PyTorch reference reported:

- Query/key rotated-column maximum absolute delta: `0.001953125`.
- Query suffix maximum absolute delta: `16.0`; key suffix: `32.0`.
- Query and key suffix sentinel preservation: false.

Timing used `time.perf_counter`, `torch.cuda.synchronize()` before stopping elapsed time, three warmup calls, and 20 steady-state calls on separate tensors. First GPU execution took `0.0008970163762569427` seconds; 20 steady iterations took `0.0001586340367794037` seconds. This installed-source baseline is recorded in `/job/baseline-first.json`; it is not proof for later checkout changes.

## Base-commit column control

The base control used a `[4, 8, 2 * rope_dim]` buffer with `-12345` suffix columns and a `[4, 8, rope_dim]` view. The independent reference computed interleaved complex RoPE in PyTorch FP32 and cast to the input dtype.

| `rope_dim` | padded `RD` | dtype | active max delta | suffix elements changed | changed suffix columns |
|---:|---:|---|---:|---:|---|
| 6 | 8 | FP16 | 0.0 | 64 | 0-1 |
| 6 | 8 | BF16 | 0.0 | 64 | 0-1 |
| 64 | 64 | FP16 | 0.0 | 0 | none |
| 64 | 64 | BF16 | 0.0 | 0 | none |
| 96 | 128 | FP16 | 0.0 | 1024 | 0-31 |
| 96 | 128 | BF16 | 0.0 | 1024 | 0-31 |
| 128 | 128 | FP16 | 9.5367431640625e-07 | 0 | none |
| 128 | 128 | BF16 | 0.0 | 0 | none |
| 192 | 256 | FP16 | 0.0 | 2048 | 0-63 |
| 192 | 256 | BF16 | 0.0 | 2048 | 0-63 |

Relative suffix column 0 corresponds to absolute column `rope_dim`. Thus the unfixed kernel changes exactly the columns from `rope_dim` through `next_power_of_2(rope_dim) - 1`; power-of-two widths have no padded suffix columns.

## Fixed-checkout result

With the candidate cherry-pick, the same control reported zero changed suffix elements for every width and both FP16 and BF16. Active-column maximum absolute deltas were 0.0 except `rope_dim=128` FP16 at `9.5367431640625e-07`.

The registered test command was:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-dbe3060fc4af/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-dbe3060fc4af/inductor
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_deepseek_v4_rope_flat.py -q --tb=short
```

Result: `40 passed, 1 warning in 14.62s`. The warning is pytest's existing unknown `asyncio_mode` config option. The test covers widths `{6, 64, 96, 128, 192}`, shapes `(4, 8)` and `(2, 4)`, FP32 sibling-kernel agreement, FP16/BF16 independent-reference agreement, and exact sentinel preservation.

## Architecture-specific limitations

- This result is for one MI300X (`gfx942`) under ROCm 7.2. It does not reproduce the NVIDIA B200 environment or timings in upstream PR 35097.
- `compute-sanitizer` is not installed in this image, so the upstream CUDA memory-check observation was not repeated. `rocprof` is available but was not needed for this numerical validation.
- FP16/BF16 comparisons use dtype-appropriate tolerances because both the kernel and reference round FP32 math to the input dtype.
- No model-level evaluation or full model weights were downloaded or run. The flat path remains opt-in, and shipped DeepSeek-V4 configurations use power-of-two widths.
