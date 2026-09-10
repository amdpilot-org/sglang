# Unified extend attention output-reuse investigation

## Scope

This investigation follows sgl-project/sglang issue 32942 and amdpilot-org/sglang
issue 205, but does not repeat the original xAI temperature trigger. The new case
checks fresh output storage against the documented piecewise-graph
`ForwardBatch._attn_output` reuse pattern across two distinct input batches.

The experiment uses one AMD Instinct MI300X (gfx942), synthetic tensors only, no
model weights, no toolchain replacement, and no native rebuild.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: AMD Instinct MI300X, capability `(9, 4)`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Triton: `3.7.0`
- Delivery base: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Kernel source: `python/sglang/kernels/ops/attention/extend_attention.py`
- Native dispatch: Triton JIT `sglang.kernels.ops.attention.extend_attention._fwd_kernel_unified`
- Prebuilt native module for this operation: none; the operation is Python/Triton JIT

## Installed-source baseline

Command:

```bash
PYTHONPATH=/sgl-workspace/sglang/python /opt/venv/bin/python -m pytest \
  /sgl-workspace/sglang/test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention_unified_vs_regular \
  -q --disable-warnings
```

Result:

```text
1 failed, 1 passed, 4 warnings, 2 subtests passed in 36.23s
```

The `(B=4, N_CTX=512, H_Q=32, H_KV=8, D=128)` and
`(B=2, N_CTX=2048, H_Q=32, H_KV=8, D=128)` configurations passed. The
nonstandard `(B=8, N_CTX=256, H_Q=64, H_KV=8, D=80)` configuration failed the
existing tolerance with max absolute difference `0.1669921875`. Wall-clock time
around the first GPU execution was 39 seconds. This result is labeled
installed-source evidence and is not proof for checkout changes.

## Reuse experiment

The registered test constructs two distinct batches in the same `q`, `k_buffer`,
and `v_buffer` storage. It compares:

1. A fresh output allocation.
2. A fixed-address output allocation reused across both batches.
3. An independent float32 Torch reference implemented in
   `reports/j-1be069ef8d8a/reuse_timing.py`.

Both output paths are protected with a NaN sentinel before every dispatch. The
test checks that all active output elements are finite, no sentinel remains,
fresh and reused outputs are identical, and both match the independent
reference within `rtol=0.05, atol=0.05`. It also checks that `q`, `k_buffer`,
`v_buffer`, and the reused output retain their addresses across both batches.

The supported dtype case intentionally uses `float16` queries with `bfloat16`
keys, values, and output. This preserves the documented behavior that query
dtype may differ while output follows the value dtype.

### Numerical result

```text
test_extend_attention_unified_output_reuse: 1 passed in 18.67s
```

All three timing configurations and both input batches passed the independent
reference comparison. Maximum absolute differences were `0.0078125` or
`0.015625`, within the `0.05` absolute tolerance.

## Bounded timing matrix

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-1be069ef8d8a/reuse_timing.py
```

Method: one warmup per mode, then five samples. Storage time uses host
`perf_counter` around allocation-or-reuse, NaN fill, kernel, and synchronize.
Kernel time uses CUDA events around only the Triton dispatch. Raw samples and
fixed addresses are in `reports/j-1be069ef8d8a/results.json`.

Median results:

| Config | Batch | Fresh storage | Reused storage | Fresh kernel | Reused kernel |
|---|---:|---:|---:|---:|---:|
| small | 1234 | 0.1368 ms | 0.0998 ms | 0.0841 ms | 0.0638 ms |
| small | 5678 | 0.1096 ms | 0.0963 ms | 0.0680 ms | 0.0605 ms |
| medium | 1234 | 0.1078 ms | 0.0942 ms | 0.0673 ms | 0.0605 ms |
| medium | 5678 | 0.1064 ms | 0.0950 ms | 0.0659 ms | 0.0594 ms |
| large | 1234 | 0.1397 ms | 0.0955 ms | 0.0745 ms | 0.0603 ms |
| large | 5678 | 0.1112 ms | 0.0957 ms | 0.0693 ms | 0.0600 ms |

The first-call matrix before warmup showed Triton compilation spikes of about
820 ms for the small case and 2.0 ms for the large case. These are recorded in
the job log and are excluded from the steady-state medians above.

## Dtype boundary

Before the wrapper change, a `float32` output with `bfloat16` values was
silently accepted and produced finite output. That violated the documented
output-dtype contract. The wrapper now derives the expected output dtype from
`v_buffer`, mapping FP8 values to `bfloat16`, and raises a clear `ValueError`
for unsupported output dtypes.

Negative control after the change:

```text
float32_output_result rejected ValueError Unified extend attention output dtype must be torch.bfloat16, got torch.float32
```

## Candidate and regression evidence

Upstream PR 32943 commit
`79dfba5440fc1a1f33088d30ba4d8907f9fd36af` was tested read-only in a separate
worktree. Its focused `xai_temperature_len=4` case passed on gfx942 in 19
seconds. The same focused case also passed on mirror main within the registered
AMD tolerance, so this investigation does not duplicate or claim to validate
that fix.

The existing unified-vs-regular registered test was rerun after the dtype
validation change. It reproduced the same pre-existing D=80 tolerance failure
with max absolute difference `0.1669921875`; the two D=128 configurations
passed. No attempt was made to force or mask that unrelated limitation.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest \
  test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention_unified_output_reuse \
  -q --disable-warnings

PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-1be069ef8d8a/reuse_timing.py
```
