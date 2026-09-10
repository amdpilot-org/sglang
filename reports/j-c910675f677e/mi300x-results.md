# MI300X shared-prefix/tail attention decomposition results

## Environment

- Job: `j-c910675f677e`
- Campaign: `repo-e2e-20260909`
- GPU: one assigned AMD Instinct MI300X (`gfx942`, 304 CUs, 206,141,652,992 bytes HBM)
- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Installed SGLang source: `/sgl-workspace/sglang/python/sglang`
- Installed SGLang revision: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- Delivery mirror: `/job/sglang`
- Delivery branch: `amdpilot/j-c910675f677e`
- PR base: `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`

## First installed-source GPU baseline

The first real GPU execution was the installed-source LSE merge test:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_merge_state_v2.py::test_merge_attn_states[output_dtype0-32-8-256] -s
```

- First GPU execution elapsed time: `9,988 ms`
- Native `merge_state_v2` result: unsupported on this stack
- Concrete error: `AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'merge_state_v2'`
- The test reached the MI300X and then failed with `ZeroDivisionError` because the native kernel returned zero timing.
- This installed-source baseline is not proof for later checkout changes.

A bounded neighboring control used Torch SDPA against an independent float32 manual softmax reference:

```bash
/opt/venv/bin/python /tmp/sglang_baseline_first.py
```

- Shape: `[2, 4, 64, 128, 64]`, `float16`
- Independent reference: manual `softmax(QK^T*scale)V` in float32
- Max absolute error: `0.00048828125`
- Mean absolute error: `0.000019541786969057284`
- Gate: `atol=0.001`, `rtol=0.001`, passed
- Timing method: CUDA events, 3 warmups, 20 timed iterations
- Average time: `0.025583100132644178 ms`

Raw baseline artifact: `/job/baseline-first.json`

## Upstream context and candidate set

- Upstream issue: `sgl-project/sglang` issue `1715`
- Stale upstream PR: `sgl-project/sglang` PR `5206`
- Newer open upstream PR: `sgl-project/sglang` PR `29288`
- Preserved candidate head: `1e377f0301d224b63db6ddcc0212ed58d92af146`

The candidate backend import succeeded on this ROCm stack, but its only available test launches a model server and would download model weights. It was stopped before health/download completion to respect the no-full-weights constraint. No upstream issue, PR, or comment was posted or modified.

## Delivery checkout results

### Correctness

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/attention/test_shared_prefix_lse_decomposition.py -s
```

Result:

```text
3 passed, 1 warning in 10.78s
```

Bounded cases:

- `[2, 128, 16, 4, 64]`, `float16`
- `[4, 256, 32, 8, 64]`, `bfloat16`
- `[8, 512, 64, 8, 64]`, `float16`

Each case compares a shared-prefix/tail decomposition with a full-attention independent reference and uses a stable pure-Torch LSE combination.

### Timing

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  test/manual/attention/bench_shared_prefix_lse_decomposition.py
```

Result:

```json
{
  "decomposed_attention_average_ms": 0.4790053993463516,
  "full_attention_average_ms": 0.22092385068535805,
  "gpu": "AMD Instinct MI300X",
  "max_abs_error": 0.0001220703125,
  "shape": {
    "batch_size": 8,
    "dtype": "float16",
    "head_dim": 64,
    "num_heads": 8,
    "prefix_len": 512,
    "tail_len": 64
  },
  "timed_iterations": 20,
  "warmup_iterations": 3
}
```

The small pure-Torch decomposition is slower than full attention on this shape. This is expected because it performs two score/softmax passes plus an LSE merge; it is not a production cascade backend.

## Architecture-specific limitations

- The installed native `sgl_kernel.merge_state_v2` op is missing on this ROCm stack.
- The in-tree Triton `merge_state` fallback produced incorrect output on MI300X for this decomposition; the maximum observed difference versus a stable manual merge was approximately `0.2037`.
- The test therefore uses a pure-Torch LSE combination, which is supported on MI300X and matches the independent full-attention reference.
- FlashInfer is not installed in this qualified image, so the upstream cascade backend could not be exercised end-to-end.
- No model weights were downloaded and no server was kept running.
