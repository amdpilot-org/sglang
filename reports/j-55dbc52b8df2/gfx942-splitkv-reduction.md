# gfx942 split-KV reduction/merge baseline

## Environment

- Campaign: `repo-e2e-20260909`
- GPU: one AMD Instinct MI300X, `gfx942`, 304 CUs.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Triton: `3.7.0`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Kernel source: `python/sglang/kernels/ops/attention/verify_splitkv.py`
- New test: `test/registered/attention/test_verify_splitkv_reduction.py`

## Installed-source first GPU baseline

- Source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed kernel: `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/verify_splitkv.py`
- Installed test: `/sgl-workspace/sglang/test/registered/attention/test_verify_splitkv.py::TestVerifySplitKV::test_numerics_head_dim_256`
- Command: `/opt/venv/bin/python -m pytest -q /sgl-workspace/sglang/test/registered/attention/test_verify_splitkv.py::TestVerifySplitKV::test_numerics_head_dim_256`
- Result: passed, `38.132106` seconds process wall time.
- Reference: installed `extend_attention_fwd` unsplit Triton reference.
- Gates: `atol=2e-2`, `rtol=1e-2`.
- Artifact: `/job/baseline-first.json`

## Current-checkout independent reduction/merge result

Command:

```bash
/opt/venv/bin/python -m pytest -q -s test/registered/attention/test_verify_splitkv_reduction.py
```

Reference:

- Independent fp32 Torch implementation, not `extend_attention_fwd`.
- Prefix attention uses full visibility and applies `k_scale` to scores and `v_scale` to output.
- Draft attention uses an explicit causal mask.
- Prefix and draft are merged with explicit stable LSE weights.
- GQA maps each query head to `head // (h_q / h_kv)`.

Bounded shape:

- Prefix lengths: `[127, 4127, 8223]`
- Draft length: `4`
- Query heads: `8`
- KV heads: `2`
- Head/value dimensions: `128`
- dtype: `bfloat16`
- `k_scale=0.5`, `v_scale=0.25`

Unchanged numerical gates:

- `atol=2e-2`
- `rtol=1e-2`

Raw results:

| `n_splits` | Max abs diff | Warm median latency |
|---:|---:|---:|
| 4 | `0.000488281` | `0.153714 ms` |
| 8 | `0.000488281` | `0.109733 ms` |
| 16 | `0.000488281` | `0.087402 ms` |

Timing method:

- One warmup call, then five timed calls per split count.
- `torch.cuda.Event(enable_timing=True)` start/end events.
- Median of the five elapsed times, in milliseconds.
- No synthetic burn, unbounded loops, sleep loops, or extra repeated work.

## Upstream context

- Read-only issue: `sgl-project/sglang` issue `35003`.
- Related open PR: `sgl-project/sglang` PR `35521`.
- Reviewed candidate commits:
  - `f707b473c0c1a92b984661774928cd16a7284035`
  - `0f96eb81f97c6bcbbb66e8f4967361860f7b97cf`
  - `f1c66fde16c0b4323cbbe45c21c969b072ef11d8`
- The candidate already covers gfx942 enablement, CU-aware split caps, and bidirectional draft blocks; this work does not duplicate those changes.
