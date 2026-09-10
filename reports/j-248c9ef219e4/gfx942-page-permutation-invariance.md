# gfx942 physical KV page permutation invariance

## Scope

- Persistent mirror base: `amdpilot-org/sglang` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Upstream context: `sgl-project/sglang` issue 34718 and merged PR 34167 were read only.
- Prior mirror PR 337 covers AITER base numerical correctness; this investigation adds the distinct physical-page permutation invariant.
- No upstream issue, PR, or comment was posted or changed.
- No kernel code was changed because the supported permutation matrix showed no mismatch.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 206 GB HBM.
- Python: `/opt/venv/bin/python`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP: `7.2.26015-fc0010cf6a`
- AITER: `/sgl-workspace/aiter/aiter/__init__.py`
- AITER kernel: `/sgl-workspace/aiter/aiter/ops/triton/attention/pa_mqa_logits.py`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Persistent checkout: `/job/sglang`

## Installed-source baseline

`/job/baseline-first.json` records the first GPU execution before cloning or editing.

- Test: `test_kvcacheio.py::test_transfer_kv[False-False-10240-256-16-128-dtype0]`
- Source: `/sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_kvcacheio.py`
- Native wrapper: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/kvcacheio.py`
- Result: passed, `3.93 s` pytest time, `11.87826 s` wall time.
- Reference: independent indexed-copy reference plus `transfer_kv_per_layer` and `transfer_kv_direct`.
- Gate: `torch.testing.assert_close` on copied bfloat16 KV pools.
- This baseline is installed-source evidence only and is not proof for the later checkout.

Actual command:

```bash
cd /sgl-workspace/sglang/python && /opt/venv/bin/python -m pytest -q \
  'sglang/kernels/aot/tests/test_kvcacheio.py::test_transfer_kv[False-False-10240-256-16-128-dtype0]' \
  -rA --tb=short
```

## Real GPU permutation matrix

The persistent checkout used AITER's `deepgemm_fp8_paged_mqa_logits` with:

- `Preshuffle=False`
- `KVBlockSize=1`
- `ChunkK=128`
- `next_n=1`
- `num_heads=32`
- `HEAD_DIM=128`

Each case built a fresh FP8 query, KV cache, UE8M0 scale, and weight tensor. The physical fused KV pages were permuted out of place, and the page map was updated with the inverse permutation. The original and permuted kernels ran on the same query, weights, context length, and output width.

The independent reference dequantized FP8 keys and scales, computed per-head dot products, applied ReLU and head weights, and summed heads. The unchanged numerical gate was `atol=2e-2, rtol=2e-2` against that reference. The permutation invariant additionally required bitwise-equal logits and bitwise-equal untouched output boundaries.

| case | context | permutation | reference error | permuted reference error | exact logits | exact boundary | max delta |
|---|---:|---|---:|---:|---|---|---:|
| identity | 64 | identity | `7.62939453125e-06` | `7.62939453125e-06` | yes | yes | `0.0` |
| adjacent | 65 | swap 0/1, rest fixed | `1.52587890625e-05` | `1.52587890625e-05` | yes | yes | `0.0` |
| reverse | 129 | full reversal | `1.1444091796875e-05` | `1.1444091796875e-05` | yes | yes | `0.0` |
| derangement | 257 | modular multiply-by-7 | `1.52587890625e-05` | `1.52587890625e-05` | yes | yes | `0.0` |

The supported matrix passed in `18.409229 s` wall time, including Python startup and JIT loading.

Actual probe invocation:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-248c9ef219e4/tvm-ffi \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-248c9ef219e4/triton \
/opt/venv/bin/python - <<'PY'
# inline probe source
PY
```

The inline probe constructed each case, ran the original and permuted kernels, computed the independent dequantized reference, and asserted the three gates above. Its raw output is preserved in `/tmp/permutation-supported-final2.log`.

## Unsupported and negative boundaries

- Non-power-of-two head counts (`4`, `8`, `16`) failed AITER Gluon compilation with:

  ```text
  RuntimeError: Every element in sizePerThread must be a power of two.
  ```

- `next_n=2` and short contexts below `64` produced an asynchronous HIP launch failure:

  ```text
  HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
  torch.AcceleratorError: HIP error: unspecified launch failure
  ```

  The first short-context result appeared numerically exact before the asynchronous fault surfaced, so it is not counted as passing evidence.

- An exploratory permuted reference that advanced-indexed FP8 tensors directly produced a large reference error (`68.44007110595703`) while the kernel outputs remained bitwise equal. The final independent reference avoids FP8 advanced indexing and maps through the permutation using float32 dequantized values.

- DeepGEMM's direct `fp8_paged_mqa_logits` path remains architecture-gated to SM90/SM100 and does not run on gfx942.

## Reproduction

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TVM_FFI_CACHE_DIR=/tmp/sglang-cache-j-248c9ef219e4/tvm-ffi
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-248c9ef219e4/triton

/opt/venv/bin/python - <<'PY'
# The probe is recorded in this report and in /job/recovery.patch.
# It constructs the four supported cases above and asserts:
# 1. original and permuted logits match the independent reference within 2e-2;
# 2. original and permuted valid logits are bitwise equal;
# 3. untouched output boundaries are bitwise equal.
PY
```

## Limitations

- This is bounded single-GPU evidence, not a full-model or production eviction workload.
- No full model weights were downloaded and no node-wide state was modified.
- The invariant is established for AITER's supported `next_n=1`, 32-head, one-token-page path on gfx942.
- `next_n=2`, short contexts, non-power-of-two heads, production preshuffle, and 64-token-block configurations remain unsupported or unproven in this environment.
- No code change is claimed because no supported-case mismatch was demonstrated.
