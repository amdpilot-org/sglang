# Triton PTX inline-asm investigation on MI300X

## Conclusion

- Current mirror `main` at `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` is already fixed at the model dispatch layer on ROCm: both `can_use_fused_layernorm_modulate` and `can_use_fused_qk_head_layernorm` return `False`. The reported FLUX warmup path therefore does not launch these PTX-using Triton kernels on gfx942.
- The five PTX helpers themselves remain unsupported if called directly on gfx942. Every one fails with the verbatim error: `error: couldn't allocate output register for constraint 'f'`.
- Upstream PR 34481 (merge commit `e805a8f98e69797d88e6e8788186ed46b00b9312`) and closed PR 34352 (head `533d8f887fbe83740fb240c21dbc91c6b678d19c`) both disable the fused fast path on ROCm. They do not make direct PTX helper compilation work. Current `main` retains the equivalent dispatch guard after the later file move.
- No speculative source fix was authored. This report and the bounded reproduction harness are the only changes.

## Scope

The investigation followed read-only upstream issue 34351 and its linked PRs 34352 and 34481. No upstream issue, PR, or comment was changed. No FLUX or other model weights were downloaded.

The affected operation names exactly as they appear in source are:

- `mul_rn_f32`
- `div_rn_f32`
- `rsqrt_approx_f32`
- `cuda_rsqrtf`
- `_rcp4`

The normalization entry points tested are:

- `fused_layernorm_modulate_raw`
- `fused_qk_head_layernorm`
- `can_use_fused_layernorm_modulate`
- `can_use_fused_qk_head_layernorm`

The runnable numerics helper tested is `round_bf16_to_fp32`.

## Environment

- Working clone: `/job/sglang`
- Current source commit: `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Current numerics source: `/job/sglang/python/sglang/kernels/ops/diffusion/common/numerics.py`
- Current normalization source: `/job/sglang/python/sglang/kernels/kda_kernels/layernorm_modulate_triton.py`
- Candidate 34481 worktree: `/job/sglang-candidate-34481`, commit `e805a8f98e69797d88e6e8788186ed46b00b9312`
- Candidate 34481 normalization source: `/job/sglang-candidate-34481/python/sglang/kernels/ops/diffusion/norm/layernorm_modulate_triton.py`
- Candidate 34352 worktree: `/job/sglang-candidate-34352`, commit `533d8f887fbe83740fb240c21dbc91c6b678d19c`
- Candidate 34352 normalization source: `/job/sglang-candidate-34352/python/sglang/kernels/ops/diffusion/triton/layernorm_modulate.py`
- Python: `/opt/venv/bin/python` (Python 3.10.12)
- Torch Python package: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Triton Python package: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Triton: `3.7.0`
- Qualified image (operator-provided local image ID): `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Hostname (not used as image identity): `banff-cyxtera-cx57-5`
- GPU: one AMD Instinct MI300X, gfx942, serial `692440004395`, GUID `9845`, card model `0x74a1`
- Job-private Triton caches were under `/job/triton-cache-*`; no node-wide cache or state was modified.

## Reproduction

Clone and baseline:

```bash
git clone --depth 50 https://github.com/amdpilot-org/sglang.git /job/sglang
cd /job/sglang
git rev-parse HEAD
rocm-smi --showproductname --showserial
```

Current-source direct helper compile test:

```bash
cd /job/sglang
./reports/j-1f1a580e9886/run_repro.sh \
  /job/sglang \
  /job/repro-current-34351 \
  /job/triton-cache-current-34351
```

Current-source normalization and semantics test:

```bash
cd /job/sglang
./reports/j-1f1a580e9886/run_norm.sh \
  /job/sglang \
  /job/repro-norm-current-34351 \
  /job/triton-cache-norm-current-34351
```

Candidate tests use the same harness with the candidate tree first on `PYTHONPATH`:

```bash
cd /job/sglang
./reports/j-1f1a580e9886/run_candidate.sh \
  /job/sglang /job/sglang-candidate-34481 34481 \
  /job/repro-candidate-34481 /job/triton-cache-candidate-34481
./reports/j-1f1a580e9886/run_candidate.sh \
  /job/sglang /job/sglang-candidate-34352 34352 \
  /job/repro-candidate-34352-retry /job/triton-cache-candidate-34352-retry
```

Each PTX helper and each direct fused wrapper runs in its own subprocess because the LLVM AMDGPU failure is a hard abort, not a catchable Python exception.

## Raw compile outcome

At current `main`, all five direct helper kernels used finite 256-element fp32 tensors and exited with status 1:

```text
mul_rn_f32: error: couldn't allocate output register for constraint 'f'
div_rn_f32: error: couldn't allocate output register for constraint 'f'
rsqrt_approx_f32: error: couldn't allocate output register for constraint 'f'
cuda_rsqrtf: error: couldn't allocate output register for constraint 'f'
_rcp4: error: couldn't allocate output register for constraint 'f'
```

Direct calls to `fused_layernorm_modulate_raw` and `fused_qk_head_layernorm` also exit with the same verbatim error. This confirms the platform-specific direct-helper failure. It does not reproduce the reported FLUX warmup crash through current model dispatch because the `can_use_*` guards reject ROCm first.

Both candidate revisions have the same five direct-helper failures. Their relevant guards return:

```text
can_use_fused_layernorm_modulate=False
can_use_fused_qk_head_layernorm=False
```

## Numerical gates and raw results

The synthetic normalization tensors were finite bf16 tensors with shapes:

- `x`: `(2, 3, 256)`
- `scale`, `shift`: `(2, 256)`
- `q`, `k`: `(2, 3, 8, 64)`

The independent explicit reference computed fp32 mean, biased variance, `rsqrt(variance + eps)`, converted normalized values to bf16, and applied the modulation in Torch. A second Torch reference used `torch.nn.functional.layer_norm`. The fallback path compared was the eager Torch chain used when the fused fast path is unavailable.

Unchanged gates and results at current `main`, candidate 34481, and candidate 34352:

```text
ln_modulate_bitwise_equal_to_torch_reference=True
ln_modulate_bitwise_equal_to_explicit_reference=True
ln_modulate_allclose_2e-2_2e-2=True
ln_modulate_max_abs_diff=0
ln_modulate_nan_mask_equal=True
ln_modulate_nan_count=256
ln_modulate_signed_zero_mask_equal=True
ln_modulate_negative_zero_count=760
qk_ln_bitwise_equal_to_explicit_reference=True
qk_ln_allclose_2e-2_2e-2=True
qk_ln_max_abs_diff=0
qk_ln_nan_mask_equal=True
qk_ln_nan_count=64
```

The `allclose` gate used `rtol=2e-2, atol=2e-2`; it passed, but bitwise equality and max absolute difference were also recorded. NaN was injected into one complete row, and the NaN mask gate passed. Signed zero was tested with `scale=-1` and `shift=-0.0`; sign-bit masks matched the explicit reference.

Raw finite output heads:

```text
ln_modulate_actual_head=[0.04296875, 2.296875, 0.546875, 0.49609375, -1.4140625, -1.3046875, -0.1875, -1.390625]
qk_ln_actual_head=[-0.921875, 0.52734375, 0.99609375, -1.6875, 0.81640625, 2.078125, 0.062255859375, 0.099609375]
```

The runnable `round_bf16_to_fp32` Triton helper used finite fp32 inputs plus explicit NaN, infinity, positive zero, and negative zero cases. Its independent Torch reference was `x.to(torch.bfloat16).to(torch.float32)`.

```text
round_bf16_bitwise_equal_finite=True
round_bf16_nan_mask_equal=True
round_bf16_signed_zero_mask_equal=True
round_bf16_inf_mask_equal=True
round_bf16_input_head=[1.0, -1.0, 1.0000001192092896, -1.0000001192092896, nan, inf, -0.826776921749115, 0.25210312008857727]
round_bf16_output_head=[1.0, -1.0, 1.0, -1.0, nan, inf, -0.828125, 0.251953125]
round_bf16_reference_head=[1.0, -1.0, 1.0, -1.0, nan, inf, -0.828125, 0.251953125]
```

## Interpretation

- The original FLUX warmup failure is already fixed on current `main` by disabling the PTX-dependent fused normalization fast path on ROCm. This is a negative/already-fixed result, not a new fix.
- Direct use of the five PTX helpers remains unsupported on gfx942. That is expected under the merged strategy, which avoids rather than ports the PTX.
- The closed PR 34352 and merged PR 34481 are consistent with that strategy. PR 34481 is the upstream fix that current `main` reflects; PR 34352 was closed without merge.
- No full-model end-to-end server run was performed because it would require model weights and exceeds the requested bounded synthetic-tensor scope.

Complete raw logs are retained under `reports/j-1f1a580e9886/raw-logs/`.
