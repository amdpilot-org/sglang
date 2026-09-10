# gfx942 OLMo-2 QK norm output-reuse report

## Outcome

This PR extends the already-covered OLMo-2 QK-norm dispatch work with the
uncovered execution-representation case: fresh output allocation versus
documented CUDA-graph buffer reuse across distinct input batches.

No source dispatch change is repeated. Mirror PR 271, commit
`8c273924a88d5b00420cebd6eae6d01daa8e8cba`, already contains the working
eager-dispatch fix. This change adds a focused GPU regression and a bounded
gfx942 timing matrix for the distinct reuse question.

## Environment

- Campaign: `repo-e2e-20260909`; job: `j-848a2ed93d5f`.
- GPU: one AMD Instinct MI300X, `gfx942`, 206141652992 bytes VRAM.
- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`,
  operator-supplied local image ID
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP: `7.2.26015-fc0010cf6a`.
- Installed source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Delivery checkout base: `/job/sglang`, commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Native paths: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`,
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`,
  `/sgl-workspace/aiter/aiter/__init__.py`,
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`, and
  `/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so`.

## Installed-source baseline

The installed-source baseline is saved at `/job/baseline-first.json` and is
not proof for later checkout changes. The relevant AOT fused QK-norm/RoPE
native op is unsupported in this installed ROCm build:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'fused_qk_norm_rope'
```

The Python wrapper exists at
`/opt/venv/lib/python3.10/site-packages/sgl_kernel/moe.py`, but the native op
is not registered. The unsupported path was not forced through. A meaningful
supported neighboring control used
`sglang.srt.layers.layernorm.RMSNorm.forward_native` on the same MI300X:

- First successful GPU execution elapsed time: `0.2521596075966954 s`.
- Independent float32 reference max absolute difference: `0.0`.
- Sentinel-padded storage checks passed.
- Fresh output storage: `0.12987129688262938 ms/call`.
- Safe reused output storage: `0.11583495140075684 ms/call`.

## New uncovered case

The new experiment uses the real `Olmo2Attention._apply_qk_norm` wrapper with
reduced synthetic `RMSNorm` modules. Q is bf16 `[tokens, 512]`, K is bf16
`[tokens, 128]`, and both norm weights are bf16. Three distinct deterministic
batches are used per shape.

The independent reference computes in float64:

```text
x * rsqrt(mean(x^2) + eps) * weight
```

then casts to bf16. The unchanged numerical gate is
`torch.allclose(output, reference, rtol=0.02, atol=0.02)`.

Fresh output storage calls the wrapper directly on sentinel-padded preallocated
inputs and keeps every output alive. Safe buffer reuse copies each distinct
input batch into sentinel-padded static graph buffers and replays the captured
graph. Captured Q/K output addresses remain fixed across replays, do not alias
the static inputs, and preserve bf16 dtype. Sentinel words around the static
inputs and graph-pool guard tensors remain unchanged after every replay.

The explicit unsupported `KernelBackend.TRITON` variant raises
`NotImplementedError: RMSNorm: no triton backend`; it is not forced through.

## Timing matrix

Timing used `torch.cuda.Event.elapsed_time`, five warmup calls and twenty timed
calls per mode. The fresh mode measures the wrapper on preallocated inputs with
freshly allocated outputs. The reuse mode measures input copies into static
buffers plus graph replay, with reused output storage. No unbounded loops,
sleep loops, synthetic burn, or repeated occupancy work were used.

| tokens | fresh output storage (ms/call) | safe buffer reuse (ms/call) |
|---:|---:|---:|
| 1 | `0.09640414714813232` | `0.27848975658416747` |
| 8 | `0.09527754783630371` | `0.26726584434509276` |
| 64 | `0.0974766492843628` | `0.2728226900100708` |

All fresh and reused outputs passed the unchanged independent-reference gate.
Fresh output addresses were distinct while their tensors were kept alive.
Captured output addresses were stable across distinct-batch replays.

## Commands

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/layers/test_olmo2_qk_norm_output_reuse.py

PYTHONPATH=/job/sglang/python /opt/venv/bin/python /tmp/bench_reuse.py
```

The timing script is not committed because it is a bounded one-off evidence
collector; the committed regression preserves the correctness, aliasing,
static-address, dtype, sentinel, and unsupported-backend contracts.

## Boundaries and evidence consulted

- Read-only context: sgl-project/sglang issue 33415 and PR 33416;
  amdpilot-org/sglang issue 213 and PRs 271 and 361.
- No full model weights were downloaded, no toolchain was replaced, and no
  node-wide state was modified.
- No upstream issue, PR, or comment was posted or changed.
- The installed AOT fused QK-norm/RoPE native op is unsupported on this ROCm
  stack and was not forced through.
