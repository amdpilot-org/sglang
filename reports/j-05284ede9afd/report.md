# Bounded MI300X reduced DeepSeek MLP study

## Scope

This is a bounded, existing-configuration study of the reduced `DeepseekV2MLP` block on one assigned AMD Instinct MI300X (`gfx942`). It uses locally generated synthetic weights and inputs, compares against an independent Torch float32 formula, and does **not** claim full-model quality or download model weights.

Upstream context is sgl-project/sglang issue 16255. Its current checklist marks MHA, MLA, backend handler, ROCm hardware backend, weight loader, and utility refactors complete. The V3.2/NSA sub-issue 16815 is closed as inactive, and no matching V3.2/NSA extraction PR was found. This study therefore does not duplicate a completed refactor and makes no model-code change.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Operator-provided local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440003964`, unique ID `0xd9479c72156455e`, 206,141,652,992 B VRAM
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Delivery source: `/job/sglang`, commit `0084030179bfba86bfeb6d43f7997d4076329d2c`
- SGLang source path: `/job/sglang/python/sglang/__init__.py`
- Torch native path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- `sgl_kernel` path: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- AITER path: `/sgl-workspace/aiter/aiter/__init__.py`
- AITER native activation module: `/sgl-workspace/aiter/aiter/jit/module_activation.so`

## First GPU baseline

The installed source was `/sgl-workspace/sglang` at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. The preinstalled relevant test was:

```bash
/opt/venv/bin/python -m pytest \
  /sgl-workspace/sglang/test/registered/gemm/test_linear_bf16_fp32_hpc.py \
  -q --no-header
```

Result: `5 skipped` in 34.74 seconds. The concrete gate is `requires HPC-Ops and a Hopper GPU`; this host is `gfx942`, so that Hopper-only path is unsupported. This installed-source result is not evidence for later checkout changes.

The required meaningful neighboring control used real SDPA forwards on the same GPU:

```bash
/opt/venv/bin/python /tmp/baseline_first.py
```

Control dimensions were batch 2, heads 8, sequence 512, head dimension 128, `torch.bfloat16`. The independent reference was float32 `q @ k.T`, scale, softmax, then `@ v`. Maximum absolute error was `0.0012751221656799316`; the gate `max_abs_error < 0.05` passed. Timing used 3 real warmups plus 30 measured forwards with CUDA events and synchronization: `0.03761330048243205 ms/forward`. Peak Torch allocation was `0.111328125 GiB`.

The first GPU execution completed 72 seconds after the recorded job wall start. The complete artifact is `/job/baseline-first.json`.

## Reduced-block benchmark

The measured block is the existing `DeepseekV2MLP` path: merged gate/up GEMM, activation, and down GEMM. The two GEMMs and all inputs/weights were identical across dispatches. Only the existing activation dispatch changed:

1. `sgl_kernel` HIP path (`SiluAndMul.forward_cuda`)
2. Torch native path (`SiluAndMul.forward_native`)
3. AITER path (`SiluAndMul.forward_aiter`)

There are three supported configurations, below the four-configuration limit, and two workload shapes, for exactly six cases:

- 1,024 tokens, hidden size 2,048, intermediate size 512
- 4,096 tokens, hidden size 2,048, intermediate size 512

Model tensors were `torch.bfloat16`. The independent reference computed in float32:

```text
gate_up = linear(x, W_gate_up)
gate, up = split(gate_up)
activated = silu(gate) * up
reference = linear(activated, W_down)
```

The unchanged numerical gate was `max_abs_error < 0.02`. Every case passed. Timing used 3 real warmups and 30 measured real forwards per repetition, 3 repetitions per case, CUDA events, and synchronization after each repetition. This is 594 bounded real forwards total, with no artificial burn or unbounded loop.

| Tokens | Dispatch | Mean ms | Stdev ms | Min ms | Max ms | Max abs error | Pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 1024 | `sgl_kernel_hip` | 0.082735 | 0.001552 | 0.080943 | 0.083639 | 0.0000476381 | yes |
| 1024 | `torch_native` | 0.091232 | 0.001396 | 0.090195 | 0.092821 | 0.0000476381 | yes |
| 1024 | `aiter` | 0.120325 | 0.002697 | 0.117233 | 0.122187 | 0.0000627199 | yes |
| 4096 | `sgl_kernel_hip` | 0.091063 | 0.002100 | 0.088801 | 0.092950 | 0.0000500102 | yes |
| 4096 | `torch_native` | 0.097314 | 0.002878 | 0.095092 | 0.100565 | 0.0000500102 | yes |
| 4096 | `aiter` | 0.121804 | 0.006685 | 0.114605 | 0.127816 | 0.0000627199 | yes |

The default `sgl_kernel` HIP dispatch was fastest for both shapes. The observed standard deviations (`0.001396`–`0.006685 ms`) represent only this one shared MI300X and this bounded run; they are not a cross-GPU or full-model performance claim.

Reproduction:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-05284ede9afd/benchmark.py \
  --calls 30 --repeats 3 \
  --output reports/j-05284ede9afd/results.json
```

## Limits and honesty

- Synthetic weights were approximately 6 MiB, well below 4 GiB; no model weights were downloaded.
- Peak Torch allocation during the reduced benchmark was `0.2050800323486328 GiB`, below the 48 GiB live-allocation limit.
- The study used one GPU, at most six workload cases, three existing dispatches, and completed well within the 120-minute wall limit.
- Investigation wall interval: `2026-09-10T09:05:10Z` through `2026-09-10T09:10:08Z` (298 seconds), below the 7,200-second limit.
- No upstream issue, PR, or comment was posted or changed.
- Full-model quality, multi-GPU scaling, attention-backend parity, and the inactive V3.2/NSA extraction were not attempted.
