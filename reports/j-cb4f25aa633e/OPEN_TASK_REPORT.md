# Open task report: j-cb4f25aa633e

## Scope

- Campaign: `repo-e2e-20260909`
- Coordination tracker: `amdpilot-org/amdpilotv2` issue `402`
- Read-only context: `sgl-project/sglang` issue `31545`
- Distinct follow-up to `amdpilot-org/sglang` issue `236`: changing only profiler marker placement must not change GPU outputs or actual event-measured kernel duration.
- Delivery base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Branch: `amdpilot/j-cb4f25aa633e`

The already-delivered issue-236 scope on branch `amdpilot/j-9e6ab99e71b6` (commit `03f550b405a14eabe7cd6f6cc1b1db44a6709abe`) was treated as context and was not duplicated. No upstream issue, pull request, or comment was posted or modified.

## Environment

- GPU: one assigned AMD Instinct MI300X, compute capability `9.4` (`gfx942`)
- Image: operator-provided local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`; container hostname was not used as image identity
- Interpreter: `/opt/venv/bin/python`, Python `3.10.12`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Native modules: `/opt/rocm/lib/libamdhip64.so.7`, `/opt/rocm/lib/libroctracer64.so.4`, and `/opt/rocm/lib/libhsa-runtime64.so.1`
- Installed-source context: `/sgl-workspace/sglang/python/sglang/__init__.py`, version `0.5.18.dev20260826+g937af8538b`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Delivery checkout: `/job/sglang/python/sglang/__init__.py`
- No full model weights, framework-stack replacement, node-wide state change, environment replacement, or artificial GPU occupancy workload were used.

## Installed-source baseline

This baseline was run before cloning or editing the delivery checkout. It is an installed-source control and is not proof for later checkout changes.

Command:

```bash
/opt/venv/bin/python /tmp/sglang_baseline_first.py
```

- Kernel: `sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul`
- Imported source: `/sgl-workspace/sglang/python/sglang/kernels/ops/elementwise/elementwise.py`
- Reference: independent `attn.float() * torch.sigmoid(gate.float())`, cast to the input dtype
- Numerical gate: `torch.testing.assert_close(..., rtol=1e-2, atol=1e-2)`
- Timing: CUDA/HIP events around one kernel call, median of five in-profile calls after two warmups
- Marker placements: no marker, marker before, marker around, and marker after
- Finite matrix: twelve float16/bfloat16 cases covering standard, zero-gate, negative, and subnormal inputs at bounded shapes
- Result: every output passed the unchanged numerical gate; the first GPU execution was reached `1.990508203394711` seconds after the in-process timing origin (after Python, Torch, and SGLang imports)
- Raw record: `reports/j-cb4f25aa633e/baseline-first.json`

## Checkout marker-placement control

Command:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  /job/marker-placement-repro.py
```

The control captured a finite 32-operation `addmm` plus `Gelu` HIP graph with `64x64` float32 tensors. Each case used one active Torch profiler session and compared these placements:

- no marker
- `record_function` immediately before the event interval
- `record_function` around the event interval and graph replay
- `record_function` immediately after the event interval

The finite adversarial matrix contained standard random, zero, negative, subnormal, large finite, and mixed-sign inputs. Each placement was replayed seven times after two side-stream warmups. Timing used CUDA/HIP events around one graph replay and synchronized the end event. The independent reference was a CPU float64 `addmm`/`Gelu` chain cast to float32.

Gates:

- Output: `torch.allclose(..., rtol=2e-5, atol=2e-5)`
- Timing: each non-baseline marker median must remain within 10% of the no-marker median

Result:

- All 24 case/placement combinations passed the output gate.
- Maximum independent-reference absolute difference: `3.725290298461914e-09`.
- Maximum median event-duration relative difference: `3.134431633846734%` in the standard-input case.
- Every per-case timing gate passed.
- Raw record: `reports/j-cb4f25aa633e/marker-placement-repro.json`

No mismatch was demonstrated, so no runtime, scheduler, profiler, or kernel code was changed.

## Unsupported boundaries

- This result does not repair or claim to repair deferred trace attribution from `sgl-project/sglang` issue `31545`. CPU marker intervals still cannot be used as proof of graph-mode GPU ordering on ROCm.
- CUDA/HIP event elapsed time measures the synchronized graph-replay interval; it does not recover per-kernel attribution or overlapping-stream wall time.
- The control is a finite synthetic graph chain, not a full model or production decode server. No model weights were downloaded.
- The MI355X production configuration in issue `31545` was not reproduced; this job had one MI300X (`gfx942`).
- Dispatch-level ROCm profiling and PyTorch in-trace graph annotations were not available in this qualified ROCm 7.2.0 path.
- The installed-source baseline is context only; checkout conclusions use the delivery checkout at `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Changes

- Added this truthful negative-result report and raw JSON records.
- No runtime code was changed because the required precondition—a demonstrated marker-placement mismatch—was not met.
