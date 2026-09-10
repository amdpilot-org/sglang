# Bounded gfx942 fused-sigmoid-mul compiled-path reuse

## Scope

- Campaign: `repo-e2e-20260909`
- Coordination tracker: `amdpilot-org/amdpilotv2` issue `402`
- Read-only context: `sgl-project/sglang` issue `31545`
- Distinct follow-up to `amdpilot-org/sglang` issue `236`; this does not repeat its original profiler-attribution trigger.
- Prior mirror PR `331` covered eager versus graph-replay attribution and partial event loss. Prior mirror PR `347` covered profiler marker-placement invariance. This report extends the uncovered cold versus warm direct-Triton compiled-path reuse case for `fused_sigmoid_mul`.
- Delivery base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Branch: `amdpilot/j-ca0f72e54fb1`

## Environment

- GPU: one assigned AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, UUID `65643763-3865-6332-3831-393339386666`
- Image: operator-provided local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`; container hostname was not used as image identity
- Interpreter: `/opt/venv/bin/python`, Python `3.10.12`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Triton: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`, `3.7.0+amd.rocm7.2.0.git89002410`
- Delivery source: `/job/sglang/python/sglang/__init__.py`
- Kernel source: `/job/sglang/python/sglang/kernels/ops/elementwise/elementwise.py`
- Native modules: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`, `/opt/rocm/lib/libamdhip64.so.7`, `/opt/rocm/lib/libroctracer64.so.4`, and `/opt/rocm/lib/libhsa-runtime64.so.1`
- Job-private Triton cache: `/tmp/sglang-cache-j-ca0f72e54fb1`
- No full model weights, framework-stack replacement, node-wide state change, unbounded loop, sleep loop, or synthetic GPU occupancy workload was used.

## Installed-source baseline

The first GPU execution objective was completed before checkout changes. The installed source is a different revision and is not proof for the delivery checkout.

- Record: `/job/baseline-first.json`, copied as `reports/j-ca0f72e54fb1/baseline-first.json`
- Label: `installed-source baseline; not proof for later checkout changes`
- Command: `/opt/venv/bin/python /job/baseline_first.py`
- Operation: `torch.mm` captured in `torch.cuda.CUDAGraph` and replayed across `(64,64,64)`, `(128,128,128)`, and `(256,256,256)`
- Reference: independent CPU `float64` `torch.matmul`
- Result: all three shapes passed `torch.allclose(..., rtol=2e-5, atol=2e-5)`; guard tensors remained unchanged
- Timing: CUDA/HIP events around one graph replay, three bounded measurements per shape
- First GPU execution elapsed from the in-process timing origin: `3.0325284684076905` seconds
- Native dispatch: `hipGraphLaunch` plus the MI300X matmul kernel

## Delivery-checkout experiment

The new case uses the real direct Triton operation
`sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul` with:

- Supported dtype: `torch.float16` inputs and `torch.float16` output
- Finite shape sequence: `(1,2048)`, `(3,2048)`, and `(7,2048)`
- Independent CPU `float64` reference: `attn * sigmoid(gate)`, cast to `float16`
- Five synchronized warm wall-clock measurements per shape
- Four bounded graph replays per shape
- Output sentinel `-12345.0` overwritten before each graph replay
- Separate guard tensor filled with the same sentinel
- Static input/output addresses checked before and after replay
- Documented `inplace=True` aliasing checked separately
- Unsupported mismatched-shape and complex-dtype variants required to fail

The apparent sentinel anomaly from the `inplace=True` probe is not a graph defect: when the output is prefilled, it is also the input, so replay correctly computes `sentinel * sigmoid(gate)`. The separate-output graph replay overwrites the sentinel and matches the independent reference.

## Results

The focused test passed:

```text
4 passed, 2 warnings in 5.15s
```

| shape | first call | warm median | graph replay median | max reference error |
|---|---:|---:|---:|---:|
| `(1,2048)` | `1043.03575 ms` | `0.041114 ms` | `0.0139925 ms` | `0.0` |
| `(3,2048)` | `0.095724 ms` | `0.040712 ms` | `0.0140725 ms` | `0.0` |
| `(7,2048)` | `0.101673 ms` | `0.038887 ms` | `0.013932 ms` | `4.76837158203125e-07` |

The first `(1,2048)` call includes Triton compilation. The later shapes reuse the same specialization: the Triton `device_caches` delta is exactly one. All direct and graph-replay outputs passed `rtol=1e-2, atol=1e-2`, output sentinels were overwritten, guard tensors remained unchanged, and static addresses stayed stable.

Profiler dispatch recorded:

- Direct path: `hipModuleLaunchKernel` and `_fused_sigmoid_mul_kernel`
- Graph replay: `hipGraphLaunch` and `_fused_sigmoid_mul_kernel`

Raw values are in `reports/j-ca0f72e54fb1/results.json`.

## Unsupported variants

- `torch.compile(fused_sigmoid_mul, dynamic=False, fullgraph=True)` returned without an exception on this Torch/Triton stack, but its output did not match the direct reference (`4.10107421875` maximum absolute error). It is recorded as unsupported negative evidence and is not forced through as a supported path.
- Direct Triton `torch.complex64` inputs failed clearly with `KeyError: 'complex32'`.
- Mismatched input shapes failed clearly with the documented `AssertionError`.

## Reproduction

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-ca0f72e54fb1/fresh

/opt/venv/bin/python -m pytest -q -s --tb=short -p no:cacheprovider \
  test/registered/kernels/ops/elementwise/test_fused_sigmoid_mul_compiled_path_reuse.py

/opt/venv/bin/python reports/j-ca0f72e54fb1/run_reuse_experiment.py
```

## Limitations

- This is bounded operator-level evidence on one MI300X, not an end-to-end benchmark or a full-model claim.
- Timing is observational; no performance threshold is asserted.
- CUDA and non-gfx942 ROCm architectures were not tested.
- The installed-source baseline is context only; checkout conclusions use commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- No runtime kernel behavior was changed. This PR adds the focused test, reproduction harness, and truthful evidence records.
- No upstream issue, pull request, or comment was posted or modified.
