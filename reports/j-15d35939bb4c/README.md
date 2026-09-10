# gfx942 AITER/FlyDSL inference bridge control

## Result

This is a narrow MI300X (`gfx942`) compatibility control for the AITER upgrade bridge described by sgl-project/sglang issue 26890. It is not an AITER upgrade, a gfx950 training/autograd claim, or a full inference benchmark.

The real AITER dispatch path was exercised with `AITER_MLA_REDUCE_FLYDSL=1`:

```text
aiter.mla._mla_decode_reduce_v1_dispatch
  -> aiter.ops.flydsl.flydsl_mla_reduce_v1
```

The supported decode shape was 4 reduce tiles, 4 splits per tile, 16 heads, and head dimension 512 in bf16. Ordinary inference tensors, `requires_grad=True` parameters, and contiguous views of `requires_grad=True` parameters all passed the unchanged numerical gates and produced bitwise-identical outputs. The harness observed 42 actual FlyDSL dispatch calls.

The integration therefore did not demonstrate the upgrade-reported DLPack bridge defect on gfx942. The actual MLA bridge passes raw pointers through `aiter.ops.flydsl.kernels.tensor_shim.ptr_arg`; it does not call `flydsl.compiler.from_dlpack` for its tensor arguments.

Direct DLPack export controls did reproduce the precise upstream error in both normal mode and the actual `torch.inference_mode()` boundary:

```text
builtins.BufferError:
Can't export tensors that require gradient, use tensor.detach()
```

The same error occurred for a parameter and for a view of a parameter. Exporting `parameter.detach()` succeeded and returned `flydsl.compiler.jit_argument.DLTensorJitArg`.

## Candidate decision

No candidate was tested. AITER PR 5327 (head commit `23b832bdb23a7d118ca7d770bd0bebfd8531d085`) detaches tensors in the gfx950 A16W16 GEMM `_dynamic_tensor_arg` DLPack path. That path is not the gfx942 MLA bridge tested here, and no gfx942 integration defect was demonstrated. Testing that gfx950-only candidate on MI300X would not answer this control and could incorrectly imply gfx950 training/autograd coverage.

## Reproduction

From the delivery checkout:

```bash
AITER_MLA_REDUCE_FLYDSL=1 \
/opt/venv/bin/python reports/j-15d35939bb4c/inference_bridge_control.py
```

The script writes the raw numerical, parity, timing, DLPack, traceback, source-path, native-path, GPU, and image data to `reports/j-15d35939bb4c/results.json`.

The installed-source baseline was recorded before cloning or editing the delivery checkout:

```bash
/opt/venv/bin/python /job/baseline_probe.py
```

The baseline used the same FlyDSL MLA reduce operator with a small 4-tile, 4-split, 2-head, 256-dimensional bf16 fixture. Its first GPU execution completed in 8.204267 seconds, including imports and JIT setup. After three warmups, ten CUDA-event timed runs had a 0.0906731 ms mean. Its independent float64 online-softmax reference had output max absolute error 0.0 and LSE max absolute error 2.384185791015625e-07. The full baseline artifact is `baseline-first.json`; this installed-source result is not proof for later checkout changes.

## Environment

- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP: `7.2.26015-fc0010cf6a`
- AITER source: `/sgl-workspace/aiter/aiter/__init__.py`, commit `c16d44b93a528b2a4bfd6d8d3409116d465872a9`
- AITER native core: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- FlyDSL: version `0.3.1`, `/opt/venv/lib/python3.10/site-packages/flydsl/__init__.py`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, capability `(9, 4)`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`

## Raw results

- Installed-source baseline: `baseline-first.json`
- Real dispatch and DLPack control: `results.json`
- Reproduction harness: `inference_bridge_control.py`

No upstream issue, pull request, or comment was posted or modified.
