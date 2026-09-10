# Reduced DSpark handoff pipeline report

## Scope and result

This is a reduced, synthetic-state exercise of existing SGLang DFlash/DSpark
hidden-state projection, KV write, accept, and commit primitives on one assigned
AMD Instinct MI300X (`gfx942`). It is not a GLM model integration, checkpoint
benchmark, or distributed-system claim.

The tested mirror commit is `0084030179bfba86bfeb6d43f7997d4076329d2c`
(`origin/main`). The qualified image is
`amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, with the
operator-provided local image ID
`sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.

The final run passed every independent-reference, eager-versus-graph, and
explicit validation gate for all three workload cases. No checkpoint was
downloaded. Total synthetic weights were 12,873,216 bytes (about 12.3 MiB),
well below the 4 GiB limit.

## Execution paths

Only two genuinely supported installed execution paths were compared:

1. **Eager native dispatch**: the reduced block calls the existing SGLang
   Triton/PyTorch/`sgl_kernel`-backed primitives directly.
2. **Captured graph replay**: `torch.cuda.CUDAGraph` captures the same complete
   reduced block on HIP and replays it.

This is an eager-versus-captured-block comparison. It does not claim a separate
vendor backend, does not relabel a fallback, and does not claim coverage of a
full model-specific constructor. The model-specific constructor boundary is
explicit: the harness uses synthetic `SimpleNamespace` state and attaches the
existing DFlash/DSpark methods to it rather than constructing or loading a GLM
model.

## Early installed-source baseline

Before cloning or modifying the delivery checkout, the installed source was
used for a bounded numerical and timing baseline. The installed source commit
was `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.

```bash
cd /sgl-workspace/sglang/test
/opt/venv/bin/python manual/layers/test_activation.py TestQuickGELU.test_quick_gelu -v
```

The test dispatched `sgl_kernel.elementwise.gelu_quick` through
`QuickGELU.forward_hip` and compared all 24 fixed cases with an independent
native expression. It passed with float16 gates `atol=1e-3, rtol=1e-3` and
bfloat16 gates `atol=1e-2, rtol=1e-2`. The first GPU execution took 33 seconds
wall-clock from process start to exit; the unittest test body reported 0.793
seconds. The complete record is in `/job/baseline-first.json`. This baseline
describes the installed source only and is not evidence about later checkout
changes.

## Reduced pipeline

Run the final delivery-checkout test with:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python reports/j-df17cc3e1e01/reduced_pipeline.py
```

The harness uses seed 0 for weights and deterministic validation seeds for
fresh input values. It performs five warmups and 30 measured complete-block
iterations per execution path. Timing uses HIP CUDA events around the complete
block; input generation and pool reset happen outside the timed interval. Graph
capture reuses static buffers, while fresh values are copied into those buffers
before each replay.

The three cases are:

| Case | Batch | Stride | Hidden | Layers | KV heads | Head dim | Vocab | Weights |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 4 | 4 | 256 | 2 | 2 | 64 | 64 | 3,145,216 B |
| medium | 8 | 4 | 512 | 2 | 2 | 64 | 128 | 5,327,616 B |
| large | 16 | 4 | 1024 | 2 | 2 | 64 | 256 | 8,401,408 B |

### Final timing

| Case | Eager median | Graph median | Median speedup |
|---|---:|---:|---:|
| small | 1.315776 ms | 0.223095 ms | 5.897839x |
| medium | 2.134925 ms | 0.232296 ms | 9.190539x |
| large | 2.164594 ms | 0.231174 ms | 9.363504x |

### Numerical gates

Floating outputs use `torch.allclose` after casting both sides to float64 with
`atol=2e-5` and `rtol=2e-4`. Integer and layout outputs require exact equality.
The independent CPU references use double-precision RMSNorm, projection, RoPE,
scatter, greedy accept, commit-layout, and cumulative KV-pool calculations.

All eager and graph comparisons passed. Eager and graph outputs were exactly
equal for every reported tensor. The largest independent-reference absolute
difference was:

| Case | Output | Max abs diff | Max relative diff |
|---|---|---:|---:|
| small | pool keys | 1.3034970e-6 | 2.4280718e-2 |
| small | pool values | 5.5671526e-7 | 8.5295825e-3 |
| medium | pool keys | 1.3353158e-6 | 2.8366571e-3 |
| medium | pool values | 8.3786584e-7 | 6.4121096e-4 |
| large | pool keys | 2.1105475e-6 | 1.7810547e-3 |
| large | pool values | 1.0566100e-6 | 2.0774263e-2 |

The complete raw JSON, including every output comparison and native module path,
is in `results.json`.

## Native paths

The final run imported:

- Python: `/opt/venv/bin/python`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- SGLang: `/job/sglang/python/sglang/__init__.py`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Triton: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`

The dispatched SGLang paths include:

- `sglang.srt.models.dflash.DFlashDraftModel.project_target_hidden`
- `sglang.srt.speculative.dspark_components.dspark_kv_inject.TargetHiddenKvInjector`
- `sglang.srt.models.dspark.DSparkDraftMixin.write_target_hidden_kv`
- `sglang.kernels.ops.speculative.dspark.dspark_verify_window.scatter_compact_to_strided_into`
- `sglang.kernels.ops.speculative.dspark.dspark_accept.accept_greedy_triton`
- `sglang.kernels.ops.speculative.dspark.dspark_accept.finalize_accept_lens_triton`
- `sglang.kernels.ops.speculative.dspark.dspark_verify_window.BuildOutTokens`
- `sglang.kernels.ops.speculative.dspark.dspark_verify_window.build_unified_commit_inject_layout`

## Context and limitations

Read-only context was taken from sgl-project/sglang issue 30734. No upstream
issue, pull request, or comment was posted or modified.

The results apply only to this reduced synthetic block. They do not establish
full-model correctness, GLM checkpoint behavior, multi-GPU behavior, production
throughput, or support for an execution backend that was not actually
dispatched. A full model-specific constructor and checkpoint load remain outside
this boundary.
