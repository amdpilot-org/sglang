# DSpark structured handoff validation

## Scope

This is a reduced, one-GPU validation of the existing DSpark handoff primitives. It does not load GLM weights, construct a full model engine, or make a distributed or throughput claim.

The new test exercises six fixed-shape structured cases through:

- `select_draft_hidden_without_anchor` for target-to-draft hidden-state selection.
- `CommitKvProj.triton` for synthetic bf16 stacked KV projection.
- `BuildCommitInjectLayout.triton` for synthetic commit-state layout.

All cases use batch size 4, gamma 5, hidden size 384, head dimension 128, three projection stages, stride 6, and the same synthetic state geometry. The six cases are zeros, tiny signed values, mixed magnitudes, cancellation pairs, skewed state, and boundary commit lengths. Synthetic weights total well under 4 GB and no checkpoint is downloaded.

## Independent references

- Hidden-state selection is compared bit-exactly to direct slicing of the same complete blocks.
- Projection is compared to independent CPU fp32 linear algebra, cast to the documented bf16 output dtype, with the existing complete-block gate `rtol=2e-2, atol=2e-3`.
- Commit layout is compared bit-exactly to explicit PyTorch tensor indexing for both complete `swa_loc` and `positions` blocks.

## Model constructor boundary

The full `DSparkWorkerV2` path is not instantiated because it requires a target worker, model runner, model config, parallel state, and NCCL port. Those dependencies are model/runtime-specific and are not available from synthetic primitive inputs. This is a labeled boundary, not an engine result.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, unique ID `0xd77d585f96863a60`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Torch native module: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Tested source paths are under `/job/sglang/python/sglang`.

## Commands

```bash
export PYTHONPATH=/job/sglang/python
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
timeout 180 /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/spec/dspark/test_dspark_structured_handoff.py
```

## Result

`1 passed, 3 warnings, 6 subtests passed in 14.11s` on gfx942. The bounded wall-clock command completed in 17.078 seconds. All complete-block outputs passed their unchanged numerical gates.

The installed-source baseline is recorded separately in `/job/baseline-first.json`. It used the existing `test_rmsnorm_hf_out_param` parity test, passed both fp16 and bf16 cases, and completed in 31.470 seconds. That baseline is evidence for the preinstalled source only and is not proof for this checkout.

## Context and limits

Read-only context came from `sgl-project/sglang` issue 30734 and `amdpilot-org/sglang` issue 405. The relevant upstream DSpark PRs remain open; this work tests the mirror `main` primitives and does not duplicate a merged fix. No upstream issue, PR, or comment was posted or changed.

The workload is finite: six cases, one GPU, no burn loop, no unbounded repetition, no sleep loop, and no environment replacement. Build logs and caches were kept under `/tmp/sglang-cache-j-956ab715995a`.
