# GLM-5.3-Flash NextN embedding shard-index report

## Scope

- Campaign: `repo-e2e-20260909`
- Upstream context: `sgl-project/sglang` issue `37548`
- Mirror base: `amdpilot-org/sglang` `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Tested candidate: upstream PR `37791`, commit `6dec14c6ebf892b1b787b8ade5a5ed134ad86e36`
- Upstream already-fixed path: PR `36507`, commit `cdfc224b0e` (multimodal `mm_input_embeds` handling)

This investigation isolates **local shard-index correctness** for GLM-5.3-Flash NextN embeddings. It does **not** initialize distributed process groups and cannot establish TP8 collective behavior.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, UUID `GPU-bf6ec0aadcf456df`
- GPU ISA: `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Working source: `/job/j-44f922ef97c2/sglang/python/sglang`
- Installed source used for imports: `/sgl-workspace/sglang/python/sglang`
- Native module path: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- AITER native module observed: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

## Contract exercised

The new regression test uses a tiny synthetic vocabulary:

- Vocabulary size: `8`
- Hidden size: `3`
- Model TP size: `2`
- Attention TP size: `1`
- Boundary token IDs: `0`, `3`, `4`, `7`

It compares:

1. The inherited attention-TP layout (`use_attn_tp_group=True`) against a globally indexed `F.embedding` reference.
2. A model-TP shard layout (`use_attn_tp_group=False`) by summing the two local GPU gathers and comparing the sum to the same global reference.
3. The out-of-range diagnostic hook for token ID `8`, ensuring invalid IDs are not silently clamped.

## Raw results

### New focused test

Command:

```bash
PYTHONPATH=$PWD/python /opt/venv/bin/python -m pytest -q \
  test/registered/kernel/embeddings/test_glm5_next_nextn_embedding.py
```

Result:

```text
3 passed, 4 warnings in 14.61s
```

### Existing numerical gates

Command:

```bash
PYTHONPATH=$PWD/python /opt/venv/bin/python -m pytest -q \
  test/registered/unit/models/test_deepseek_nextn_mm_embed.py \
  test/registered/kernels/ops/embeddings/test_vocab_parallel_embedding.py
```

Result:

```text
24 passed, 4 warnings in 15.49s
```

No production code was changed, so these numerical gates remain unchanged.

### Candidate PR 37791 local probe

The candidate commit was applied locally and tested on gfx942. It changes GLM NextN from the inherited attention-TP embedding layout to a model-TP layout:

```text
inherited {'enable_tp': True, 'use_attn_tp_group': True}
model_specific {'enable_tp': True, 'use_attn_tp_group': False}
inherited_equal True
model_specific_equal True
rank0 [[0.0, 1.0, 2.0], [9.0, 10.0, 11.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
rank1 [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [12.0, 13.0, 14.0], [21.0, 22.0, 23.0]]
```

This proves local shard indexing for the candidate, not TP8 collective correctness.

### Out-of-range diagnostic probe

Command:

```bash
SGLANG_ENABLE_ASYNC_ASSERT=1 \
PYTHONPATH=$PWD/python \
/opt/venv/bin/python /tmp/sglang-cache-j-44f922ef97c2/probe_oob.py
```

Expected failure:

```text
torch.AcceleratorError: HIP error: unspecified launch failure
Kernel Name: _ZN2at6native25_assert_async_cuda_kernelIbEEvPKT_NS0_3MsgE
probe_status=1
```

The async assert fires before the gather, preserving the out-of-range diagnostic rather than silently clamping token ID `8`.

## Formatting

Command:

```bash
PRE_COMMIT_HOME=/tmp/sglang-cache-j-44f922ef97c2/pre-commit \
pre-commit run --files test/registered/kernel/embeddings/test_glm5_next_nextn_embedding.py
```

All hooks passed on the delivery branch.

## Limitations

- Only one assigned MI300X was used.
- No distributed process groups were initialized.
- No full model weights were downloaded.
- No upstream issue, PR, or comment was posted or modified.
- The candidate’s local shard-index correctness does not establish TP8 collective behavior.
