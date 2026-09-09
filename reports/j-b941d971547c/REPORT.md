# IndexerKPool logits workspace investigation

## Status

This is an open, synthetic-operator investigation. It does **not** claim that the
candidate fixes the reported issue, and it does not change production code on
`main`.

The reported native abort was **not reproduced** on the assigned MI300X with the
image's current AITER revision. The unpatched PR #36607 source returned for
below-boundary, above-boundary, and production-shaped synthetic logits.

The candidate query-row chunking produced a real numerical mismatch on MI300X:
top-k **values** differed while top-k **indices** matched in the mismatch cases.
The mismatch is explained by AITER's query-length specialization when a chunk
has at most 1024 rows.

## Environment

- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Container hostname: `banff-cyxtera-cx57-4` (hostname is not image identity)
- GPU: one AMD Instinct MI300X, `gfx942`, 206141652992 bytes VRAM
- Python: `/opt/venv/bin/python`, 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch HIP: `7.2.26015-fc0010cf6a`
- Triton: `3.7.0`
- AITER source: `/sgl-workspace/aiter`, commit `c16d44b93a528b2a4bfd6d8d3409116d465872a9`
- AITER wrapper: `/sgl-workspace/aiter/aiter/ops/triton/attention/fp8_mqa_logits.py`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Installed SGLang source: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Delivery source: `/job/sglang`, commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Unpatched PR source: `/job/sglang-unpatched`, commit `aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e`
- Candidate source: `/job/sglang-candidate`, commit `ee7c66d8cc44fc7f78e43e64acdad8c2320ac6a4`

No model weights were downloaded. Build caches were kept job-private under
`/job/.cache`; the measured cache footprint was 46 MiB.

## Source findings

The real class is `IndexerKPool`, not the separate `Indexer` class.

### Current `main`

At commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`:

- `IndexerKPool` is in `python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py`.
- `_should_chunk_mqa_logits` exists around line 862.
- It has no call site in `IndexerKPool`.
- The class has no `_fp8_mqa_logits` helper.
- Its direct calls are `deep_gemm.fp8_mqa_logits` around lines 919 and 1151.
- `deep_gemm` is not installed in this image, so the current-main ROCm path is unsupported here.

The separate `Indexer` class does have a live chunking path on `main`; that does
not cover `IndexerKPool`.

### PR #36607 source

At merge commit `aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e`:

- `IndexerKPool` starts around line 56.
- `_fp8_mqa_logits` starts around line 170.
- The HIP branch imports AITER's `fp8_mqa_logits` and calls it directly around lines 181-191.
- `_should_chunk_mqa_logits` exists around line 988 but has no callers.
- The three `IndexerKPool` call sites are around lines 1060, 1175, and 1410.

### Candidate

At commit `ee7c66d8cc44fc7f78e43e64acdad8c2320ac6a4`, parent
`aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e`:

- Adds `_MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1`.
- Adds query-row chunking inside `IndexerKPool._fp8_mqa_logits`.
- Does not wire `_should_chunk_mqa_logits` to a caller.
- The complete tested diff is in `candidate.diff`.

## AITER behavior on MI300X

The current AITER wrapper loads the Gluon kernel only for `gfx950` or `gfx1250`.
On `gfx942`, it uses the non-Gluon Triton path. That path does not select the
2-GiB `buffer_store` fallback described in issue 37478, which explains why the
reported native abort did not reproduce on this MI300X stack.

The non-Gluon path also specializes on query length:

```python
matrix_instr_nonkdim = 32
if seq_len <= 1024:
    matrix_instr_nonkdim = 16
```

Therefore, splitting query rows can change the numerical kernel specialization
when a chunk has at most 1024 rows.

## Synthetic validation

The driver is `validate_indexer_kpool.py`. It:

- Uses only synthetic tensors; no model or server is loaded.
- Uses `NUM_HEADS = 32` and `HEAD_DIM = 128`.
- Runs each abort-prone case in a separate subprocess.
- Uses a 420-second timeout per case.
- Bounds all caches under `/job/.cache`.
- Compares against an independent 4096-row chunked AITER reference.
- Compares top-k values, top-k indices, checksums, and maximum absolute difference.
- Does not relax any numerical gate.

### Results

| Case | Source | Shape | Result |
|---|---|---|---|
| Current-main support | `ffe98a4279` | n/a | No `_fp8_mqa_logits`; `deep_gemm` unavailable |
| Forced 1000-row chunks | `ee7c66d8c` | 8192 x 20000 | Returned; checksum and top-k values mismatched; indices matched; max diff `4.57763671875e-05` |
| Below boundary | `aa8c950a3d` | 23168 x 23168 | Returned; exact vs reference |
| Above boundary | `aa8c950a3d` | 23180 x 23180 | Returned; exact vs reference |
| Production shape | `aa8c950a3d` | 8192 x 120000 | Returned; exact vs reference |
| Candidate above boundary | `ee7c66d8c` | 23180 x 23180 | Returned; checksum and top-k values mismatched; indices matched; max diff `1.52587890625e-05` |
| Candidate production shape | `ee7c66d8c` | 8192 x 120000 | Returned; exact vs reference |
| Balanced 11590-row probe | `ee7c66d8c` | 23180 x 23180 | Returned; exact vs reference |

Peak allocated memory stayed between 1.873 GiB and 11.036 GiB, far below the
206-GiB MI300X capacity.

The candidate's natural 23180-row split was `[23160, 20]`. The 20-row tail
crossed AITER's 1024-row specialization boundary and caused the value mismatch.
Its natural 8192 x 120000 split was `[4473, 3719]`, both above 1024, and was
exact. A separate balanced probe using 11590 rows was also exact, but this is
only a policy probe, not a proposed or validated general fix.

## Reproduction

The exact source worktrees were created with:

```bash
git worktree add --detach /job/sglang-unpatched aa8c950a3df62b6642c4ea60a93a5e3eb1a1450e
git worktree add --detach /job/sglang-candidate ee7c66d8cc44fc7f78e43e64acdad8c2320ac6a4
```

The suite was run with:

```bash
cd /job/sglang
/opt/venv/bin/python reports/j-b941d971547c/validate_indexer_kpool.py \
  --output /job/investigation-logs/indexer_kpool_results.json \
  --timeout 420
```

The balanced probe was run with:

```bash
cd /job/sglang
env XDG_CACHE_HOME=/job/.cache \
  TRITON_CACHE_DIR=/job/.cache/triton \
  TORCHINDUCTOR_CACHE_DIR=/job/.cache/torch-inductor \
  TORCH_HOME=/job/.cache/torch \
  HF_HOME=/job/.cache/huggingface \
  PYTHONPATH=/job/sglang-candidate/python \
  /opt/venv/bin/python reports/j-b941d971547c/validate_indexer_kpool.py \
  --child \
  --case candidate_balanced_above_boundary_23180x23180 \
  --kind boundary \
  --source-root /job/sglang-candidate \
  --num-q 23180 \
  --num-k 23180 \
  --forced-rows 11590 \
  --reference-rows 4096 \
  --seed 37478
```

Raw results are in `indexer_kpool_results.json`.

## Open questions

- Whether the reported abort still reproduces on `gfx950` with this AITER revision.
- Whether a general chunk policy can guarantee every chunk stays above 1024 rows without violating the 2-GiB bound.
- Whether top-k indices can mismatch for other seeds, tie patterns, or shapes even though they matched here.
- Whether upstream wants to re-land PR #36607 before any `IndexerKPool` bound is considered.

No upstream issue, PR, or comment was posted or modified.
