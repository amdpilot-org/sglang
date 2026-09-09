# EAGLE draft backend KV index translation investigation (gfx942)

## Result

On one AMD Instinct MI300X (`gfx942`), the owning mock runner's `KVIndexTranslator`
does **not** reach the late-created EAGLE DSA backends:

- `DraftBackendFactory.create_decode_backend()` creates a
  `DeepseekSparseAttnMultiStepBackend` with two `DeepseekSparseAttnBackend`
  children; both children have `kv_index_translator = None`.
- `DraftBackendFactory.create_draft_extend_backend()` creates a
  `DeepseekSparseAttnBackend`; its `kv_index_translator` is also `None`.
- The owning runner itself has a non-`None` `KVIndexTranslator`.

For the synthetic static-pool FP8 MHA prefix read on `gfx942`, this missing
binding did not misaddress the cache. The `gfx942` path takes the non-gfx95
branch of `_get_mla_kv_buffer_from_fp8_for_dsa()` and passes
`page_table_1_flattened` directly to `dequantize_k_cache_paged()`. The read
indices and dequantized nope/rope rows matched an independent reference exactly.
The runner translator is a passthrough for this pool
(`needs_read_translate = False`).

This is an observation from bounded synthetic validation, not a claim that the
upstream gfx950 crash is fixed by binding or not binding the translator.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-supplied local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, capability `(9, 4)`
- GPU unique ID: `0x6e8448eb6f49db1d`; serial: `692440004395`
- ROCm SMI driver: `6.19.14.31400000`
- Python: `3.10.12` at `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Working clone: `/job/sglang`, current `main` commit
  `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Installed source context: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- AITER: `/sgl-workspace/aiter/aiter`
- Job-private Triton cache: `/job/.cache/triton`

The raw environment capture is in `environment.txt`.

## Upstream context

- Read sgl-project/sglang issue 38029. It reports that late EAGLE DSA backends
  retain the `AttentionBackend` class default `None` and reach
  `translate_dcp_read_ids()` on gfx950.
- Read sgl-project/sglang PR 38318, merged as commit
  `6287ebf43a4408a706d42358d4dba0b688c1f3d8`. That change makes the gfx95
  read door skip translation when `kv_index_translator` is `None`; it does not
  bind the owning runner translator to DSA backends.
- Current mirror `main` already contains PR 38318. No upstream issue, PR, or
  comment was modified.

## Synthetic validation

`validate.py` uses the repository's synthetic DSA test kit to build:

- Two requests with nonempty prefixes of 64 and 32 tokens.
- Two extend tokens per request.
- `page_size = 1`, required by the gfx942 legacy HIP DSA pool path.
- A `DSATokenToKVPool` with `torch.float8_e4m3fn` storage and the 656-byte
  DSA FP8 layout.
- Shuffled physical page/slot assignments.
- A real owning runner `KVIndexTranslator`.
- Late EAGLE draft and draft-extend DSA backend creation through
  `DraftBackendFactory`.

The raw cache rows are populated directly with known FP8 nope bytes, four
independent float32 scales, and known BF16 rope bytes. The expected physical
indices are derived independently from the shuffled location function. The
expected nope and rope outputs are computed with pure PyTorch from those raw
bytes; the production quantizer is not used to create the reference.

`gfx942` does not naturally select DSA MHA one-shot (`get_device_sm()` returns
94, while the selector accepts SM90, SM100+, or gfx95). For this operator-level
test, the draft-extend backend's `_get_device_sm` is forced to 90 so the MHA
one-shot FP8 dequantization path runs on the assigned gfx942 GPU. This is a
synthetic dispatch override and is recorded in the limitation section.

The unchanged numerical gates are exact equality for:

- `page_table_1_flattened` versus independently expected physical indices.
- Dequantized nope rows versus the pure-Torch reference.
- BF16 rope rows versus the pure-Torch reference.

## Raw results

| Revision | Commit | Draft children | Draft-extend | Indices equal | Nope equal | Rope equal | Max abs diff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current `main` | `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` | `None`, `None` | `None` | true | true | true | 0.0 |
| upstream candidate | `6287ebf43a4408a706d42358d4dba0b688c1f3d8` | `None`, `None` | `None` | true | true | true | 0.0 |
| candidate parent | `a7117854754f52b0dd648d04094794de8102aa79` | `None`, `None` | `None` | true | true | true | 0.0 |

All three runs also recorded:

- `gfx95_read_door = false`
- `translator_passthrough_equal = true`
- `translator_needs_read_translate = false`
- `use_mha = true`
- `using_mha_one_shot_fp8_dequant = true`

Raw JSON and logs:

- `current-main.json`, `current-main.log`
- `candidate-6287ebf43a.json`, `candidate-6287ebf43a.log`
- `prefix-6287ebf43a-parent.json`, `prefix-6287ebf43a-parent.log`

## Commands

Current `main`:

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export PYTORCH_TUNABLEOP_ENABLED=0
export TRITON_CACHE_DIR=/job/.cache/triton
/opt/venv/bin/python reports/j-5b0bbdb74fe4/validate.py \
  --label current-main \
  --output reports/j-5b0bbdb74fe4/current-main.json
```

Candidate and pre-fix worktrees:

```bash
cd /job/sglang
git worktree add --detach /job/sglang-candidate-6287ebf43a 6287ebf43a
git worktree add --detach /job/sglang-prefix-6287ebf43a 6287ebf43a^

PYTHONPATH=/job/sglang-candidate-6287ebf43a/python \
  /opt/venv/bin/python reports/j-5b0bbdb74fe4/validate.py \
  --label candidate-6287ebf43a \
  --output reports/j-5b0bbdb74fe4/candidate-6287ebf43a.json

PYTHONPATH=/job/sglang-prefix-6287ebf43a/python \
  /opt/venv/bin/python reports/j-5b0bbdb74fe4/validate.py \
  --label prefix-6287ebf43a-parent \
  --output reports/j-5b0bbdb74fe4/prefix-6287ebf43a-parent.json
```

## Interpretation and limitations

- The late-binding observation is present on current `main`: the generic
  `ModelRunner.init_attention_backends()` bind pass happens before
  `EagleDraftWorker.init_attention_backend()` creates replacement DSA backends.
- `DeepseekSparseAttnBackend.__init__()` does not assign
  `model_runner.kv_index_translator`, so the late backends retain the base-class
  default `None`.
- On `gfx942`, `_get_mla_kv_buffer_from_fp8_for_dsa()` takes the non-gfx95
  branch and does not call `translate_dcp_read_ids()`. Therefore PR 38318's
  gfx95-only `None` guard does not change this gfx942 synthetic result.
- The static-pool translator is a passthrough. No unified-pool or nontrivial
  virtual-to-physical translation was exercised.
- No full model weights were downloaded and no multi-GPU serving was run.
- The MHA one-shot selector was forced for operator validation; natural gfx942
  dispatch was not claimed.
- The gfx950 branch from upstream issue 38029 was not exercised because the
  assigned GPU is gfx942.
- This validation covers late backend creation and the FP8 prefix read door,
  not a complete EAGLE worker forward or end-to-end speculative decoding.

No production code change is proposed because current `main` already contains
the merged upstream candidate and the gfx942 measurements show no prefix-slot
misaddressing in this bounded case.
