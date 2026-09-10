# MiniMax sparse backend probe: j-5e528df1c60e

## Scope and conclusion

This is the MiniMax sparse-backend boundary probe requested for batch 20260910. It is distinct from the MiniMax H3 denoising task and does not claim full MiniMax-M3 speculative-decode serving on MI355X.

The actual GPU backend supports the tested ordinary decode and extend shapes. It does not support speculative `TARGET_VERIFY` on GPU: that shape routes through `forward_extend`, may leave `extend_seq_lens` as `None`, and has no reference-tested GPU verify path. This change preserves that boundary and fails during backend construction with a precise diagnostic instead of failing later on `NoneType.device`. No speculative-decode path was added.

Upstream sgl-project/sglang issue 33383 was read only. At the time of the probe it had no comments and no linked PR was found by the bounded searches below. No upstream issue, PR, or comment was posted or changed.

## Environment

- Mirror and PR base: `amdpilot-org/sglang`, branch `amdpilot/j-5e528df1c60e`, base commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, operator-supplied local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 191.98 GiB reported by Torch.
- Qualified stack: `/opt/venv/bin/python`, Torch `2.9.1+rocm7.2.0.git7e1940d4`, ROCm/HIP `7.2.26015-fc0010cf6a`.
- Working Python source: `/job/sglang/python/sglang`; preinstalled source context: `/sgl-workspace/sglang/python/sglang`.
- Torch: `/opt/venv/lib/python3.10/site-packages/torch`; Triton: `/opt/venv/lib/python3.10/site-packages/triton`.
- Native modules: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`; AITER JIT objects under `/sgl-workspace/aiter/aiter/jit/`.
- Job-private caches: `/tmp/sglang-cache-5e528df1c60e/triton` and `/tmp/sglang-cache-5e528df1c60e/inductor`.
- No model weights were downloaded and no node-wide state was modified.

## Commands

```bash
gh issue view 33383 --repo sgl-project/sglang \
  --json title,body,comments,state,author,createdAt,updatedAt,url
gh search issues --repo sgl-project/sglang --match body '33383' \
  --json number,title,state,url,updatedAt --limit 30
gh search prs --repo sgl-project/sglang --match body '33383' \
  --json number,title,state,url,updatedAt --limit 30

export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-5e528df1c60e/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-5e528df1c60e/inductor
export SGLANG_DISABLE_MSA=1
/opt/venv/bin/python reports/j-5e528df1c60e/probe_minimax_sparse_mi300x.py \
  --output reports/j-5e528df1c60e/probe_results.json

/opt/venv/bin/python -m pytest -q \
  test/registered/unit/layers/attention/test_minimax_sparse_spec_decode_boundary.py \
  test/registered/unit/models/test_minimax_m3_vl_eagle_hooks.py
```

## Raw probe results

The probe uses synthetic tiny batches and the real `MiniMaxSparseAttnBackend`, its paged KV/index pools, and its Triton kernels. The independent reference is implemented with ordinary Torch tensor operations in `probe_minimax_sparse_mi300x.py`; it follows causal block scoring, init/local forcing, top-k selection, and selected-block attention.

- Decode: batch 2, one query per request, `extend_seq_lens=None`, output shape `[2, 64]`, `idx_out=None`, maximum absolute difference from Torch `0.0032256245613098145`, mean absolute difference `0.0003738238592632115`.
- Extend: extend lengths `[3, 2]`, output shape `[5, 64]`, `idx_out=None`, maximum absolute difference from Torch `0.006807565689086914`, mean absolute difference `0.0003644036769401282`.
- Before the diagnostic change, GPU `TARGET_VERIFY` with `extend_seq_lens=None` failed late with `AttributeError: 'NoneType' object has no attribute 'device'` in `forward_extend`.
- After the diagnostic change, the same speculative configuration fails during backend construction with `NotImplementedError` and the message preserved in `probe_results.json`.
- New unit result: `4 passed` for the speculative-boundary and EAGLE-hook tests.

The complete machine-readable result is committed as `reports/j-5e528df1c60e/probe_results.json`.

## Boundary and hooks

- Ordinary GPU decode remains supported and was exercised with `extend_seq_lens=None`.
- Ordinary GPU extend remains supported and was exercised with populated extend metadata.
- GPU speculative `TARGET_VERIFY` remains unsupported and now fails early. A minimal verify path was intentionally not added because there is no concrete passing reference test for that shape.
- The NPU branch retains its existing native `TARGET_VERIFY` path; this change does not alter it.
- `MiniMaxM3SparseForConditionalGeneration.get_embed_and_head` returns the text embedding and LM head weights.
- `set_eagle3_layers_to_capture` enables capture, applies the required `+1` layer offset, and marks `_is_layer_to_capture` on the selected inner layers.
- Non-last pipeline ranks do not enable capture. These behaviors are locked by the new hook unit test.

## Not done

- No full EAGLE3 server run was attempted because it would require full target and draft weights and is outside the bounded synthetic probe.
- No MI355X/gfx950 claim is made; the assigned device is MI300X/gfx942.
- No speculative verify kernel or metadata reconstruction was implemented. That work still needs a concrete target-verify reference test and numerical gates.
