# Masked KV representation follow-up

This follow-up to amdpilot-org/sglang issue 207 and read-only context
sgl-project/sglang issue 31568 covers execution representations only. It does
not repeat the already-working batch-size cache separation fix from upstream
pull request 31689.

The installed-source and mirror-main evidence is in `results.json`. The kernel
honors non-unit batch and head strides, but its address expression assumes unit
stride on the final dimension. A final-dimension permuted view was therefore
silently mis-dispatched before this change. The wrapper now rejects that
representation and packed 2-D K/V tensors before launching Triton.

Reproduce on one gfx942 GPU with the qualified `/opt/venv/bin/python` stack:

```bash
TRITON_CACHE_DIR=/tmp/sglang-cache-j-6eac346fa6c2/patched \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest -q \
test/registered/mem_cache/test_masked_set_kv_buffer_layout.py
```

The raw numerical reference comparison, sentinel checks, native paths, and
bounded timing matrix are in `results.json`.
