# gfx942 execution-representation follow-up

This report extends the prior QSA dtype-boundary investigation with the uncovered
non-contiguous Q/K/V representation used by the unpacked sparse-prefill Triton
kernel. It does not repeat the original BF16-query/FP8-KV trigger.

Run the bounded numerical and timing matrix with:

```bash
TRITON_CACHE_DIR=/tmp/sglang-cache-j-247bd4ecafcb/triton-benchmark \
PYTHONPATH=python /opt/venv/bin/python reports/j-247bd4ecafcb/reproduce.py
```

The script compares contiguous and stride-based non-contiguous inputs against
`qsa_sparse_attention`, protects each non-contiguous input with unchanged guard
regions, and records the actual Triton gfx942 dispatch artifacts. Mixed Q/K/V
dtypes are rejected by a focused test rather than forced through the kernel.
