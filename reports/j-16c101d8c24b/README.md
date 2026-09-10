# MI300X structured-input reduced DeepSeek block probe

This report exercises the existing reduced DeepSeek-style MLA attention path and the real `DeepseekV2MLP` path as a residual decoder block. It uses six fixed-shape structured inputs: zeros, tiny finite values, mixed magnitudes, pairwise cancellation, skewed probability-like state, and negative sign skew. No full model weights are downloaded and no full-model quality is claimed.

The probe compares attention, MLP, and complete-block outputs with independent float32 Torch formulas. It preserves the existing FP16 dtype and the unchanged MLA gate (`atol=3e-2`, `rtol=3e-2`), requires finite outputs, and records cold and three warm complete-block timings with explicit synchronization.

## Reproduce

```bash
cd /job/work/sglang
export PYTHONPATH=/job/work/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-16c101d8c24b/triton
export HF_HOME=/tmp/sglang-cache-j-16c101d8c24b/hf
/opt/venv/bin/python reports/j-16c101d8c24b/structured_block_probe.py
```

The raw record is `results.json`. The required early installed-source baseline is `baseline-first.json`; it is explicitly not evidence for later checkout changes.

This is distinct from the completed MLA chunk-partition study in PR 486. It does not change production code and does not claim serving-wide performance.
