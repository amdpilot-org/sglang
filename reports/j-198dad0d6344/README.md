# Bounded MI300X BF16/FP8 KV experiment

This report exercises the supported Triton MHA decode backend on one MI300X
(`gfx942`) with locally generated synthetic weights. It compares BF16 and FP8
KV outputs against independent references built by reading and, for FP8,
dequantizing the actual cache with explicit scales. It records recurrent
decode state checks, bounded latency, live allocation, and ROCm profiler
kernel dispatch. It makes no model-quality claim.

## Reproduce

From the repository root:

```bash
PYTHONPATH=$PWD/python /opt/venv/bin/python \
  reports/j-198dad0d6344/benchmark_fp8_kv.py \
  --output reports/j-198dad0d6344/results.json
```

The run uses three finite decode shapes for BF16 and FP8 (six workload cases),
three recurrent decode steps per case, two warmup forwards, and ten timed
forwards. The live-allocation guard is 48 GiB. No checkpoint is downloaded.
