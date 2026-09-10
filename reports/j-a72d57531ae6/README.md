# MI300X reduced DeepSeek MLA chunk-partition probe

## Scope

This report exercises the existing reduced DeepSeek-style MLA attention path in
`python/sglang/test/kits/attention_unittest/attention_methods/mla_attention.py`
on one assigned AMD Instinct MI300X (`gfx942`). It uses locally generated
synthetic weights and inputs only; it does not download or evaluate a full
model, and it does not claim full-model quality.

The probe is a distinct chunk-partitioning follow-up to
`amdpilot-org/sglang` issue 384. Read-only context from
`sgl-project/sglang` issue 16255 shows the DeepSeek refactor is already split
into `deepseek_common` modules in this checkout, so this report preserves that
implementation and measures it rather than proposing another refactor.

## Environment

- GPU: one AMD Instinct MI300X, capability `(9, 4)`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local
  image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Python: `/opt/venv/bin/python`.
- Delivery checkout: `amdpilot-org/sglang`, branch
  `amdpilot/j-a72d57531ae6`, base commit `008403017`.
- Backend: existing `triton` MLA attention backend.
- Numerical gate: unchanged `MLA_ATOL = 3e-2`, `MLA_RTOL = 3e-2`.

## Reproduction

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-a72d57531ae6/chunk_partition_probe.py
```

The script writes the full raw record to
`reports/j-a72d57531ae6/results.json`. It uses one cold execution and the median
of three warm executions per path, synchronizing around every forward. It
records peak CUDA memory after each case. The total wall time for the six-case
run was under one minute on the assigned MI300X.

## Cases

The finite matrix contains six legal partition cases over identical synthetic
workload data within each case:

| Total tokens | Partition |
|---:|---|
| 64 | 32, 32 |
| 128 | 64, 64 |
| 256 | 128, 128 |
| 512 | 256, 256 |
| 768 | 256, 256, 256 |
| 1024 | 512, 512 |

Each case compares:

- unchunked output against an independent Torch reference,
- every chunk output against an independent Torch reference,
- the final chunk output against the corresponding unchunked output slice,
- the final MLA cache state against the unchunked cache state,
- warm throughput and peak memory.

## Results

All six cases pass the unchanged numerical gate for every comparison. Final
cache state is bit-exact in all cases. Final output is bit-exact in the 128,
256, and 512 token cases; the 64, 768, and 1024 token cases remain within the
unchanged gate with maximum absolute errors of `1.52587890625e-05`,
`1.52587890625e-05`, and `3.0517578125e-05`, respectively.

Warm throughput is consistently lower for chunked execution than for unchunked
execution at these small sizes, as expected from extra kernel launches and
attention over a growing prefix. Peak memory grows with total context and
remains far below the 48 GB limit; the largest case peaks at 372,580,864 bytes.

## Installed-source baseline

`baseline-first.json` records the required early installed-source baseline. It
uses `torch.nn.functional.scaled_dot_product_attention` on a small causal
workload and compares against an independent float32 Torch formula. The first
GPU execution elapsed `0.7791734803467989` seconds, and the mean of three
synchronized warm timings was `5.539196232954661e-05` seconds. This baseline is
clearly labeled as installed-source context and is not proof for later checkout
changes.

## Limitations

- This is a reduced MLA attention probe, not a full DeepSeek decoder-block or
  full-model benchmark.
- The `triton` backend is the only backend exercised; no unsupported backend
  result is fabricated.
- Timing is wall-clock with explicit synchronization and is not a substitute
  for production serving throughput.
- The probe uses synthetic weights under 4 GB and does not download model
  weights.
