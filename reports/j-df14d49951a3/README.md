# gfx942 speculative verifier chunk-partition study

This report-only study measures the installed `sgl_kernel` greedy tree verifier
and tree-builder primitives on one AMD Instinct MI300X (`gfx942`). It uses
locally generated synthetic target probabilities and deterministic binary draft
trees, and compares unchunked execution with bounded row-chunk partitions.

## Scope

- Uses the installed `sgl_kernel::build_tree_kernel_efficient` and
  `sgl_kernel::verify_tree_greedy` GPU primitives.
- Uses synthetic target probabilities to derive target predictions.
- Uses two workloads and six total partition cases:
  - small: batch 8, 7 tree tokens, vocabulary 128, partitions 1/2/4
  - large: batch 2048, 15 tree tokens, vocabulary 2048, partitions 1/8/16
- Compares every chunked result with the unchunked result and with an
  independent CPU greedy-tree reference.
- Does not claim a full-model speedup, distributed result, or CUDA-graph result.

## Reproduce

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-df14d49951a3/chunk_partition.py \
  --output reports/j-df14d49951a3/results.json
```

The installed-source first GPU baseline is in `baseline-first.json` at the job
workdir root and is copied here for review. It is not evidence for checkout
changes.

## Results

| Workload | Chunks | Median ms | Useful tokens/s | Peak allocated |
|---|---:|---:|---:|---:|
| small | 1 | 0.011707 | 2,050,055.53 | 35,840 B |
| small | 2 | 0.021529 | 1,114,775.41 | 40,448 B |
| small | 4 | 0.031492 | 762,086.22 | 44,544 B |
| large | 1 | 0.011467 | 714,428,967.03 | 253,574,656 B |
| large | 8 | 0.062665 | 130,727,921.25 | 253,984,256 B |
| large | 16 | 0.115806 | 70,738,690.55 | 253,984,256 B |

See `results.json` for exact dimensions, dtypes, timing samples, medians,
standard deviations, IQRs, memory measurements, useful work, and equality gates.

All cases matched the independent CPU reference exactly. All chunked cases
matched their unchunked workload exactly, including final output state and
useful work. Timing uses CUDA events around the verifier kernel calls only;
output reset and chunk-result mapping are outside the timed region.

## Boundaries

- The installed stochastic tree-sampling and top-k renormalization ops are not
  registered in this environment. They are recorded as unsupported and were not
  fabricated.
- The study uses the installed greedy verifier and tree builder only.
- No model weights, full-model execution, distributed execution, or CUDA-graph
  replay is included.
