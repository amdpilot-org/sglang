# Eager versus captured block replay on MI300X

## Scope

This is a reduced-block study for `sgl-project/sglang` issue 30344.  It uses the
real installed Triton speculative verifier
`sglang.kernels.ops.speculative.reject_sampling.chain_speculative_sampling_triton`
with synthetic target probabilities and chain draft trees.  It does **not**
claim a full-model speedup and does not build a new graph framework.

The measured block is:

1. `torch.nn.functional.softmax` over float32 logits.
2. `chain_speculative_sampling_triton` verification and final sampling.

Eager execution is compared with `torch.cuda.CUDAGraph` replay using static
buffers.  Every case is checked against an independent CPU implementation of
the chain verifier semantics.

## Installed-source baseline

Before cloning or editing, the installed source at
`/sgl-workspace/sglang` was used for a bounded GPU baseline:

```bash
cd /sgl-workspace/sglang/test/registered/cuda_graph/breakable
PYTHONPATH=/sgl-workspace/sglang/python /opt/venv/bin/python -m unittest \
  test_breakable_cuda_graph.TestBreakableCUDAGraphBasic
```

Result:

- 9 tests passed.
- One-shot wall time: 34.804 s.
- Test path:
  `/sgl-workspace/sglang/test/registered/cuda_graph/breakable/test_breakable_cuda_graph.py`.
- Installed source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Installed `sglang` version: `0.5.18.dev20260826+g937af8538b`.

This baseline is clearly labeled as installed-source evidence only; it is not
proof for later checkout changes.  The full record is in
`/job/baseline-first.json`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`.
- Local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- GPU: one AMD Instinct MI300X, `gfx942`, capability `9.4`.
- GPU UUID: `37386364-3065-3365-3637-316234323939`.
- Python: `/opt/venv/bin/python`, version `3.10.12`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Triton: `3.7.0`, path
  `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- SGLang source path: `/job/sglang/python/sglang`.
- Verifier source:
  `/job/sglang/python/sglang/kernels/ops/speculative/reject_sampling.py`.
- Branch: `amdpilot/j-a27f0bc48bd2`.
- Commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- PR base (`main`): `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Method

The benchmark uses six fixed cases:

| Case | Batch | Draft slots | Vocab |
|---:|---:|---:|---:|
| 1 | 1 | 5 | 1024 |
| 2 | 2 | 9 | 2048 |
| 3 | 4 | 17 | 4096 |
| 4 | 8 | 33 | 8192 |
| 5 | 16 | 65 | 16384 |
| 6 | 32 | 129 | 32768 |

Dtypes:

- `logits`, `target_probs`, `draft_probs`, `uniform_samples`: float32.
- `predicts`, `accept_index`, `accept_num`: int32.
- `candidates`, `retrive_index`, `retrive_next_token`,
  `retrive_next_sibling`: int64.

Accuracy gate:

- Exact equality for `predicts`, `accept_index`, and `accept_num`.
- Eager and graph outputs are also compared directly.
- Three fresh input values are used per case, with logit offsets `0.0`, `0.25`,
  and `0.5`.

Timing:

- 10 warmup complete-block executions.
- 3 measured batches of 50 complete-block executions.
- CUDA events bracket each batch.
- The median batch mean is reported as milliseconds per complete block.
- No artificial burn, unbounded loop, or sleep loop is used.

Static-buffer reuse:

- Input logits, target probabilities, draft probabilities, uniforms, tree
  metadata, and outputs are allocated once per case.
- Eager and graph paths use separate output tensors.
- Graph capture records the same softmax-plus-verifier block using those static
  buffers.

## Results

| Case | Batch × slots × vocab | Eager ms/block | Graph ms/block | Eager/graph |
|---:|---:|---:|---:|---:|
| 1 | 1 × 5 × 1024 | 0.077552 | 0.027399 | 2.830465 |
| 2 | 2 × 9 × 2048 | 0.071748 | 0.023646 | 3.034283 |
| 3 | 4 × 17 × 4096 | 0.051192 | 0.026435 | 1.936572 |
| 4 | 8 × 33 × 8192 | 0.040262 | 0.045745 | 0.880123 |
| 5 | 16 × 65 × 16384 | 0.124486 | 0.131455 | 0.946986 |
| 6 | 32 × 129 × 32768 | 0.829686 | 0.837241 | 0.990976 |

Capture was supported for all six cases.  Graph replay improved the three
smallest cases, but was slightly slower than eager for cases 4–6.  Capture
overhead ranged from 8.485 ms to 61.401 ms and is not included in the
steady-state per-block timing.

Raw results are in `reports/j-a27f0bc48bd2/results.json`.

## Limits

- Wall limit: 7200 s.
- Actual benchmark elapsed time: 7.601 s.
- Maximum generated weights: 1.512 GiB, below the 4 GiB limit.
- Maximum peak CUDA allocation: 2.016 GiB, below the 48 GiB live-allocation
  limit.
- Exactly six workload cases were used.

## Unsupported capture boundaries

The captured block excludes these unsupported or intentionally eager
boundaries:

- Fresh CPU input generation and host-to-device copies.
- Independent CPU reference execution.
- Output snapshots from GPU to CPU.
- Host synchronization and scalar reads.
- `torch.multinomial` and other dynamic host-side sampling.

No unsupported boundary was observed inside the softmax-plus-verifier block.

## Reproduction

From the repository root:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-a27f0bc48bd2/bench_speculative_graph_replay.py \
  --output reports/j-a27f0bc48bd2/results.json
```

The script requires exactly one CUDA/ROCm device and enforces the case count,
weight limit, live-allocation limit, and wall-clock limit listed above.

## Context and limitations

Upstream issue 30344 is a DSpark roadmap tracker.  Its comments point to the
CUDA-graph robustness roadmap in issue 34297 and related PRs 31195, 32183,
32467, 33795, 34286, and 34410.  This change does not duplicate those fixes; it
adds a bounded measurement and report only.

Limitations:

- Draft trees are chains, not arbitrary branching trees.
- The native `sgl_kernel.tree_speculative_sampling_target_only` op is not
  registered in this installed wheel, so the installed Triton chain verifier is
  used instead.
- This is a reduced-block study, not a full-model or end-to-end serving
  benchmark.
- No production runtime code is changed.
