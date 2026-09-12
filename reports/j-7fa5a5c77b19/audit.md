# `torch.nonzero` call-site audit

Upstream issue: https://github.com/sgl-project/sglang/issues/9889

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3449

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Environment: PyTorch 2.11.0+rocm7.2, ROCm 7.2.26015, one AMD Instinct
MI355X (`gfx950`). The prepared interpreter was
`/tmp/amdpilot-repo-j-7fa5a5c77b19/venv/bin/python`.

## Findings by originally reported call site

### `python/sglang/srt/configs/janus_pro.py:479`

`process_one` turns the tokenizer's Python list into `torch.LongTensor(input_ids)`
without a device argument. Dispatch instrumentation of the actual method recorded
the `aten::nonzero` input as CPU, shape `[5]`, dtype `bool`. Its downstream
`add_image_token` loop also performs Python iteration and slicing with every
index, so moving just this operation to the GPU would not make the path
asynchronous.

This site is unchanged. A controlled CPU comparison (11 rounds) found
`nonzero(as_tuple=True)[0]` consistently slower than the existing form:

| Input | Existing median | Tuple-form median |
| --- | ---: | ---: |
| empty | 2.102 us | 2.539 us |
| length 16, 2 matches | 2.340 us | 2.876 us |
| length 4096, 64 matches | 5.716 us | 6.178 us |
| length 4096, all match | 8.027 us | 8.521 us |

Empty, sparse, and dense comparisons returned the same ascending indices,
`int64` dtype, and CPU placement. The tuple form intentionally has shape `[N]`
instead of the current `[N, 1]`; changing that public-to-helper shape offered no
measured benefit.

### Former `python/sglang/srt/models/gemma3_mm.py:212`

The reported `(positions == 0).cpu().nonzero()` no longer exists. Current main
contains `prepare_attn_masks`, but it was replaced by the bidirectional image
attention implementation merged in upstream PR 10707 and is called from
`forward`. That current implementation contains no `nonzero`. No edit or
runtime model-weight validation was applicable to the stale call site.

### `python/sglang/srt/models/phi4mm_utils.py:84`

The production caller builds `chunk_start_idx` with NumPy, and
`adaptive_enc_mask` explicitly constructs CPU tensors (`torch.Tensor`,
`torch.arange` without device arguments). Thus the reported operation did not
synchronize the assigned GPU in the actual path.

The operation nevertheless materialized an `[x_len, num_chunks + 1]` interval
matrix solely to recover one ordered chunk index per sequence position.
`torch.bucketize(..., right=True)` computes exactly that fixed-size mapping from
the sorted first-position boundaries without `nonzero` or the temporary matrix.
The focused tests compare against the prior implementation for empty, singleton,
sparse, dense, exact-boundary, tail, and left/right-window cases. They verify the
same ascending index order and final shape, boolean dtype, CPU device, and exact
values.

Full-function CPU timings used 11 controlled rounds:

| `x_len` (chunk size 18) | Existing median | Bucketize median |
| ---: | ---: | ---: |
| 18 | 26.750 us | 23.343 us |
| 64 | 29.209 us | 25.756 us |
| 256 | 146.379 us | 197.265 us |
| 1024 | 337.870 us | 177.682 us |
| 4096 | 623.917 us | 460.993 us |

The 256-position case was slower in the final run, while the shorter boundary
cases and larger masks improved. The output mask allocation is still quadratic
and dominates large inputs; this change only removes the extra interval matrix
used to derive `idx`. Earlier retained runs showed the 256 case near-neutral,
so no claim is made that every input size becomes faster.

## GPU synchronization control

To validate the issue's premise independently of the actual CPU placements, a
100,000,000-cycle GPU sleep was queued immediately before each tested operation.
Inputs and bucket boundaries were allocated before the sleep. Eleven rounds gave:

| Operation | Host-call median | Meaning |
| --- | ---: | --- |
| `nonzero()` | 41.537 ms | waited for pending GPU work |
| `nonzero(as_tuple=True)[0]` | 41.527 ms | waited for pending GPU work |
| `bucketize(..., right=True)` | 0.0044 ms | returned asynchronously |
| elementwise add control | 0.0049 ms | returned asynchronously |
| sleep plus explicit synchronize | 41.515 ms | calibrated pending work |

This proves that tuple-form `nonzero` is not a synchronization fix on this ROCm
GPU. It also confirms why leaving the CPU-only Janus site alone is preferable and
why the fixed-size Phi replacement is materially different.

## Scope and limitations

- The audit addresses all three paths in the original issue snapshot. It does
  not claim to optimize the many unrelated `nonzero` calls added elsewhere in
  the repository since that report.
- Janus was exercised with the real processor method and deterministic fake
  tokenizer/image-processor inputs; no Janus model weights were needed.
- Phi was tested directly through the real `adaptive_enc_mask` implementation.
- The removed Gemma call was verified from source and history only. No Gemma
  model-weight or serving accuracy run was performed because the current source
  has no corresponding operation to profile.
- No native code changed, so no native rebuild was required.

Raw command output is retained in the job runtime directory at
`/tmp/amdpilot-repo-j-7fa5a5c77b19/evidence/`.
