# Independent review of PR 2856

Candidate: https://github.com/amdpilot-org/sglang/pull/2856 at exact commit
`6fbcae3f6367d95c85a3f71d3105224145dee8f4`

Upstream issue: https://github.com/sgl-project/sglang/issues/36338

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2834

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2893

## Recommendation

Request changes. The candidate fixes the three concrete performance regressions
reported by PR 2799, and its isolated dequantization/remapping tests pass, but it
does not fully resolve the original feature request's performance contract.

The original base at `358c163250ad3b1f62939b01ce1314a0a31a0365`
unconditionally dequantizes every flattened prefix row. It has no selective
helper, and importing `dequantize_k_cache_paged_selective` fails. The candidate
adds the helper and integrates it into the BF16 `flashmla_sparse` prefix path.

On the assigned AMD Instinct MI355X, the candidate's new entry-count gate sends
the inherited 64, 256, and 1,024 query-row cases to full dequantization before
`torch.unique`. Each used the full path's 150,994,944-byte peak and ran at
0.890x, 0.872x, and 0.875x of the separately warmed full baseline in this run.
The preserved 262,144-row/8-query compact case reduced 262,144 logical rows to
4,029 rows and ran 2.541x faster.

Independent boundary testing found remaining counterexamples at a 131,072-row
prefix and top-k 2,048. Random selections with 1, 4, 8, 16, 24, and 32 query
rows all selected the compact path but measured 1.011x, 1.020x, 1.060x, 1.210x,
1.327x, and 1.347x the warmed full-dequantization latency. These results retain
large staging-memory savings, but contradict a general performance benefit at
the candidate's 131,072-row threshold.

The gate also uses entry count without considering overlap. At 33 query rows,
67,584 repeated selections of the same 2,048 logical rows fell back immediately
to the 150,994,944-byte full buffer. Bypassing only the gate with
`max_topk_ratio=1.0` produced the 2,048-row compact result at a 4,930,048-byte
peak and essentially equal latency (0.993x full). This is evidence that the
hard gate avoids the inherited regressions by rejecting some memory-beneficial
inputs, rather than establishing a generally beneficial selection policy.

## Correctness and architecture limits

The four candidate GPU unit tests passed against the exact commit. They cover
an independent packed-FP8 numerical reference, physical alias deduplication and
remapping, fallback, and invalid-index sentinels. The imported implementation
was `/job/repo/python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py`.

The assigned device is AMD Instinct MI355X (`gfx950`) with Torch
2.11.0+rocm7.2. It is not NVIDIA Hopper or Blackwell, where
`flashmla_sparse` runs. The repository integration fixture stopped before
attention because its page size is 64 and the available legacy HIP DSA path
requires page size 1. Therefore BF16 FlashMLA output correctness and end-to-end
sparse-prefill performance are unverified. No serving smoke can substitute for
that missing kernel execution.

No C++, CUDA, HIP, or FlyDSL source changed in the candidate, so a native
library rebuild was not applicable. A fresh private Triton cache was used and
the checked-out Triton dequantization kernel compiled and executed on the GPU;
the cache inventory is retained in `evidence/triton-cache-files.txt`.

## Evidence

- `evidence/base-original-failure.log`: recorded base call site and failed
  selective-helper import.
- `evidence/candidate-regression.log`: exact-candidate GPU regression tests.
- `evidence/candidate-adversarial.log`: inherited 131,072-row counterexamples
  after the correction.
- `evidence/candidate-beneficial.log`: preserved 262,144-row compact case.
- `evidence/independent_boundaries.py` and its log: independently authored
  boundary and high-overlap cases.
- `evidence/dsa-integration-rocm.log`: full pre-attention architecture blocker.
- `evidence/candidate-code.diff`: reviewed implementation and tests only.
