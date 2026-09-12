# Independent review of amdpilot-org/sglang PR 2736

Candidate reviewed exactly at `21e526df7844155adec0c2163992178bf899ba99`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a partial implementation,
not a full resolution of the original issue.

## Finding

The selective dequantization and index remapping stage is numerically correct in
the candidate's focused cases, and its 262,144-row / 8-query microbenchmark
reproduces a staging-path speedup. However, the default heuristic cannot satisfy
the requested "fallback when selective dequantization is not beneficial"
contract because it decides only after `torch.unique` has processed all
query-by-top-k entries.

On the assigned MI350X at the candidate's default 131,072-row threshold, an
independent warmed seven-repeat median test measured:

| Query rows | Top-k entries | Candidate decision | Candidate/full latency | Peak allocation |
|---:|---:|---|---:|---:|
| 8 | 16,384 | selective | 0.931x | 18,370,048 B |
| 64 | 131,072 | selective | 1.363x | 100,483,584 B |
| 256 | 524,288 | full fallback | 1.841x | 166,724,096 B |
| 1,024 | 2,097,152 | full fallback | 2.082x | 213,910,016 B |

The warmed full path was 0.454 ms with 150,994,944 B peak allocation. Thus the
candidate can choose a slower selective path, and its nominal fallback can be
more than twice as slow and allocate about 42% more peak memory than simply
using the existing full dequantization path. The heuristic accounts for prefix
rows and final union size, but not query count/top-k input size or the cost paid
to discover the union. This is directly tied to the original performance and
fallback contract.

## What was verified

- The prepared checkout initially matched the recorded base exactly and was
  restored to `amdpilot/j-2e5631a47cfa` before this report was committed.
- On the base, the actual paged function materialized all 131,072 BF16 rows
  (150,994,944 bytes) and exposed no selective API.
- At the exact candidate commit, its three GPU regression tests passed. They
  validate independent FP8 packing/dequantization, alias deduplication, remap,
  fallback output compatibility, and invalid sentinels.
- The candidate benchmark reproduced 262,144 logical rows to 4,029 compact rows,
  301,989,888 to 4,641,408 staged BF16 bytes, and 0.7912 to 0.3099 ms (2.553x).
- Runtime imports resolved to `/job/repo/python/sglang/...`, so both base and
  candidate runs used the checked-out source rather than an installed copy.
- No C++, CUDA, HIP, FlyDSL, or other native source changed. A native rebuild was
  therefore not applicable; the changed Triton kernel path compiled/executed
  through the private job cache.

## Architecture limits

The assigned device is an AMD Instinct MI350X (`gfx950`) with ROCm 7.2 and Torch
2.11.0+rocm7.2. It can execute and numerically validate the changed Triton
dequantization stage, but the requested BF16 `flashmla_sparse` attention kernel
is NVIDIA Hopper/Blackwell-only. The repository's real DSA FP8 prefill attempt
also stops before attention because its fixture uses page size 64 while the
prepared legacy HIP DSA path requires page size 1 (2 failed, 1 passed, 4 skipped).
Consequently, target-architecture end-to-end attention correctness and
end-to-end sparse-prefill performance remain unverified.

Raw commands, scripts, issue/PR snapshots, and outputs are retained in
`evidence/`.
