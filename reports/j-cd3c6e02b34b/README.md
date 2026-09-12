# Correction generation 2: selective DSA KV dequantization

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2856 at
`6fbcae3f6367d95c85a3f71d3105224145dee8f4`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/2909

Upstream issue: https://github.com/sgl-project/sglang/issues/36338

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2944

## Result

The candidate's valid selective dequantization implementation and its four GPU
regression tests are preserved. Independent execution reproduced the review's
remaining policy defects in substance on the assigned `gfx950` GPU: at 131,072
prefix rows and top-k 2,048, random 8 through 32-query compact cases took
1.012x through 1.285x the separately warmed full-path latency. At 33 completely
overlapping query rows, the default gate allocated the 150,994,944-byte full
buffer, while forcing the union produced 2,048 rows at a 4,930,048-byte peak.

No further source correction is claimed. An explored pre-gate overlap check
required a GPU-to-host synchronization: random fallback cases became
1.113x--1.134x full latency and the shared compact case became 1.092x full
latency. That approach was reverted. Raising a global threshold from this AMD
microbenchmark would be speculative for the actual NVIDIA Hopper/Blackwell
`flashmla_sparse` path, which is unavailable here.

The candidate's proven positive case remains intact: 262,144 logical rows and
8 queries selected 4,029 compact rows, used 4,641,408 BF16 bytes instead of
301,989,888, and ran 2.540x faster. Its four numerical/remapping GPU tests pass.

## Evidence

- `evidence/candidate_boundaries.log`: exact-candidate boundary reproduction.
- `evidence/candidate_beneficial.log`: preserved large-prefix beneficial case.
- `evidence/pytest_candidate.log`: four candidate GPU tests passing.
- `evidence/corrected_boundaries.log`: rejected synchronization-based
  experiment, retained to explain why it was not shipped.
- `evidence/dsa_integration_rocm.log`: architecture blocker before attention;
  the HIP legacy DSA path requires page size 1 but the fixture uses page size
  64.

No native source changed, so a native rebuild was not applicable. The checked
out Python/Triton source was compiled using private Triton caches and executed
on the assigned GPU. BF16 FlashMLA attention output correctness and end-to-end
sparse-prefill performance remain unverified on NVIDIA Hopper/Blackwell.
