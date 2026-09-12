# Investigation of sglang#36081

Upstream issue: https://github.com/sgl-project/sglang/issues/36081

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1251

## Result

The prepared `main` source already contains the issue-specific solution. No new
runtime correction is justified at base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

The report's idle invariant is not evidence of leaked capacity: its full pool
line has `available=768` and `evictable=9115392`, which sum exactly to
`total=9116160`. The crash came from the older 0.5.18 free/accounting path used
by hybrid SWA plus paged allocation. The source issue's follow-up identifies
upstream PRs 37729 and 37876 as the fix. Their relevant changes are present in
this checkout:

- page-aligned `free_segment(s)` releases physical page representatives once;
- DCP over-allocation cleanup aligns with the allocator page size;
- hybrid-SWA full-side row releases use the segment API rather than token-level
  `free()`, including grouped and unified-cache paths;
- debug checks reject shared-page segments and cross-call double frees.

The upstream merge objects were fetched only for comparison. Raw diffs from
their first parents are retained outside the worktree at
`/tmp/amdpilot-repo-j-370c677a68a1/evidence/pr37729-relevant.diff` and
`/tmp/amdpilot-repo-j-370c677a68a1/evidence/pr37876-relevant.diff`. Those diffs
show the old boundary-trim/token-free behavior and the regression added with
the correction. The current focused regression
`test_overallocated_tail_uses_allocator_page_size_under_dcp` exercises the
reported DSPARK/DCP double-release boundary and passes.

## Validation and limits

- The two allocator regression files pass: 49 tests and 44 subtests.
- The issue-specific DCP/DSPARK over-allocated-tail regression passes alone.
- On the assigned single AMD Instinct MI355X (`gfx950`), three independent
  page-aligned boundary cases matched a `torch.unique` page-ID reference
  exactly. Raw output is retained at
  `/tmp/amdpilot-repo-j-370c677a68a1/evidence/gpu_page_free_reference.log`.
- No native code was changed or rebuilt.

The original H20 x8 DeepSeek-V4-Flash/DSPARK serving workload was not reproduced:
the assigned host has one AMD gfx950 GPU and the reported model weights were not
provided. The GPU fixture validates allocator execution only; it does not
qualify the original model architecture, CUDA/H20 behavior, tensor parallelism,
or long-running serving stability.
