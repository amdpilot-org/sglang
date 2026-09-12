# Independent review of PR 2608

Reviewed `https://github.com/amdpilot-org/sglang/pull/2608` at exact commit
`8bb43fd1dd0ad1d4910508a98bfa464c86b0835c` against upstream issue
`https://github.com/sgl-project/sglang/issues/26794`.

## Finding

Request changes. The candidate is test-only hardening: it adds no production or
native source change. Its three focused tests pass and independently exercise
the correct canonical-layout reload behavior. The prepared base already has the
production correction that keeps NPU MoE parameters in canonical `[E,N,K]`
layout and passes transpose semantics to grouped matmul at execution.

The candidate does not fully verify the original issue because this host has no
Ascend NPU, CANN, or `torch_npu`, and the reported DeepSeekV3.2 checkpoint was
not available. Consequently neither real `npu_format_cast`/grouped matmul nor
full Engine/HTTP scheduler survival after `update_weights_from_disk` was run.

The candidate also changes the retained GPU identity from MI350X to MI355X, but
in this prepared review environment both PyTorch and `rocm-smi` report **AMD
Instinct MI350X** with `gfx950:sramecc+:xnack-`. Thus its corrected raw record
and result metadata do not describe the device assigned to this review.

## Evidence

- Exact candidate regression: 3 tests passed.
- Independent one-GPU check at the issue's 1408-row size: two consecutive
  separate gate/up reload generations replaced both fused halves correctly.
- Independent TP=2 check: both ranks selected the correct source slice and
  populated the correct gate/up halves.
- Recreating the obsolete persistent transpose caused the actual `_load_w13`
  copy to fail, while the canonical prepared-base layout passed.
- No native files changed, so no native rebuild was applicable.

Raw command output is retained in `raw/`.
