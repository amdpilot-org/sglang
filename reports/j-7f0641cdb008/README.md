# Independent review of PR 2641

Candidate: https://github.com/amdpilot-org/sglang/pull/2641 at exact commit
`e04feb629fd9333d65b9a3f25f421cad96528d50`.

Upstream issue: https://github.com/sgl-project/sglang/issues/26794

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2644

## Recommendation

Accept as test-only hardening. The candidate adds no production change. Its
three focused tests pass at the exact candidate commit, but they also pass
unchanged on the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.
This is consistent with the production correction already being present on the
base through merged upstream changes #26717 and #29503.

The test is relevant: it takes the NPU branch of
`UnquantizedFusedMoEMethod.process_weights_after_loading`, mocks only the
unavailable `npu_format_cast`, then uses the real `FusedMoE._load_w13` and
`_load_w2` methods to reload separate gate, up, and down tensors. It verifies
canonical persistent parameter shapes and independent replacement of both
halves of `w13_weight`.

## Independent evidence

- Exact candidate regression: 3 passed.
- The candidate test copied outside the checkout and run against the recorded
  base: 3 passed. This establishes that the candidate is coverage, not a new
  original-issue source fix.
- On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), independent
  loader checks passed for intermediate sizes 1, 3, and the reported 1408,
  with two successive reload cycles and numerical comparisons after each load.
- Reconstructing the obsolete persistent-transpose layout made the real loader
  fail with a tensor dimension mismatch. This reproduces the reported failure
  mechanism, but not the original Ascend scheduler crash.
- Imports resolved to `/job/repo/python/sglang`; PyTorch was
  `2.11.0+rocm7.2`. `torch_npu` was not installed.

## Scope and limitations

`fully_resolves_original` is false as a review claim. There was no Ascend
device, CANN installation, or `torch_npu`, so actual `npu_format_cast` and
Ascend grouped matmul were not executed. The reported DeepSeekV3.2 checkpoint
was unavailable, so neither a full Engine call nor the HTTP
`update_weights_from_disk` route established scheduler survival. The AMD GPU
run validates the generic tensor loader only and cannot qualify NPU kernels or
the complete serving path.

No native source changed in the candidate, and the prepared environment lists
no native rebuild target, so no native rebuild was applicable.

Raw commands and output are retained under `raw/`.
