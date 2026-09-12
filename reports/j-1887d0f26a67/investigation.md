# Investigation: SGLang issue 32569

Upstream issue: https://github.com/sgl-project/sglang/issues/32569

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2027

## Finding

The prepared base already contains the source fix for the reported ROCm
`top_k_renorm_prob` crash. No production-code correction was necessary.

The original failure called an unavailable optional sampling function directly.
On the prepared base, `dflash_utils.py` instead:

- imports `top_k_renorm_probs_triton` and `top_p_renorm_probs_triton` on HIP;
- routes calls through `_dflash_top_k_renorm_prob` and
  `_dflash_top_p_renorm_prob`; and
- retains PyTorch fallbacks if either optional function is `None`.

The related upstream changes are:

- https://github.com/sgl-project/sglang/pull/32621, merged 2026-07-28,
  which added the ROCm top-p path;
- https://github.com/sgl-project/sglang/commit/e15fe26912b86b226537f93840b8b923e790f996
  (`[AMD] Add triton topk renorm kernel and enable renorm CI unittest
  (#32641)`), which added the ROCm top-k implementation and wired it into
  `dflash_utils.py`; and
- https://github.com/sgl-project/sglang/pull/32541, whose merged history
  contains both changes.

The reporter also confirmed in the upstream issue that the updated image no
longer crashed under their original Kimi-K3 workload.

## Validation

`test_dspark_sampling_renorm.py` runs on the assigned AMD Instinct MI355X
(gfx950) and covers two independent paths:

1. The actual imported ROCm Triton top-k and top-p kernels are checked against
   explicitly constructed expected distributions.
2. Both optional renormalizer globals are forced to `None`, reproducing the
   precondition behind the reported `NoneType` call. The production DSPARK
   target-probability builder must still succeed and is compared against an
   independent sort/mask/cumulative-mass PyTorch reference. Per-request cases
   cover top-k 1, an interior value, a value above vocabulary size, active
   top-p filtering, and top-p 1.0.

Raw output: `reports/j-1887d0f26a67/raw/pytest_dspark_sampling_renorm.log`.

## Limitations

The original Kimi-K3 checkpoint and DSpark draft weights were unavailable, and
the assigned environment has one MI355X rather than the reported eight-MI350X
TP8 system. Therefore this investigation validates the issue-specific GPU
sampling/verify boundary, not full-model startup, semantic accuracy, long-run
serving behavior, or a distributed workload. No native C++ or FlyDSL source was
changed or rebuilt.
