# Independent review of BWAP candidate ee5f9f9

Reviewed `amdpilot-org/sglang` PR 2760 at exact commit
`ee5f9f9b83040ce90406872a74557bc439dfab67` against the open upstream feature
request and arXiv 2608.14003. The prepared base was exactly the recorded commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Request changes. The candidate is a substantial partial implementation: its
CLI is default-off, its focused suite passes, its source checkout is imported,
and the reduced-width GPU arithmetic agrees with an independent dense masked
reference on the assigned AMD Instinct MI350X. It does not faithfully implement
the original method's scoring contract, however:

1. Equation 2 computes one score for an exploration phase using column-wise L2
   pooling across all `T_E` tokens (and normalizes by `sqrt(T_E)`). The candidate
   calls `compute_decode_scores` once per decode step and folds those vectors
   into memory with element-wise maximum. That is max-over-time, not the required
   phase score. The retained neuron differs in the recorded four-token case.
2. The method defines `k = floor((1-rho) * D_FF)`. Candidate source and tests use
   Python `round`, producing a different retained cardinality for fractional
   products (for example, `D_FF=3`, `rho=0.5`: required 1, candidate 2).

Accordingly this is not a full original-issue fix. The candidate's server smoke
and random tiny-model evidence establish transport/engine execution only and do
not repair or independently qualify these algorithmic mismatches.

## Environment and limits

- Source interpreter: `/tmp/amdpilot-repo-j-ee8d5c34faef/venv/bin/python`
- Torch `2.11.0+rocm7.2`, HIP `7.2.26015`
- One AMD Instinct MI350X (`gfx950`) was visible. The candidate report names an
  MI355X, so this review records the device actually observed here.
- No C++, HIP, CUDA, FlyDSL, or other native source changed. A native rebuild was
  therefore not applicable; the already installed AITER native module was loaded.
- Production reasoning-model weights/corpus were unavailable. Semantic accuracy,
  batch-invariance on a real reasoning model, and production throughput remain
  unverified. The qualified tiny-Llama fixture cannot establish those properties.

Raw command output is retained in `evidence/`.
