# Independent review of PR 2959

Candidate: https://github.com/amdpilot-org/sglang/pull/2959 at `642e52a8e98ede176a3c4942bd18f926b40b3434`

Upstream issue: https://github.com/sgl-project/sglang/issues/16255

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2904

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2998

## Verdict

Recommendation: **request changes**. The patch is a credible, behavior-preserving partial extraction, but `fully_resolves_original` is false. It extracts DSA indexer construction and pipeline top-k policy into `deepseek_common/v32_mixin.py`; it does not complete the original issue's full hardware/backend structure, and real V3.2 runtime behavior remains unverified.

## Evidence

At the recorded base, the requested module cannot be imported and the relevant policy remains inline in `deepseek_v2.py`. At the exact candidate, imports resolve to the checked-out source, the candidate's focused suite passes, and an independent truth-table test passes for skip-topk, NextN, K-pool, MHA/MLA-related policy boundaries, import compatibility, and empty top-k device preservation.

The independent GPU portion ran on one AMD Instinct MI355X (`gfx950`) using Torch 2.11.0+rocm7.2. It proves only that the helper preserves the ROCm device, dtype, and shape. No model weights were available, so it does not prove Indexer kernel numerics, checkpoint loading, logits, serving, or semantic accuracy. No distributed pipeline boundary was executed.

No native source changed and the prepared environment declares no native target, so a native rebuild was not applicable.

Raw commands and output are retained under `evidence/`; structured conclusions and remaining counterexamples are in `result.json`.
