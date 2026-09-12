# Consolidated correction for unified-memory exact-KL coverage

Candidate: https://github.com/amdpilot-org/sglang/pull/2877 at exact commit `d8788d51d7c567fea6c0dd8ccbf64e40f8dd5f03`

Independent review: https://github.com/amdpilot-org/sglang/pull/2963

Upstream issue: https://github.com/sgl-project/sglang/issues/34899

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3000

## Result

The candidate's six unified-memory cases are preserved. They now request direct
elementwise logprob equality instead of relying solely on averaged KL below
`1e-9`. Existing models that need numerical tolerances retain the legacy
default behavior.

The review's adversarial inputs were independently reproduced. Unequal
logprobs `[0.0]` and `[-1e-5]` produce
`avg_kl_div=5.0000069648240756e-11` and pass the candidate/default comparator.
The regression tests prove exact mode rejects that case and a `1e-12`
difference whose computed KL rounds to zero, while accepting identical values.

## Commands

```bash
/tmp/amdpilot-repo-j-c44ff39e0704/venv/bin/python -c "from sglang.test.kl_test_utils import compare_kl_divergence; compare_kl_divergence([[0.0]], [[-1e-5]], {'model': {'kl_div': 1e-9}}, 'model', 'candidate_default')"

/tmp/amdpilot-repo-j-c44ff39e0704/venv/bin/python -m pytest -q test/srt/test_kl_test_utils.py

/tmp/amdpilot-repo-j-c44ff39e0704/venv/bin/python -m pytest --collect-only -q test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_hybrid_bitexact.py
```

## Remaining limitation

This host has one AMD Instinct MI350X (`gfx950`) under ROCm 7.2. Importing
Inkling reaches its NVIDIA CUTE path and fails because `cutlass` is unavailable.
No qualifying Inkling weights or supported NVIDIA path were available.
Therefore neither Python nor Rust tree backend completed an end-to-end Inkling
request, no exact-zero GPU logprob result was measured, and no unified-memory
FULL, SWA, or MAMBA restoration fault was demonstrated revert-then-red or by
fault injection. The tiny Llama fixture cannot qualify this different hybrid
architecture. No speculative state-restoration source change was made.
