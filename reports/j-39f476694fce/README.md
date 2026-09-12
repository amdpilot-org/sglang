# Independent review of PR 1772

Reviewed exact commit `3bbdf389dd68d4110dafe780d4e9ea90479c97e7` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/34384
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1823
- Candidate: https://github.com/amdpilot-org/sglang/pull/1772

## Verdict

Recommendation: **accept as test-only hardening**, not as a newly demonstrated full fix of the original issue. `fully_resolves_original` is false.

The candidate changes only tests and review artifacts. Its new integration test directly invokes the runner staging method, verifies stable captured `verify_lens` and `qo_indptr` pointers, covers the original 32 requests x 6 tokens geometry, covers 32 x 5 tokens with 32 nonzero synthetic slots, and passes the staged values into the DSV4 resolver. Independently monkeypatching staging to a no-op made all three integration cases fail.

The prepared recorded base already has the production correction in the relevant runner, ragged-layout, and DSV4 files. Those files are identical between the earlier correction commit `0a0b787c4dbc878065c48926d18c5b43eafcf19a` and the recorded base. Consequently, the original failure could not be reproduced on this base as requested; the passing base is evidence of an already-integrated correction, not failing-before evidence for PR 1772.

## Results

- Recorded base helper suite: 10 passed.
- Exact candidate focused suite: 15 passed.
- Staging-disabled sensitivity check: 32x6, 32x5, and DSV4 cases all failed as expected.
- Candidate gfx950 script: passed for 32x6 and 32x5.
- Independent gfx950 adversarial probe: passed for seven admitted tier-192 uniform/irregular layouts; rejected an over-tier 210-token layout.

The DSV4 resolver preserves staged values but creates a derived layout, so the candidate proves value flow through `_resolve_verify_layout`, not pointer identity across that resolver. Stable pointer assertions apply to the runner's captured buffers.

## Architecture limits

Execution used one AMD Instinct MI355X (`gfx950`) under ROCm 7.2. The reported environment requires four NVIDIA H20 GPUs, CUDA/Hopper graphs, TP4, and unavailable DeepSeek-V4-Flash DSpark weights. Actual DSV4 metadata kernels, attention/compressor/indexer execution, distributed collectives, and the reported first CUDA Graph replay remain unverified. No native source changed, so no native rebuild was applicable.

See `raw/` for concise command transcripts and `result.json` for the structured review result.
