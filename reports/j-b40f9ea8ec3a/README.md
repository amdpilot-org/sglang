# Independent review of PR 958

Upstream issue: https://github.com/sgl-project/sglang/issues/39072

Mirror issue: https://github.com/amdpilot-org/sglang/issues/992

Candidate: https://github.com/amdpilot-org/sglang/pull/958 at exact commit
`fb6171b515c97df55661f9de12c61f63ba25bdc3`.

## Verdict

Recommendation: **accept**, as a narrow, source-level correction for the
reproduced `cal_padded_tokens` / `pad_dsa_cache_seqlens` shape mismatch.
The candidate is not proof that the full original deployment is fixed, so
`fully_resolves_original` is false.

On the recorded base, the candidate regression failed three shape cases: an
issue-class SUM_LEN case returned 9 rows instead of 12, a second DP rank
returned 6 instead of 8, and MAX_LEN returned 9 instead of 12. At the exact
candidate commit all four regression tests passed. An independent exhaustive
probe passed 100 combinations across SUM_LEN/MAX_LEN, DP ranks, TP widths
1/2/3/4/8, zero/aligned/unaligned token counts, two-dimensional metadata,
prefix preservation, and zero-filled padding. A real gfx950 tensor probe also
expanded 9 rows to 12 and preserved/zero-filled the expected regions.

The source import resolved to the checked-out repository at
`/job/repo/python/sglang/srt/layers/attention/dsa/utils.py`; Torch resolved to
the prepared environment at `/opt/venv/lib/python3.12/site-packages/torch`.
The candidate changes Python only, so no native rebuild was applicable. The
exact base-to-candidate diff also passes `git diff --check`, confirming that
the earlier trailing-whitespace evidence defect was corrected.

## Scope and limitations

This review used one AMD Instinct MI350X (gfx950), ROCm 7.2, and Torch
2.11.0+rocm7.2. It could not run the reported eight-NVIDIA-H20 setup, CUDA
FlashMLA kernel, GLM-5.3-W4AFP8 model, Mooncake/InfiniBand disaggregation,
TP=8/DP=2 synchronization, or EAGLE warmup/serving path. Therefore the result
is a verified partial fix for the directly exercised metadata-sizing contract,
not verification of complete resolution of the original issue.

Raw commands, exit codes, import paths, diffs, and outputs are retained in
`raw/`.
