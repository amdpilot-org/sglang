# Independent review of PR 800

Reviewed `https://github.com/amdpilot-org/sglang/pull/800` at exact commit
`995ac1056eef6bea1c17c728327c334268ef09e1` against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

The candidate is a narrow source-level correction for the observed shape
mismatch. `ForwardBatch.prepare_mlp_sync_batch` rounds each DP rank's query
count up to `attn_tp_size`; Eagle may construct DSA metadata before that
rounding. The base therefore retained 9 DSA metadata rows while the eventual
query had 12 rows. The candidate applies the same alignment in
`cal_padded_tokens` and produces 12 metadata rows (hence a `num_splits` length
of 13).

Evidence:

- The candidate's regression was copied outside the checkout and run on the
  recorded base: 3 failed and 1 passed. The issue-class case returned 9 rows
  instead of 12.
- At the exact candidate commit the same regression passed: 4 passed.
- An independent 97-case matrix exercised SUM_LEN, MAX_LEN, idle/zero ranks,
  uneven counts, attention TP widths 1/2/4/8, preservation/zero filling, and
  CP-interleave ordering. It passed on a GPU tensor on the assigned MI350X.
- Imports resolved to `/job/repo/python/sglang` and the changed
  `/job/repo/python/sglang/srt/layers/attention/dsa/utils.py`, not an installed
  wheel copy.
- No native source changed, so no native rebuild was applicable.

Recommendation is `request_changes`, despite the functional source-level fix:
the exact candidate records `git diff --check` as exit 0, but independently
running it from the recorded base reports trailing whitespace in the
candidate's committed raw regression logs. The candidate report should not
retain a false validation claim. No source-code counterexample was found in
the exercised padding contract.

The original deployment is not fully reproduced or fully verified. This host
has one AMD Instinct MI350X/gfx950 with ROCm 7.2, not eight NVIDIA H20 GPUs; it
lacks the reported GLM-5.3-W4AFP8 weights, CUDA FlashMLA, Mooncake/InfiniBand,
and the TP=8/DP=2 distributed allocation. The tests establish the shared
metadata-shape correction, not full model semantics, transport behavior, or
the actual CUDA kernel invocation.
