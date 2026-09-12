# Correction-generation report

Upstream issue: https://github.com/sgl-project/sglang/issues/39072

Mirror issue: https://github.com/amdpilot-org/sglang/issues/900

Candidate PR: https://github.com/amdpilot-org/sglang/pull/800

Independent review PR: https://github.com/amdpilot-org/sglang/pull/863

The candidate was inspected at exact commit
`995ac1056eef6bea1c17c728327c334268ef09e1`. Its functional correction is
preserved: Eagle may plan DSA metadata before `prepare_mlp_sync_batch` aligns
the query batch to attention TP, so `cal_padded_tokens` must apply the same
idempotent alignment. On the recorded base, the regression produced 9 rows
where the eventual query requires 12; with the correction it produces 12 rows.

The review's evidence-integrity counterexample was independently reproduced.
`git diff --check 358c163250ad3b1f62939b01ce1314a0a31a0365
995ac1056eef6bea1c17c728327c334268ef09e1` exits 2 and identifies nine
trailing-whitespace lines in the candidate's committed raw pytest logs, despite
the candidate report claiming exit 0. Those raw logs are cleaned in this
consolidated correction, and the complete final diff passes the check.

The focused regression was rerun both without and with the source correction:
3 failed/1 passed before and 4 passed after. A production
`pad_dsa_cache_seqlens` tensor operation also ran on the assigned AMD Instinct
MI350X/gfx950, padding 9 rows to 12 and zero-filling the new rows.

The original deployment remains unexecuted. This environment has one AMD GPU,
not eight NVIDIA H20s, and lacks GLM-5.3-W4AFP8 weights, CUDA FlashMLA,
Mooncake/InfiniBand, and the TP=8/DP=2 disaggregated allocation. Accordingly,
the evidence validates the shared shape contract only; it does not establish
full model, kernel, transport, synchronization, or warmup behavior.
