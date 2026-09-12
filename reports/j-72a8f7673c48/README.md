# Independent review of candidate PR 2306

Upstream issue: https://github.com/sgl-project/sglang/issues/31384

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2256

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2346

Candidate: https://github.com/amdpilot-org/sglang/pull/2306 at exact commit
`82024c3e396efc6b4f667f901f0daf25c4349029`.

## Verdict

Recommendation: **accept**, with the scope classified as a verified narrow
metadata-contract fix rather than a full reproduction of the original issue.

The candidate is based directly on the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, its regression test
fails because `cal_padded_tokens` returns raw 11/1 row counts while
`ForwardBatch.prepare_mlp_sync_batch` pads those ranks to 16/8 for attention TP
8. At the exact candidate commit, the same regression and the existing
forward-metadata plan tests pass. The source import was confirmed to resolve to
the checked-out tree, not an installed SGLang copy.

An independent exhaustive fixture compared the helper against the production
padding-order oracle (align each rank, then select MAX_LEN or the SUM_LEN local
rank) across 140 combinations. It covered TP sizes 1, 2, 3, 8, and 16;
single/multiple and zero-token ranks; aligned and unaligned counts; both padding
modes; both DP ranks; and verified that caller counts are not mutated. It also
exercised the real tensor-padding consumer on the assigned gfx950 GPU, including
an idle rank and non-power-of-two TP. No counterexample was found in this
isolated contract.

The candidate changes Python only. No C++/CUDA/HIP/FlyDSL/native source or
native import path changes, so no native rebuild was applicable.

## Classification and limitations

This review does **not** establish a full original-deployment fix. The report
requires two H20 nodes, CUDA FA3, TP16/EP16/DP2, DeepEP, GLM-5.2-FP8 weights,
EAGLE, and concurrent serving. The prepared environment has one AMD Instinct
MI350X (`gfx950`) with ROCm 7.2 and no GLM-5.2 weights. Consequently the exact
FA3 exception, distributed collectives, full serving path, model semantics, and
concurrency were not executed. Those are remaining unverified deployment
counterexamples, not evidence against the narrow source correction.

The image-prepared checkout exactly matched the recorded base. During review,
`origin/main` had advanced to `bd45cd50ca900dd821f829ca9adfbf9aa3336bda`;
all failing-before comparisons and this report branch intentionally use the
campaign-recorded base.

## Evidence

- `raw/base-regression.txt`: candidate regression temporarily applied to the
  recorded base; 3 failures and 2 passes.
- `raw/candidate-regression.txt`: exact candidate; 12 passes plus 2 subtests.
- `raw/candidate-import-environment.txt`: checked-out source paths and measured
  Torch/ROCm/device architecture.
- `raw/adversarial_padding_check.py` and `raw/candidate-adversarial.txt`:
  independent 140-case oracle and real GPU tensor checks.
