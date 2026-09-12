# Independent review of amdpilot-org/sglang PR 2000

Upstream issue: https://github.com/sgl-project/sglang/issues/32924

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1977

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2040

Candidate commit: `51ee81b5d924c479141bf17423a74a9e459c8062`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept the candidate as a narrow diagnostic and mitigation feature. It adds
the independently requested same-precision field A/B switch and correctly
prevents import, availability probing, compilation, and launch of the fused
route/pack/MXFP8-quant path when enabled. It does **not** fully resolve the
original issue: the unknown asynchronous CUDA failure was neither reproduced
nor root-caused, and the candidate intentionally makes no such claim.

## Evidence and classification

On the recorded base, the candidate's three tests fail because the dedicated
environment setting does not exist. This is a valid failing-before regression
for the requested kill-switch contract, but it is not a reproduction of the
original production crash.

At the exact candidate commit, the focused candidate suite passed (8 tests),
and the environment plus handoff suite passed (16 tests and 2 subtests). An
independent adversarial script checked all documented true spellings (`1`,
`true`, `TRUE`, `yes`, `y`) while making any import of the fused kernel module
fatal; each spelling returned the fallback sentinel without importing it.
With `0`, the mocked fused path was invoked exactly once. Loaded SGLang source
paths were under `/job/repo/python`, proving the checkout rather than an
unrelated installed SGLang package was tested.

No C++ or other native source changes in the candidate, so a native rebuild
was not applicable. `git diff --check` reports trailing whitespace in three
committed raw pytest log files; executable source and tests were unaffected.

## Architecture limitations

The assigned device is one AMD Instinct MI355X, `gfx950`, with ROCm 7.2. A
deterministic Torch GPU calculation ran successfully and matched its CPU
reference exactly. The original report requires CUDA/PDL on 8 NVIDIA B300
GPUs, TP8/DCP8, Kimi-K3 MXFP4 weights, CUDA graphs, DSPARK speculative decode,
about 119k cached tokens, and a multi-hour traffic soak. None of those
issue-specific conditions were available. Therefore neither the original
failure nor the fused and standalone CUDA kernels were executed here.

## Preserved evidence

Raw command output was preserved outside the checkout while revisions were
switched, under
`/tmp/amdpilot-repo-j-bc1e19550921/review-evidence/`. The structured summary
in `result.json` records every command, exit status, and measured contract.
