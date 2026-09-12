# Independent review of PR 748

Reviewed exact candidate commit `8eb340a9601aa6951737cbe31c5cb0c2a64074c8` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original zero-token FlashInfer MXFP4 issue.

Recommendation: **accept**. The candidate is a source-level full fix of the reported wrapper contract. It is not test-only hardening: the base calls the external boundary for an empty activation, while the candidate allocates the normal output and returns before that boundary only after routing has been materialized and validated. The previous malformed-width counterexamples are rejected, and independent malformed-rank and padding cases also pass. No remaining source-level counterexample was found.

The evidence is bounded. Execution used one AMD Instinct MI355X with PyTorch ROCm 7.2. The unavailable FlashInfer CUTLASS call was instrumented, so NVIDIA H20/SM90 kernel execution and four-rank DeepSeek-V4 DP-attention/TBO serving remain unverified. No native code changed and no native rebuild was applicable.

Raw outputs, the reviewed diff, and the independent test are under `raw/`.
