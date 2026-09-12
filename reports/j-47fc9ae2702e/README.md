# Independent review of PR 1413 at 678b659

Recommendation: **reject**. The candidate is useful test-only hardening, but it does not fully resolve or verify the original issue.

The exact candidate changes only tests and prior reports. Its new loader fixture constructs destination buffers through `Mxfp4FlashinferCutlassMoEMethod`, which correctly ties the reported packed weight width 2048 and scale width 128 to `hidden_size=4096`. However, the same seven loader/model-shape tests pass unchanged against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Therefore they do not reproduce the original `Hidden size mismatch`, demonstrate a production fix, or identify which main commit fixed the release behavior.

The candidate also makes the backend-selection test independent of the physical ROCm host by explicitly mocking `is_hip=False`. That fixes a real test isolation failure on this machine, but it is unrelated to the original H800 checkpoint-load failure.

## Evidence

- Candidate focused suite: 8 passed and 3 subtests passed.
- Candidate loader suite copied outside the checkout and run on the recorded base: 7 passed.
- Base backend-selection test: failed because host ROCm state suppressed the expected mocked FlashInfer backend.
- Candidate SM90 suite: skipped because this host is AMD Instinct MI355X gfx950 with ROCm 7.2.
- Import verification: SGLang loaded from `/job/repo/python`; Torch loaded from `/opt/venv/lib/python3.12/site-packages` and reported `2.11.0+rocm7.2`.

Raw logs were preserved outside the checkout under `/job/review-evidence-j-47fc9ae2702e` while revisions were switched.

## Scope limitations

The checkpoint and NVIDIA SM90/H800 hardware were unavailable. No full load, FlashInfer post-load permutation, CUDA kernel, TP=4 process execution, DSPARK, HTTP serving, or semantic validation was possible. No production/native files changed in the candidate, so a native rebuild was not applicable.

Upstream issue: https://github.com/sgl-project/sglang/issues/37342

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1447

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1413
