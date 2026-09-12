# Independent review of candidate cc72c0ec94a5617dde397bced1b939e284b473c8

Upstream issue: https://github.com/sgl-project/sglang/issues/32942

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1967

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2035

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1997

## Recommendation

Accept. The candidate fully resolves the original issue's kernel-level contract on the available AMD gfx950 architecture.

At the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the unified deterministic extend kernel used the reported reciprocal scaling and failed the candidate's exact numerical reference. At exact candidate commit `cc72c0ec94a5617dde397bced1b939e284b473c8`, the implementation uses the same logarithmic scaling and strict `position > threshold` boundary as regular extend and both decode kernels. The regression passes, as do independent fixed-seed boundary cases.

This is a full fix for the original formula mismatch, not merely test hardening. The candidate changes the affected Triton kernel and adds relevant coverage.

## Evidence

- Prepared checkout initially matched the recorded base exactly. The review returned to `amdpilot/j-1d0a24d40407` before adding this report.
- Base failure on one assigned AMD Instinct MI355X (`gfx950`), using the candidate's exact regression: per-position maximum differences for positions `[3, 4, 5, 6]` at threshold 4 were `[0.0, 0.37117624282836914, 0.3268224895000458, 0.9590050578117371]` (failed).
- Candidate regression at the exact candidate commit: passed on the same GPU.
- Candidate fixed-seed unified-versus-regular case with `xai_temperature_len=4`: passed.
- Independent adversarial reference: thresholds 2, 3, 4, and 8, head dimensions 32 and 64, and positions immediately below, exactly at, and above each threshold all passed. Maximum discrepancies were at most `5.364418029785156e-07`.
- Imported source was `/job/repo/python/sglang/kernels/ops/attention/extend_attention.py`; Triton was `/opt/venv/lib/python3.12/site-packages/triton/__init__.py`, version 3.7.0. Candidate kernels were compiled with a private fresh Triton cache under `/tmp/amdpilot-repo-j-1d0a24d40407/triton-review-candidate`.
- Raw review logs and the independent test script are preserved outside the revision-switching checkout at `/job/review-evidence-j-1d0a24d40407/`.

## Non-blocking observations and limitations

- The complete candidate parity matrix had one pre-existing/unrelated stochastic gfx950 failure in the temperature-disabled `D=80` subcase (`max diff 0.1669921875`). The issue-specific xAI subcase still ran and passed, and the independent references avoid this loose cross-kernel tolerance.
- `git diff --check` over the entire candidate commit reports trailing whitespace in committed raw log artifacts. It does not report a source or test-code defect and does not affect the original fix.
- The environment is ROCm 7.2 with PyTorch 2.11.0+rocm7.2 on AMD gfx950, not the issue reporter's CUDA system. CUDA execution remains unverified here.
- No Grok weights or end-to-end serving/model-semantic run were available. The review validates the actual attention GPU kernel contract, not full-model accuracy.
- No native library rebuild was required: the candidate changes Python Triton JIT source and Python tests, with no C++/native source changes. A fresh private Triton cache ensured the candidate kernel was compiled from the checked-out source.
