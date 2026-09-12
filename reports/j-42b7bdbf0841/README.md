# Independent review of PR 2877

Candidate: https://github.com/amdpilot-org/sglang/pull/2877 at exact commit `d8788d51d7c567fea6c0dd8ccbf64e40f8dd5f03`

Upstream issue: https://github.com/sgl-project/sglang/issues/34899

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2912

## Finding

Request changes. The candidate adds the requested Python and Rust unified-memory test classes, but it does not enforce the issue's defining contract that **any nonzero KL must fail**. All six new cases call the existing averaged-KL helper with `KL_DIV_THRESHOLD = 1e-9`. An independent adversarial call supplied unequal logprobs and produced nonzero `avg_kl_div=5.0000069648240756e-11`; the candidate oracle accepted it with exit code 0.

The base correctly lacks the new class (targeted collection exits 4), while the exact candidate collects 24 tests, including its six additions. Collection is not execution evidence. This AMD Instinct MI355X (`gfx950`) / ROCm 7.2 environment cannot qualify Inkling's NVIDIA CUTE/FA4 execution path, and no Inkling weights were available. Thus neither backend completed an end-to-end request, no exact-zero KL was measured, and no FULL/SWA/MAMBA restoration fault was shown red then green.

## Source and native paths

- `sglang`: `/job/repo/python/sglang/__init__.py`
- `sglang.srt.models.inkling`: `/job/repo/python/sglang/srt/models/inkling.py`
- `sglang.test.kl_test_utils`: `/job/repo/python/sglang/test/kl_test_utils.py`
- Torch: `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`, version `2.11.0+rocm7.2`
- No C++, CUDA, HIP, or Rust files differ between the recorded base and candidate, and `repository-environment.json` supplies no native rebuild target. A native rebuild was therefore not applicable.

Raw command output was preserved outside the revision checkout in `/job/review-evidence-j-42b7bdbf0841/` while revisions were switched.

## Classification

This is partial, test-only hardening, not a full original-issue fix. The added scenarios are structurally relevant, but the bit-exact oracle is permissive and the intended Inkling/unified-memory execution remains unverified on the assigned architecture.
