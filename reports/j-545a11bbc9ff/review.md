# Independent review of PR 1645

Candidate: `bab33e2e1c26450edc7661eb4de484af3d331740`

Upstream issue: https://github.com/sgl-project/sglang/issues/34259

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1683

## Recommendation

Request changes. The two added tests are valid test-only hardening for mixed-request hidden-state response slicing, but the candidate does not verify or fully resolve the reported Kimi-K3 generated-reasoning leak. Its report overstates the causal link and labels the candidate verified.

## Evidence

The candidate changes only tests and its own report; it changes no runtime source. Its nine-test focused suite passes at the exact candidate commit. Independent adversarial cases also passed on the assigned MI350X/gfx950 GPU for FULL and LAST hidden-state offset accounting, including a non-stored/non-requesting request.

That mechanism is not sufficient evidence for the original contract:

- The original symptom is generated reasoning belonging to another prompt. The cited path slices optional hidden states for responses after `next_token_ids` have already been materialized. It is entered only when the batch requests hidden-state capture. Ordinary generation requests do not use it.
- The official `v0.5.17` tag is commit `29481685462732237d80d86076d6563e1f658102`, dated 2026-08-07. Upstream PR #30177 merged on 2026-08-02, is an ancestor of that tag, and the tag already has the corrected offset accounting. The original Kimi-K3 issue was opened on 2026-08-10 and explicitly reports the symptom on 0.5.17. This contradicts the candidate statement that #30177 landed after the affected release and prevents that change from explaining the report as stated.
- The issue supplies no deterministic reproducer. Kimi-K3 weights, eight NVIDIA B300 GPUs, CUDA 13.0, and the reported distributed serving setup are unavailable here. The prepared environment has one AMD Instinct MI350X (gfx950), ROCm 7.2, and PyTorch 2.11.0+rocm7.2.

The prepared base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`. It already contains the hidden-state offset correction, so the original generated-reasoning failure was not reproduced there. The candidate's focused tests demonstrate only that this existing correction behaves as expected.

## Source and native validation

The measured imports resolved `sglang` and `batch_result_processor` to `/job/repo/python/...`, so the checked-out source was exercised rather than a wheel copy. PyTorch resolved from the prepared environment. The candidate has no C++/CUDA/HIP/native changes, so no native rebuild was applicable.

Raw command output and fetched issue/PR metadata were retained outside the checkout at `/job/review-evidence-j-545a11bbc9ff` while revisions were switched.
