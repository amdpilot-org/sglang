# Independent review of DFlash2 dynamic verification

Reviewed PR: https://github.com/amdpilot-org/sglang/pull/2861 at exact commit `cdd6ac7f5940e15c97cafbee61c0baf3ce4a9603`.

Upstream issue: https://github.com/sgl-project/sglang/issues/36127

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2894

Recommendation: **request changes**. The candidate is a substantive partial implementation, not test-only hardening: it adds an opt-in trained confidence-head path, checkpoint parameter enforcement, STS loading/application, and retains dynamic SPS scheduling and ragged verification. The candidate's focused suite passed (`75 passed, 18 subtests passed`), and shared scheduling/verification kernels passed 22 numerical-reference subtests on one AMD Instinct MI355X.

It does not fully resolve the original issue. No actual DFlash2 checkpoint validates the assumed DSpark-compatible head schema or its loading/inference path, no real DFlash2 pair validates lossless serving semantics, and no mixed-workload throughput measurement exists. The tiny Llama fixture cannot qualify those claims.

Independent adversarial testing also found a concrete STS validation defect. A calibration containing the finite JSON number `1e308` passes the positive-value validator, overflows to `inf` when converted to float32 in `DFlashWorkerV2._load_confidence_sts_calibration`, and silently maps any finite logit at that position to confidence `0.5`. Runtime calibration should reject non-finite values after conversion.

The recorded base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, the candidate regression file failed during collection because `sglang.srt.speculative.dflash_confidence` did not exist, confirming the original missing feature. Candidate imports resolved to `/job/repo/python`, not an installed wheel. No native C/C++/CUDA/HIP/FlyDSL files changed, so no native rebuild was applicable.
