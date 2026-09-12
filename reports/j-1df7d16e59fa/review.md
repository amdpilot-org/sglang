# Independent review of PR 718

Reviewed exact commit `f8853833a75e992e618a9e5d74efe493c3bfefe2` against prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The change is a partial fix, not a verified full resolution.

The prepared base reproduced the concrete defect: when fully idle, a generation health probe entered `_request_dispatcher`; in the regression model this changed waiting-queue length, attempted prefill size, PrefillDelayer outcome accounting, local prefill progress, and the ordering of the following user admission. On the exact candidate, all 28 focused controller/scheduler tests passed and the probe no longer enters normal admission.

The correction introduces a separate contract regression. `/health_generate` is documented as checking health by generating one token, and the HTTP layer still constructs that generation request. At the candidate, an idle scheduler intercepts it before `_request_dispatcher` and immediately returns `HealthCheckOutput`. Consequently an idle server can answer 200 without a model forward, so the endpoint no longer checks the model/GPU execution path precisely when no other request can provide that evidence.

The reported performance outcome remains unverified. The prepared host has one AMD Instinct MI355X (`gfx950`, ROCm 7.2), not 8 NVIDIA B30Z GPUs, and DeepSeek-V4-Pro weights were unavailable. No claim is made about TP8/DP8 throughput, 16,384-token rank divergence, `mixed/delay` persistence, or running-request collapse.

No native source changed and no native rebuild was applicable. Python imports on the candidate resolved to the checked-out sources under `/job/repo/python/sglang/srt/managers/`.
