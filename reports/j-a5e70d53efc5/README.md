# Independent review of PR 2419

Upstream issue: https://github.com/sgl-project/sglang/issues/31588

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2455

Candidate: https://github.com/amdpilot-org/sglang/pull/2419 at `1bc5685e2240b64b679772049f1b6f401fcebe04`

Recommendation: **accept**, with the original 8x B300 integration sequence explicitly unverified.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduced the sizing defect with the candidate's allocator-accurate regression: target plus eight draft pools required 72,456,601,600 bytes from a 70,276,402,380-byte budget. The exact candidate passed all 44 pool-configurator tests and 7 subtests.

An independent exhaustive sweep checked 26,415 satisfiable combinations around page boundaries. It varied page size, hybrid ratio, target full/SWA layer counts, full-capacity and ordinary draft-SWA layer mixes, fixed SWA caps, and MXFP8-like data-plus-scale byte geometry. For every case, the returned page-aligned capacity fit the modeled allocator geometry and the next page exceeded the budget.

Source imports resolved to `/job/repo/python/sglang/...`. The candidate contains no native-code changes, so a native rebuild was not applicable. The assigned device was one AMD Instinct MI350X under ROCm 7.2. It cannot reproduce or qualify the report's 8x NVIDIA B300, CUDA 13, TP8, Inkling-NVFP4 startup, FP4 kernels, or model behavior. Therefore this review verifies the platform-independent sizing correction but does not claim an executed full original deployment reproduction.

Raw logs and the independent adversarial driver are retained under `raw/`.
