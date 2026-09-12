# Independent review of PR 1934

Reviewed exact commit `29464b24d07e3365cf8c15f3f7138d0b6053d9cd` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**. The candidate fully resolves the original issue within the tested decode API contract.

## Evidence

- The prepared checkout initially matched the recorded base exactly. The original CPU reproducer failed there: `(24, 16)` was returned instead of the viewer-normalized `(16, 24)`, and pixels differed.
- An independent base matrix failed orientations 2-8, direct-PIL orientation 6, and simulated CUDA routing for orientations 2, 6, and 8.
- The interpreter imported `/job/repo/python/sglang/srt/utils/common.py` at both revisions. At the detached candidate revision it exposed the candidate helper, confirming that tests exercised checkout source rather than another wheel.
- On the exact candidate, an independent matrix matched `PIL.ImageOps.exif_transpose` pixel-for-pixel for orientations 1-8. Direct PIL orientation 6 returned the normalized `(16, 24)` size. Rotated JPEGs bypassed both standard and fancy mocked GPU decoder routes. Missing, orientation 1, and invalid values 0, 9, and 99 retained the fast decoder route.
- Independent exception cases confirmed fail-open behavior when `getexif()` or the tag lookup raises. An unrotated two-frame GIF retained its identity and frame count.
- The candidate's focused regression file passed: 17 tests.

## Scope and limitations

The assigned device was an AMD Instinct MI350X (gfx950), Torch `2.11.0+rocm7.2`, ROCm `7.2.26015`. The affected accelerated paths are NVIDIA nvJPEG/nvImageCodec paths and `sglang.srt.utils.common.is_cuda()` returned false, so real NVIDIA GPU decoding was unavailable. Decoder selection was tested with deterministic mocks; this review does not claim NVIDIA kernel execution.

No server/model run was needed for the decode-level contract, and a tiny text-only Llama fixture would not establish VLM semantic correctness. The candidate changed Python and tests only; native rebuilding was not applicable. No remaining counterexample was found.
