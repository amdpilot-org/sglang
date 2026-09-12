# Independent review of DFlash2 dynamic verification correction

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/3051 at exact commit `8a63b907bc515192023000460e5c1fad23411535`.

Upstream issue: https://github.com/sgl-project/sglang/issues/36127

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3083

The candidate's immediate parent (`a6054529682765cc1846bc6239d283363162f3ad`) reproduced the reported overflow defect: a valid positive JSON STS temperature of `1e308` converted to float32 infinity and changed confidence for logit 1.0 to 0.5. The pinned candidate rejects that value and leaves the existing head state unchanged. Its focused regression, broader confidence/scheduler tests, and shared ROCm kernel numerical-reference tests pass.

Recommendation: **request changes**. The correction is valid but incomplete for the same runtime calibration contract. A finite positive JSON temperature of `1e-50` passes the shared validator, converts to float32 zero, is accepted by the new `isfinite` check, and is installed into the confidence head. This violates the validator's strictly-positive temperature invariant and causes division by zero during STS calibration. The post-conversion validation must require both finiteness and strict positivity.

The accumulated candidate also cannot be called a full original-issue verification. No real compatible DFlash2 checkpoint with a trained survival head or compatible DFlash2 target/draft pair was available. Checkpoint naming/schema and end-to-end lossless semantics across sampling, grammar, continuous batching, overlap relay, and live compact graph replay remain unverified. Mixed-quality concurrent-serving throughput remains unmeasured. The tiny Llama fixture was not used because it cannot qualify DFlash2 architecture or semantics.

No native source changed between the recorded base and candidate, so a native rebuild was not applicable. Imports were confirmed from `/job/repo/python/sglang`, with Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, on an AMD Instinct MI350X (`gfx950`).
