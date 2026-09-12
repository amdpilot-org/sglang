# Independent review of PR 3354

Upstream issue: https://github.com/sgl-project/sglang/issues/37904

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3358

Candidate: https://github.com/amdpilot-org/sglang/pull/3354 at `8b48178f4a476e73c205c7aee1991e48fcc37be8`

Recommendation: **unverified**. The candidate implements the proposed per-stream workspace invariant and its host-side boundaries pass on the assigned gfx950 GPU, but this environment cannot execute the NVIDIA Marlin/green-context behavior that constitutes the original bug. Accordingly, this review does not claim a full original-issue fix.

The recorded base was exactly `358c163250ad3b1f62939b01ce1314a0a31a0365`. Running the candidate regression against it fails at import because the workspace selector is absent. At the exact candidate commit, four selector/init/sizing tests pass and the actual Marlin deep-queue test skips on ROCm. The existing FP8 logical-order regression also passes.

The candidate changes Python only. Imports resolved to `/job/repo/python/sglang` through the prepared interpreter. There was no native source change and no prepared native build in `repository-environment.json`, so a native rebuild was not applicable.

Raw logs and the audited remaining direct-workspace matches are retained beside this report. The original service reproducer, NVIDIA deep-queue kernel case, 27B model correctness, and serving throughput remain unverified.
