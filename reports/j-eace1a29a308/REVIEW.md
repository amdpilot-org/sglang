# Independent review of PR 1167

Reviewed `https://github.com/amdpilot-org/sglang/pull/1167` at exact commit `2d3ad6945a885d12487e53d696ca13bd9fe7d868` against upstream issue `https://github.com/sgl-project/sglang/issues/38156`.

Recommendation: **accept**. The candidate fully resolves the original host-accounting contract within its documented coordination domain. It replaces the timing-sensitive equal-share guard at every host-pool allocation site with a host-local advisory lock covering both a fresh `psutil.virtual_memory().available - 10 GiB` check and the actual allocation. This avoids double charging, permits unequal in-job requests that fit in aggregate, and coordinates separately launched cooperating SGLang jobs that share the same temporary directory.

Evidence:

- Recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`: mocked 100 then 60 GiB readings with four ranks produced 22.5 then 12.5 GiB equal-share budgets, rejecting a 20 GiB late rank although four 20 GiB pools fit the initial 90 GiB usable memory.
- Exact candidate: real spawned-process fixture admitted 30+20+20+20 GiB against 90 GiB usable, and admitted only one of two concurrent 60 GiB jobs.
- Candidate focused regression and adjacent suite: 35 passed, 12 subtests passed.
- Imports resolved to `/job/repo/python/sglang/...`; Torch was 2.11.0+rocm7.2. No native source changed, so native rebuild was not applicable. The existing AITer native module loaded from the prepared private cache.

Limitations: only one AMD Instinct MI355X (`gfx950`) was available. These are CPU/OS-lock accounting tests; no GPU computation was required or claimed. The exact 8xH200 GLM-5.2-W4AFP8 serving workload, weights, multi-node topology, and heterogeneous pipeline serving run were unavailable. Cross-job locking requires cooperating processes to resolve the same temporary-directory lock path; isolated container `/tmp` mounts and non-SGLang allocators remain outside its domain.

Raw evidence is retained outside the revision-switched checkout at `/job/review-evidence-j-eace1a29a308/`.
