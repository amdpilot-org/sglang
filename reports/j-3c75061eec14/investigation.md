# Independent candidate review

Upstream issue: https://github.com/sgl-project/sglang/issues/30609

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2372

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2461

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2426

Exact candidate commit: `b8902dbde7a9d707fc5729883551ed082c110baa`

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

Recommendation: **request changes**. The candidate is credible partial hardening,
but it does not establish a full fix for the original issue.

The base already contains ordered Mooncake index batching from merged upstream PR
32758, but `SGLANG_MOONCAKE_MAX_TRANSFER_BATCH_INDICES` is `0`, so the original
commands (which do not set that variable) retain the unbounded path. The candidate
changes the default to 1024 and adds a regression for the default. Its focused test
and independent boundary/failure cases pass.

The candidate's `fixed` conclusion is stronger than its evidence. Upstream PR 32758
validated 1024-index batching with `nvidia/GLM-5.2-NVFP4` on separate 4x L20D nodes
and an approximately 800K-token workload. Its discussion says the failure could not
be reproduced on FP8. The original report instead uses `glm5.2-fp8`, H100s, DSA,
HiCache, HiSparse, DeepEP, two 2-node prefill instances, and a 4-node/32-rank decode
instance. A later reporter also associates a freeze with `--moe-a2a-backend deepep`.
Changing a global default may mitigate oversized Mooncake calls, but no evidence
connects it conclusively to every failure mode in that original configuration.

There is also a provenance discrepancy: PR 32758's prose says 1024 is the default,
but its merged commit `a6b542813ff46de8ba3856e0236f1af35f4fca54` actually uses
`EnvInt(0)` and describes the cap as opt-in. The candidate should not cite that
merged change as proof that a global 1024 default was already validated or accepted.

## Reproduction and validation

The prepared interpreter was
`/tmp/amdpilot-repo-j-3c75061eec14/venv/bin/python`. Imports resolved to
`/job/repo/python/sglang/srt/environ.py` and
`/job/repo/python/sglang/srt/disaggregation/mooncake/conn.py`.

On the recorded base, the existing batching suite passed (5 tests and 3 subtests),
showing the opt-in implementation works, while the environment default was observed
as `0`. Thus the issue's unchanged launch commands still select the unbounded path.

At the exact candidate commit, the candidate suite passed (6 tests and 3 subtests).
An independent script exercised 1023, 1024, 1025, and 2049 indices. The resulting
transfer byte lengths were respectively `[16368]`, `[16384]`, `[16384, 16]`, and
`[16384, 16384, 16]` for a 16-byte page. Explicit zero preserved one 32784-byte call,
an injected second-call failure returned `-7` and stopped after two calls, and
explicit environment values 0 and 17 were honored. Forty related disaggregation
timeout/cleanup/deferred-release tests also passed.

No native source changed, `repository-environment.json` specifies no native artifact,
and no native rebuild was applicable.

## Environment limits

The assigned hardware is one AMD Instinct MI355X (`gfx950`) with Torch
2.11.0+rocm7.2. GLM-5.2 weights, NVIDIA H100 hardware, Mooncake/RDMA peers, and the
reported multi-node topology were unavailable. No GPU execution or tiny-model server
smoke is claimed as proof, because neither would exercise the model-specific,
multi-node Mooncake/DeepEP contract.

Raw issue/PR snapshots, diffs, imports, and test logs were preserved outside the
checkout at `/job/review-evidence-j-3c75061eec14/` while revisions were switched.
