# Independent review of PR 3154

Reviewed `https://github.com/amdpilot-org/sglang/pull/3154` at exact commit
`ea61cd6c46cc63d61262adac4ea33621c06019c5` against upstream issue
`https://github.com/sgl-project/sglang/issues/19090` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/3155`.

Recommendation: **request changes**. The candidate correctly adds the missing
`DiffGenerator` request surface and hardens malformed scheduler responses, but it
does not fully resolve the original feature contract. In particular, there is no
real diffusion checkpoint sleep/wake/generation/refit run, no comparison against
kill-and-relaunch, and FSDP inference remains unsupported.

## Evidence

- Prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`: the candidate regression
  produced 9 failures because the three `DiffGenerator` lifecycle methods were
  absent; the controller-only GPU test passed.
- Exact candidate: 15 focused lifecycle, shutdown, residency, and snapshot tests
  passed.
- Independent adversarial script: `None` and missing-`output` replies became
  lifecycle-specific `RuntimeError`s; the broker-shaped plain dictionary also
  became a `RuntimeError`, though its original message was discarded. FSDP sleep
  raised the documented unsupported-path `RuntimeError`.
- Independent ROCm GPU controller round trip on AMD Instinct MI350X: allocated
  bytes fell from 33,280 to 0 while sleeping, wake restored the module, and output
  matched an independently computed CPU reference with maximum absolute error
  `7.152557373046875e-07`.
- Imports resolved to `/job/repo/python/sglang/...` and pinned Torch
  `2.11.0+rocm7.2`. No native source changed, so no native rebuild applied.
- No diffusion checkpoint was present under the private prepared runtime.

Complete raw command output was preserved outside the revision-switching checkout
at `/job/review-evidence/j-424497a0bebe/`.
