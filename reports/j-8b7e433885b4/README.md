# Independent review of PR 3502

Reviewed `amdpilot-org/sglang` PR 3502 at exact commit
`e681fb82bac9961a901c84b233ed494ee28efca1` against upstream issue 31953.
The candidate is a single commit directly on recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully implements the requested
runtime `schedule_policy` switching contract for the tested single-rank
configuration. It is a functional fix, not test-only hardening.

## Evidence

- `evidence/base-reproduction.log`: on the exact recorded base, msgpack rejects
  the string `schedule_policy` value and the scheduler allowlist lacks the key.
- `evidence/candidate-tests.log`: candidate serialization and scheduler policy
  suites pass, as do `py_compile` and `git diff --check`.
- `evidence/independent-adversarial.log`: independent enum, wire, ordering,
  identity-preservation, mixed-update atomicity, and invalid-input checks pass.
- `evidence/http-summary.json`: independently generated real HTTP/GPU result.
- `evidence/http-run-metadata.json`: server command, readiness, probe result,
  and owned-process cleanup.
- `evidence/environment.txt`: source imports and assigned GPU architecture.

No native source changed, so no native rebuild was applicable. SGLang imported
from `/job/repo/python/sglang`, Torch imported from the pinned ROCm environment,
and the GPU run used one AMD Instinct MI350X (`gfx950`). The deterministic tiny
Llama fixture came from the qualified PR 649 recipe at exact commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`; weights were generated under
`/tmp/amdpilot-repo-j-8b7e433885b4/models/`, outside the checkout.

The run does not reproduce or qualify GLM-5.2-NVFP4, B300, TP8, 115K-token
inputs, 96% production cache-hit traffic, semantic accuracy, or the reported
latency/starvation distribution. Those production-load claims remain outside
the available AMD single-GPU environment.
