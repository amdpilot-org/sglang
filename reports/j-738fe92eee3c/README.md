# Independent review of PR 1285

Reviewed https://github.com/amdpilot-org/sglang/pull/1285 at exact commit `e190107d29fdc776b6b0c9466b4561bc6d579ac9` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/37022
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1323

Recommendation: **request changes**. The candidate is a meaningful partial control-plane fix: it enables recovery probing by default and prevents a probe for failure count N from clearing a concurrently recorded failure count N+1. Its focused suite passes. It does not fully resolve the original issue because the recurring transfer failure in the reported 5P3D GLM5.2 NVIDIA/XCCL RoCEv2 deployment remains undiagnosed and unreproduced.

Independent adversarial testing also found an ABA race. Registration clears `failed_sessions` and deletes `session_failures[session_id]`. If the same session then fails again while an older probe is in flight, its counter restarts at the old value (normally 1). The old successful probe sees the same session ID and count and incorrectly clears the new failure generation. The exact candidate produced an empty blacklist and assertion failure in `raw/candidate_adversarial_aba.log`.

## Evidence

- `raw/base_candidate_regressions.log`: candidate regressions applied as tests only to recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; 2 failed, 10 passed, 3 subtests passed. This reproduces disabled-by-default probing and the N/N+1 stale-probe race.
- `raw/candidate_regressions.log`: exact candidate; 12 passed, 3 subtests passed.
- `raw/candidate_adversarial_aba.log`: exact candidate; fails because an older successful probe clears a newly failed re-registered session whose counter restarted at 1.
- `raw/candidate_import_paths.log`: SGLang and the reviewed Python modules import from `/job/repo`; Mooncake imports from the installed site-packages. `MooncakeTransferEngine.send_probe` exists.
- `raw/hardware.log`: the prepared host exposes AMD `gfx950`; `nvidia-smi` is unavailable.

No native source changed in the candidate, so no native rebuild was applicable. No GPU execution was used as substitute evidence: the issue's NVIDIA/XCCL multi-node network, GLM5.2 weights, and 5P3D deployment were unavailable.
