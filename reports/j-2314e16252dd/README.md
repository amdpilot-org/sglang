# Independent review of DSpark draft LoRA candidate

Reviewed `amdpilot-org/sglang#2710` at exact commit
`ad222b0a04e766e3feb92b67bac9c7d98ce943cb` against the open feature request.

The recorded base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) has no DSpark draft
LoRA module or draft-side manager path. The candidate's focused mocked tests
pass and establish a narrower feature: one pinned LoRA adapter is applied to
every DSpark draft request in a server process.

That narrower feature does not fulfill the original contract of choosing
different draft models/adapters for different tasks. An independent
two-request probe supplied distinct task adapter IDs and observed that the
candidate replaced both with `server-wide-fixed-adapter`. Request-level draft
adapter selection, multiple resident draft adapters, admission/lifecycle
policy, and mixed-adapter batching remain absent.

The available AMD Instinct MI355X (`gfx950`, ROCm 7.2) ran the existing GPU
TARGET_VERIFY LoRA metadata tests successfully. No compatible public DSpark
checkpoint plus stage-2 LoRA adapter was available, so real DSpark numerical
semantics, acceptance rate, and CUDA-graph replay with adapter weights remain
unverified. The candidate changes Python and reports/tests only; no native
source changed and no native rebuild was applicable.

Verdict: **request changes**. This is a partial fixed-adapter implementation,
not a full resolution of the original issue.
