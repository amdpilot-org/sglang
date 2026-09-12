# Independent review of amdpilot-org/sglang PR 1709

Upstream issue: https://github.com/sgl-project/sglang/issues/34149

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1741

Candidate reviewed exactly at `68815346b5fee9a1dd9b947c8fe93d6f7a5f21ba`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate is a full source-level fix for the
original delayed final-prefill abort race, not merely test hardening. On the
base, the candidate regression tests independently reproduced all three key
failures: an aborted prefill was admitted to optimistic decode, its delayed
token was appended, and sampling-mask overflow replaced the earlier abort.
The exact candidate passed the focused regression suite and existing adjacent
abort/grammar tests.

Two review-only adversarial cases were kept outside the candidate checkout.
They verify that an aborted final-prefill request reaches normal KV/request-slot
cleanup without being cached unfinished, and that an abort on a decode member
of a mixed batch retains the previously established decode partial-output
semantics. Both passed.

The candidate source was imported from `/job/repo/python`, not an installed
SGLang wheel. Torch was `/opt/venv/lib/python3.12/site-packages/torch`, version
`2.11.0+rocm7.2`. The available GPU was an AMD Instinct MI355X with
`gfx950:sramecc+:xnack-`. No C++, FlyDSL, or other native source changed, so a
native rebuild was not applicable. Candidate-retained server evidence shows
real gfx950 prefill execution with the qualified tiny random Llama fixture, but
that smoke did not hit the precise cancellation race and is not used as proof
of the fix.

Limitations: no real-model Qwen run, exact-race HTTP/GPU pause, multi-GPU, PP,
DP, distributed, or multi-node execution was performed. The deterministic
scheduler/result-processing tests provide the race evidence. The complete PR
diff fails `git diff --check` because committed raw pytest logs contain trailing
whitespace; the source and test portion passes. This artifact hygiene issue does
not change the functional recommendation.

Full command outputs used for review were preserved outside revision switches
under `/job/review-evidence-j-bb1e41183003/`.
