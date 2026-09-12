# Independent review of amdpilot-org/sglang PR 805

Upstream issue: https://github.com/sgl-project/sglang/issues/39070

Mirror issue: https://github.com/amdpilot-org/sglang/issues/847

Candidate: https://github.com/amdpilot-org/sglang/pull/805 at `a0905d0fad4c0d6756bdbfc02ecf66dfd03a579b`

Recommendation: accept.

The candidate is test-only hardening, not a new production fix. The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the broader production behavior from upstream PR #37441: a model's `_supported_attention_backends` set constrains automatic selection, while an explicit request is admitted only after the platform resolves it and the backend satisfies the layer's semantic requirements.

FLUX.2 continues to omit `SAGE_ATTN` from its automatic-selection set. On the exact candidate, the added regression demonstrates that an explicit Sage request bypasses that set, implicit selection still respects it, and unsupported packed-varlen semantics still fail closed. Independent probes also covered a component-specific override and direct per-call explicit selection using the FLUX.2 set.

For failing-before evidence, I temporarily restored the pre-#37441 strict membership condition and ran the candidate's issue-specific regression. It failed with the original whitelist error, including the FLUX.2 backend list. I then restored the exact candidate and verified no diff remained before returning to the prepared review branch.

The candidate modifies only the selector unit test and its own report artifacts. It does not modify `flux_2.py`, selector production code, native code, or dependencies. Therefore no native rebuild was applicable.

Architecture limitation: the assigned GPU is an AMD Instinct MI350X (`gfx950`) with ROCm 7.2. The real ROCm resolver rejects `SAGE_ATTN`, and the prepared interpreter has no `sageattention` installation. No CUDA Blackwell resolver/kernel, full FLUX.2 model construction, serving path, image output, or numerical/image-quality claim was verified. Those limitations do not produce a remaining source-level counterexample to the original explicit-selection contract.

Raw command output was preserved outside the revision-switching checkout under `/job/review-evidence/`.
