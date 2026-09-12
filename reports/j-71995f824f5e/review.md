# Independent review of PR 3122

Candidate: https://github.com/amdpilot-org/sglang/pull/3122 at `155d433aa15aa7105cc619decaf00a22c86ecee6`

Upstream issue: https://github.com/sgl-project/sglang/issues/1763

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3136

Recommendation: **request changes**. The candidate is useful partial integration work, but it does not fully resolve the original feature request and its native/quality/performance claims remain unverified.

## Findings

1. The exact base `358c163250ad3b1f62939b01ce1314a0a31a0365` has no autoregressive `sage` backend: the CLI rejects the value and the registry lacks it. The candidate is directly based on that commit, so there is no prepared-checkout/base discrepancy.
2. At the exact candidate commit, its registered tests pass. Independent testing also confirms CLI registration, ROCm fail-closed behavior, and correct gathered-prefix causal/GQA adapter semantics against an explicit PyTorch reference on the assigned MI355X.
3. Those numerical checks do not execute SageAttention. The candidate test deliberately replaces `sageattn` with full-precision PyTorch SDPA, and `sageattention` is not installed in the prepared interpreter. Thus they cannot verify native quantization, numerical drift, installation, compiler/ISA compatibility, or performance.
4. The exact dependency revision documented by the candidate (`d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`) has public dispatcher branches for SM80/86/89/90/120 and raises for other architectures. SM100 is recognized by its build script but absent from the dispatcher. The candidate's broad “NVIDIA CUDA only” documentation and admission path do not expose this counterexample before runtime.
5. The original issue is motivated by roughly 2x inference speed with similar accuracy. There is no real model, serving, accuracy, long-context, throughput, latency, or memory result. Dense KV gathering and GQA expansion are specifically likely to alter the performance outcome.

## Classification

This is a **partial fix**, not a full original-issue fix. It is more than test-only hardening because it adds an opt-in backend and wiring, but the core native feature and the motivating accuracy/performance contract are unverified. The candidate's report is candid about most hardware limitations, yet its `candidate_verified` outcome overstates what its substituted tests prove.

## Environment and architecture

- Python 3.12.3 via `/tmp/amdpilot-repo-j-71995f824f5e/venv/bin/python`
- Torch 2.11.0+rocm7.2
- One AMD Instinct MI355X, gfx950
- SGLang imports from `/job/repo/python/sglang`
- Torch imports from the pinned environment under `/opt/venv`
- No `sageattention` module and no NVIDIA CUDA device
- No SGLang native code changed by the candidate; no native rebuild was applicable. The external CUDA extension could not be qualified on ROCm.
