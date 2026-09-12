# Independent review of PR 2228

Upstream issue: https://github.com/sgl-project/sglang/issues/32056

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2156

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2261

Candidate: https://github.com/amdpilot-org/sglang/pull/2228 at
`ba54bcf0546d2d1bb2ce24899e0036f29f552181`.

## Verdict

Recommendation: **accept**, as test-only hardening. The candidate does not add
the production correction and therefore is not, by itself, a full fix for the
original issue. The recorded base already contains the relevant correction in
both current standard-layout branches: disposal of `hidden_states_ref` is
guarded by `runner_config.inplace`. The candidate adds focused regression
coverage for that existing behavior.

The exact candidate changes only a Python test and its prior investigation
artifacts. It changes no Python production source, C++, HIP/CUDA, or other
native code. Consequently no native rebuild was required or performed.

## Independent evidence

The prepared checkout initially matched the requested base
`358c163250ad3b1f62939b01ce1314a0a31a0365`; no difference from the recorded
base was found. Candidate and issue metadata plus all command output were
preserved under `/tmp/amdpilot-repo-j-1e1f896927bb/review-evidence` while
revisions were switched. The prepared interpreter imported
`/job/repo/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py`, confirming
that tests exercised the checked-out source rather than another installed
copy.

On the exact candidate, its four-case regression passed on the assigned AMD
Instinct MI350X (`gfx950`) with Torch 2.11.0+rocm7.2. It covers compact and
masked layouts for both borrowed (`inplace=False`) and owned
(`inplace=True`) inputs.

To reproduce the original failure, I temporarily restored unconditional
`dispose_tensor(hidden_states_ref)` in both standard-layout branches and ran
the candidate test. Both `inplace=False` cases failed because the caller's
shape collapsed from `[4, 4]` to `[0]`; both `inplace=True` cases continued to
pass. The temporary mutation was then reverted and is absent from this review
branch.

As an independent adversarial extension, I reused the caller-owned activation
after pre-permute in a BF16 linear followed by sigmoid, modeling the shared
expert gate operation described by the issue. Both compact and masked
`inplace=False` cases matched a preserved reference; all four ownership cases
passed.

## Scope and limitations

The reported Qwen3.5-397B-A17B-FP8 weights, eight H800 GPUs, CUDA 12.x,
TP8/EP8 topology, and serving workload were unavailable. This review therefore
does not claim a full-model, CUDA, or distributed reproduction. Packing
kernels are mocked by the candidate regression, so it validates the Python
ownership contract and GPU tensor reuse, not DeepGEMM numerical correctness.

A CPU-only adversarial invocation could not collect because this ROCm build
queries GPU properties during the SGLang import path and raises when GPUs are
hidden. The candidate is registered only for CUDA/AMD GPU CI, so this does not
invalidate its intended coverage, but CPU portability of the test was not
established.

No counterexample was found within the scoped ownership contract. Full-model
behavior remains unverified for the architecture and topology above.
