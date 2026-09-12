# Independent review of candidate PR 2204

Upstream issue: https://github.com/sgl-project/sglang/issues/31929

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2169

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2243

Candidate: https://github.com/amdpilot-org/sglang/pull/2204 at exact commit `71c775c08c3254f6a253ba818283c6f2e49dc3b5`.

## Recommendation

Accept. The candidate is a full code-level fix for the original unsynchronized producer/consumer contract, not merely test hardening. It replaces the global-memory reload of `expert_start_loc[cur_expert]` with a reduction over the already-loaded token counts. That removes the communication that required a CTA barrier while preserving the prefix output for later kernels.

No remaining counterexample was found within the kernel's documented padded-segment preconditions. The exact NVIDIA H20 runtime symptom and the downstream multi-node DeepEP timeout remain unverified because this review had one AMD MI355X (`gfx950:sramecc+:xnack-`), not the reporter's NVIDIA H20, and no full model/DeepEP deployment.

## Failing-before evidence

The recorded base was `358c163250ad3b1f62939b01ce1314a0a31a0365`, identical to the prepared checkout. The candidate's test file was preserved outside the checkout before revision switching and run against that base. All 5 tests failed: the AST regression found the unsafe `expert_start_loc` reload, and every numerical case reached its compiled-TTIR assertion showing `tt.load %cur_expert_start` remained. The numerical tensors themselves happened to be correct on gfx950.

A separate 1,000-launch guarded stress probe on the base observed no numerical corruption on gfx950, but its compiled TTIR retained `tt.load %cur_expert_start`. This is consistent with the issue's intermittent, architecture/scheduling-sensitive behavior and is not treated as proof that the base was safe.

## Candidate and adversarial evidence

The exact candidate was checked out detached. Python imported SGLang from `/job/repo/python/sglang`, Torch 2.11.0+rocm7.2 from the prepared interpreter, and Triton 3.7.0 from `/opt/venv/lib/python3.12/site-packages/triton`.

The candidate's in-tree regression passed 5/5. An independent guarded reference probe passed 120 randomized launches covering 2, 7, 24, 33, 64, and 129 experts; block sizes 16, 32, 64, and 128; zero experts' counts; varied padded segment lengths; and valid counts from zero through each padded length. It compared both prefix locations and every output index against Torch references and checked leading/trailing sentinels for out-of-bounds writes. Candidate TTIR contained no `tt.load %cur_expert_start`.

The source change is Triton Python only. No C++, FlyDSL, wheel, or other native source changed, so a native rebuild was not applicable. GPU kernels were freshly compiled with a private Triton cache.

## Preserved evidence

Raw commands' output, base/candidate TTIR and gfx950 ISA, API snapshots, and the independent probe are under `reports/j-0675ae56f8a2/raw/`.

