# Independent review of PR 2561

Candidate: https://github.com/amdpilot-org/sglang/pull/2561 at exact commit `a2cc4cf09ea789b28e8c3db0c80d3b71d6e9f0f2`

Upstream issue: https://github.com/sgl-project/sglang/issues/31719

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2521

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2565

Parent candidate: https://github.com/amdpilot-org/sglang/pull/2401 at exact commit `8926f3ebb5219cba35b8fcd68916bde4bb8fe74c`

Parent independent review: https://github.com/amdpilot-org/sglang/pull/2487

## Recommendation

Accept. The candidate is a full source-level correction for the reported conv-cache dtype contract and for both concrete counterexamples identified in the parent review. It is not merely test hardening: the production GDN backend now casts tracked state at the cache boundary, gives the Triton prefill kernel an activation-typed temporary cache, scatters back in the configured cache dtype, preserves `PAD_SLOT_ID=-1`, and routes the MIS path through the same helper.

No remaining source-level counterexample was found in the tested contract. Exact full-model behavior on the reporter's NVIDIA environment remains unverified because the model weights and architecture were unavailable.

## Failing before

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; it did not differ from the requested failing-before revision. On one AMD Instinct MI355X (`gfx950`) with Torch `2.11.0+rocm7.2` / HIP `7.2.26015`, the independent fixture reproduced:

- FP16 tracked activations assigned into a BF16 conv cache: `RuntimeError: Index put requires the source and destination dtypes match`.
- BF16 tracked activations assigned into an FP32 conv cache: the inverse dtype mismatch.
- BF16 activations with FP32 and FP16 conv caches: real Triton compilation failed with `Mismatched type for new_conv_state`.
- `GDNAttnBackend._forward_mis_segments` called `causal_conv1d_fn` directly rather than the dtype-safe helper (which did not exist on the base).

Raw output was preserved outside the revision-switching checkout at `/job/review-evidence-j-45e322cf6cc7/base-independent-boundary.txt`.

## Candidate verification

After temporarily checking out exact commit `a2cc4cf09ea789b28e8c3db0c80d3b71d6e9f0f2`, imports resolved to `/job/repo/python/sglang/srt/layers/attention/linear/gdn_backend.py` and `/job/repo/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py`. The candidate changes Python and reports/tests only; it changes no C++, HIP, CUDA, FlyDSL, or other native source, so no native rebuild was applicable.

The same independent fixture passed both direct store directions, both prior Triton compiler counterexamples, grouped-convolution output references, final-state references, padded-row preservation, and confirmed MIS helper routing. The candidate's four dtype combinations and corrected-boundary regression also passed.

An additional independent adversarial fixture used FP16 activations with a BF16 cache, noncontiguous real slots, an empty `PAD_SLOT_ID=-1` segment, and both initial-state branches. Outputs and active cache updates matched independent grouped-convolution references; the padded alias target and unrelated rows were unchanged.

Focused repository tests passed: 19 tests plus 27 subtests in the GDN policy suite, and the existing cloned-query-state causal-convolution regression.

## Limitations

The reported `QuantTrio/Qwen3.6-27B-AWQ` weights, `qwen3_5` full serving path, NVIDIA RTX 5090, SM120, CUDA 13.3, and multi-node execution were unavailable. The assigned hardware was a single AMD Instinct MI355X/gfx950. Therefore this review verifies the production Python/Triton boundary and numerical cache contract on ROCm, but does not claim full-model startup, semantic accuracy, or direct NVIDIA compiler execution. The tiny Llama serving fixture cannot qualify hybrid-GDN and was not substituted.
