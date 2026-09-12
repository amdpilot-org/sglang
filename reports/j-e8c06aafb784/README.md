# Independent review of BWAP candidate a74bf498

Upstream issue: https://github.com/sgl-project/sglang/issues/35987

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2935

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2888

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2968

Reviewed exact commit: `a74bf498ee66251ac0ba18a9ae2e200c657ae557`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a partial original-issue fix. It fixes both
counterexamples supplied by the earlier independent review, and its focused
suite and an independent ROCm numerical check pass. It does not fully implement
the original scoring/cardinality contract for all valid inputs.

## Reproduction and findings

The prepared base has no `sglang.srt.bwap` package; importing
`sglang.srt.bwap.bwap_manager` fails with `ModuleNotFoundError`. This records the
original feature absence rather than pretending the base has a narrower unit
failure.

The exact earlier candidate
`ee5f9f9b83040ce90406872a74557bc439dfab67` has the same implementation tree as
this PR immediately before its correction commit. Against that tree I reproduced
both supplied failures:

- For `[[0.8,0,0.6], [0.8,0,0.6], [0.8,0,0.6], [0,1,0]]`, an independent
  Equation 2 calculation is approximately `[0.6928203, 0.5, 0.5196152]` and
  selects neuron 0, while the old implementation returned per-token maxima
  `[0.8, 1.0, 0.6]` and selected neuron 1.
- For `D_FF=3`, sparsity `0.5`, the old implementation retained 2 neurons; the
  mathematical floor contract retains 1.

At exact candidate `a74bf498...`, both cases pass. The focused suite reports 21
tests plus 3 subtests passing.

Independent adversarial review found two remaining counterexamples:

1. Decimal top-k underflow: with `D_FF=5`, sparsity `0.8`, Python evaluates
   `(1 - 0.8) * 5` as `0.9999999999999998`; the candidate applies `math.floor`
   directly and retains 0 neurons. The specified mathematical formula retains
   1. This also permits a zero-width fused path for a valid CLI sparsity.
2. Prompt batching does not apply Equation 2 per request followed by Equation 3.
   For request 1 containing four rows `[0.8,0,0.6]` and request 2 containing one
   row `[0,1,0]`, per-request Equation 2 then element-wise maximum gives
   `[0.8,1.0,0.6]` (neuron 1). `compute_prompt_scores` pools all five rows as one
   sample and gives approximately `[0.715542,0.447214,0.536656]` (neuron 0).
   The source itself calls this a "Phase-1 simplification", but the original
   proposal describes per-sample Equation 2 followed by max aggregation.

These are selection/cardinality errors in the requested pruning method, not
test-only hardening or unrelated serving-smoke concerns.

## Source, native, GPU, and architecture evidence

Imports at the candidate resolved to the checkout:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/bwap/bwap_manager.py`
- `/job/repo/python/sglang/srt/bwap/bwap_fused.py`

The complete base-to-candidate file list contains no C, C++, CUDA, HIP, Cython,
header, or FlyDSL-native source. Therefore no native rebuild was applicable.
The `bwap_fused.py` path is Python/PyTorch and was exercised from the checkout,
not from an installed SGLang wheel.

On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) with Torch
`2.11.0+rocm7.2` and HIP `7.2.26015`, an independent `D_FF=255`, sparsity `0.5`
comparison retained 127 neurons. The gathered reduced-width FFN matched a dense
masked PyTorch reference with maximum absolute error `4.76837158203125e-07`
and mean absolute error `5.939872238513999e-08`.

No production model weights were available. I did not count the candidate's
archived tiny-Llama HTTP smoke as independent proof of semantic accuracy or
throughput. That fixture can validate transport and engine execution only. A
production-model quality/throughput evaluation, NVIDIA, quantized, biased,
TP>1, and speculative-decode paths remain unverified or explicitly unsupported.
The prompt aggregation counterexample is architecture-independent.

Raw revision-specific command output is retained outside the checkout at
`/job/review-evidence-j-e8c06aafb784/`.
