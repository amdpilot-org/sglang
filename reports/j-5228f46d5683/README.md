# BWAP correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/35987

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3067

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2935 at exact
commit `a74bf498ee66251ac0ba18a9ae2e200c657ae557`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3032

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

Both review counterexamples reproduced against the exact candidate before any
source edit. The consolidated branch preserves the candidate's feature and
earlier corrections, then:

- computes `floor((1-sparsity)*D_FF)` from the decimal CLI value so valid
  integer boundaries do not underflow in binary floating point;
- uses that same retained-width helper for ordinary masks and graph buffers;
- passes packed `extend_seq_lens` to BWAP and applies Equation 2 independently
  to each request's prompt rows before Equation 3 updates shared memory;
- leaves prompt memory unchanged, rather than inventing a split, if activation
  rows do not match the supplied packed lengths.

Raw failing-before and passing-after evidence is in `evidence/`.

## Limitations

No production model weights were available, so production semantic accuracy
and throughput were not measured. The prior qualified tiny-Llama fixture only
validates transport and engine execution and was not treated as semantic proof.
NVIDIA, quantized, biased, TP>1, speculative-decode, and unsupported model
architecture paths remain unverified or explicitly unsupported. No native
C/C++/CUDA/HIP/FlyDSL source changed, so a native rebuild was not applicable.
