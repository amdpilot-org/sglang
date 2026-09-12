# Independent review of PR 3028

Reviewed `https://github.com/amdpilot-org/sglang/pull/3028` at exact commit
`499abece260bdb1a529fd131a59487f3bdcd214b` against upstream issue
`https://github.com/sgl-project/sglang/issues/31248` and recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Finding

The candidate is a valid partial, fail-closed hardening, not an implementation of
`CompressedTensorsW4A16Sparse24`. On the base, the top-level sparse W4A16 case
raises the original ImportError, the mixed/per-group case incorrectly selects
dense `CompressedTensorsWNA16`, and the unquantized case raises an unsupported
2:4 error. At the exact candidate, all three reject the unsupported sparse
layout consistently. The candidate's focused suite passes 9 tests and 2
subtests.

Independent adversarial cases confirm that `dense` plus `2:4` also fails closed,
while a non-2:4 structure and nonmatching sparsity target retain dense selection.
The downloaded live model `config.json` confirms that the presently published
Gemma checkpoint is mixed FP8/NVFP4 and ignores vision projections; that fact
does not supply the sparse scheme, loader, or kernel requested by the issue.

Recommendation: **accept** this accurately scoped safety correction, while
recording `fully_resolves_original: false`. It must not be described as solving
the original feature request.

## Environment and limitations

The prepared interpreter imported SGLang and compressed-tensors code from
`/job/repo/python`. Torch is `2.11.0+rocm7.2`, HIP is `7.2.26015`, and the
assigned device reports AMD Instinct MI350X / capability `(9, 5)`. No native
files changed, so no native rebuild was applicable. This AMD environment cannot
qualify a prospective NVIDIA sparse kernel or its generated ISA. The 26B model
weights were not available, and no sparse GPU numerical execution, full model
load, NEXTN, serving, or semantic validation was possible. The tiny Llama
fixture cannot establish Gemma 4 or sparse W4A16 correctness and was not used.

Raw commands and outputs are retained in `evidence/`.
