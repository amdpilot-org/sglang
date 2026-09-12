# Correction review for CompressedTensorsW4A16Sparse24

- Upstream issue: https://github.com/sgl-project/sglang/issues/31248
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/2966
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/2846
- Independent review PR: https://github.com/amdpilot-org/sglang/pull/2931
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Reviewed candidate: `b1b2f71e8032aff28d638ef356fb62ee61f249c1`

## Outcome

The candidate is rejected as an implementation of the requested feature. Its
seven focused tests pass, but the review's three counterexamples reproduce at
the exact candidate commit. Most importantly, a per-group `pack-quantized`
W4A16 scheme with matched 2:4 metadata returns dense
`CompressedTensorsWNA16`, silently discarding the sparse execution contract.

This correction preserves the candidate's useful format diagnostics and
current-checkpoint coverage. It separates detection of a declared 2:4 layout
from the old CUTLASS compatibility predicate, then fails closed for every
matched 2:4 layout. The top-level sparse W4A16, per-group packed W4A16, and
unquantized sparse cases now all produce the same explicit unsupported error;
the error distinguishes quantized from unquantized weights.

This is a contract-safety correction, not implementation of
`CompressedTensorsW4A16Sparse24`. No maintained sparse scheme, weight loader,
or kernel exists in this source tree. Adding a speculative source path without
an NVIDIA target on which to build and independently validate it would not be
justified.

## Reproduction

All commands used the prepared interpreter and checkout source:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-9ad427772b78/venv/bin/python reports/j-9ad427772b78/evidence/reproduce_sparse24_selection.py
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-9ad427772b78/venv/bin/python -m pytest -q test/registered/unit/layers/quantization/test_compressed_tensors_mixed_precision.py
```

At the candidate, the reproducer emitted one format `ImportError`, one dense
`CompressedTensorsWNA16` selection, and one unsupported-sparsity `ImportError`.
After correction it emits explicit unsupported-sparsity `ImportError`s for all
three cases. The focused suite changed from `7 passed` to `9 passed`.

Raw before/after logs, the exact reproducer, and environment inventory are in
`evidence/`.

## Limitations

The assigned accelerator is AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not
an NVIDIA architecture suitable for validating the requested sparse path. No
GPU numerical execution, NVIDIA compiler/ISA evidence, or native rebuild is
claimed. The reported 26B Gemma weights were not available, and the tiny Llama
fixture cannot qualify Gemma 4, NVFP4/W4A16, 2:4 sparsity, NEXTN, or semantic
accuracy. Full model loading and HTTP serving therefore remain unverified.
