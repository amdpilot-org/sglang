# DSA kpool logits budget correction

Upstream issue: https://github.com/sgl-project/sglang/issues/37712

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1286

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1153 at exact commit `20e6fa7d408c689ffc3753969396a745d5c462b2`

Independent review: https://github.com/amdpilot-org/sglang/pull/1255

The candidate's request-grouped row chunking fixes the earlier concatenated-K and PAGED mapping defects and is retained. The remaining counterexample was independently reproduced on the exact candidate: with one request containing 10 pooled K rows and a 4-byte logits budget, `fp8_mqa_logits` was still invoked and returned one 40-byte fp32 row.

`fp8_mqa_logits` produces a complete K-wide row and the downstream fused kpool top-k transform consumes that complete row. There is no semantics-preserving K-split interface in this path. The correction therefore detects when the budget cannot hold even one atomic request-local row and raises before calling DeepGEMM, instead of silently exceeding the computed budget and risking a much larger CUDA allocation. Feasible rows retain the candidate's request grouping and row chunking unchanged.

The focused GPU suite covers RAGGED and PAGED mappings, multiple row budgets, the exact-fit boundary, empty K, padding, and the newly rejected 40-byte/4-byte case. It passes 12 tests after the correction.

Execution used one AMD Instinct MI350X (`gfx950`) with PyTorch 2.11.0+rocm7.2 and HIP 7.2.26015. The tests allocate real FP8 and metadata tensors on the GPU, but mock DeepGEMM and fused top-k to inspect orchestration and allocation shape. No GLM-5.3-Flash/RadixArk weights or four-B300 CUDA TP/EP environment was available. Actual NVIDIA kernels, the reported serving workload, and end-to-end semantic accuracy remain unverified. An atomic row that exceeds the budget is now rejected rather than computed; supporting it would require a K-splittable logits/top-k interface with globally correct score merging.
