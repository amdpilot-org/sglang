# Investigation evidence

- Source and mirror issues were open and had no comments when inspected on 2026-09-12. Targeted GitHub searches for `AMX router fp32` found no related upstream commit or PR.
- Before the fix, `MoEGate.forward` selected `weight_packed_linear` on AMX and returned its BF16 result. The focused regression failed with `AssertionError: torch.bfloat16 != torch.float32`.
- Native source evidence: `python/sglang/kernels/aot/csrc/cpu/gemm.cpp` creates the result with `at::empty({M, N}, mat1.options())`; its BF16 kernel accumulates through an FP32 temporary and converts the result to the activation scalar type.
- A numerical boundary test shows FP32 logits `[1.001, 1.002]` become equal after a BF16 round trip, proving a cast inside TopK cannot recover the lost ordering.
- After the fix, the focused regression and the adjacent GLM correction-bias suite passed: `12 passed`.
- The assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) ran `linear_bf16_fp32`; it returned FP32 and matched CPU FP32 matmul of the same BF16 operands with maximum absolute error `2.86102294921875e-06`.

Raw command output is retained under `/tmp/amdpilot-repo-j-a22edf96cfe6/evidence/`:

- `failing-before.txt`
- `passing-after-unit.txt`
- `gfx950-router-reference.txt`

The machine has an AMD EPYC 9965 CPU without Intel AMX. No full model weights were available. Therefore no native AMX execution, full-model behavior, semantic model accuracy, serving path, or distributed workload is claimed.
