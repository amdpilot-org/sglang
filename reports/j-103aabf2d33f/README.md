# Independent review of PR 2677

Reviewed `https://github.com/amdpilot-org/sglang/pull/2677` at exact commit
`ce113fb8d9d5b8a6fac88e0d2364dfcfc7ff2b6b` against upstream issue
`https://github.com/sgl-project/sglang/issues/38695` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/2680`.

## Verdict

Request changes. The candidate is a substantial partial fix, but it does not
fully deliver the issue's one-gate-layer consolidation. It independently fixes
the AITER fp32 output-store defect and migrates Hunyuan-v3, MiniMax-M3,
Sarvam-MoE, Step3.5, and Nemotron-H. It also moves ownership of the shared GEMM
implementation to the neutral `kernels/ops/gemm` location and leaves the DSv4
path as a compatibility re-export.

However, issue-relevant consumers still retain model-local router modules and
therefore do not all inherit `RouterGate`'s output and deterministic dispatch
policy. Concrete examples in the exact candidate are Kimi Linear's
`ReplicatedLinear` gate and the custom gates in Bailing-MoE,
Bailing-MoE-Linear, LLaDA2, and Bailing-MoE-v3. The first three custom gate
families return the dtype selected by their local weight/config path; the
candidate only removes an explicit final downcast. This is valid hardening but
not the promised universal shared policy. DeepSeek-v2 also retains a model-local
gate wrapper, although it does call the newly shared helpers.

## Evidence

- On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an
  assigned AMD Instinct MI350X with ROCm 7.2 reproduced the AITER failure at
  M=1,7,8,15,16,17,64. Every fp32-typed result was exactly bf16-representable
  and exactly equal to the independent CPU fp32 reference rounded through
  bf16; maximum fp32-reference error reached 0.191894531.
- At candidate `ce113fb8d9d5b8a6fac88e0d2364dfcfc7ff2b6b`, the same
  test produced fp32 values that were not bf16-representable, with maximum
  error no larger than 7.62939453e-06.
- The candidate's focused RouterGate suite passed 4 tests. An additional GPU
  adversarial check made both small-token alternatives raise if consulted
  under deterministic inference; neither was reached, and the fallback kept
  an fp32 output-store result.
- All fifteen directly reviewed model modules imported successfully from the
  prepared source checkout.
- The source audit records only nine model files importing `RouterGate`, while
  the candidate's own broad audit and the independent scan show remaining
  model-local routing implementations.

No candidate source was modified. No native source changed, so no native
rebuild was applicable. Full-model checkpoint loading, semantic evaluation,
and throughput were not run because model weights were unavailable. CUDA-only
tiny GEMM, DeepGEMM, HPC-Ops, AMX, NPU, XPU, distributed, and NVIDIA paths were
not executable on the assigned AMD MI350X.
