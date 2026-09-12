# Independent review of PR 2662 at 86a6658c

Recommendation: **request changes**. The candidate is a useful partial fix, but it does not fully resolve the original RouterGate consolidation issue.

The original failure was reproduced on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` using the actual Bailing MoE, Bailing MoE Linear, and LLaDA2 gate implementations on an MI350X. Each returned bf16 and differed from an independent fp32 reference. At candidate commit `86a6658c7c3de43a5de96b8726423da0994df53b`, all three returned fp32 and exactly matched the reference.

The candidate's focused test passes, and the fallback bf16-input/fp32-output GEMM path behaves as intended. The enabled ROCm/AITER tier does not: it invokes `tgemm.mm(..., otype=x.dtype).float()`. With bf16 inputs, measured results at M=1, 8, 16, 17, and 64 were bit-identical to the independent fp32 reference rounded through bf16, demonstrating a bf16 output store followed by an upcast. This is directly contrary to the issue's central precision contract.

The consolidation is also incomplete. Hunyuan-v3, MiniMax-M3, Sarvam-MoE, Step3.5, and Nemotron-H retain separate model-local routing paths. Some already produce fp32 logits, but they do not receive the shared deterministic or small-token dispatch policy, and several retain the expensive activation-upcast behavior called out by the issue. The new neutral GEMM module is only a re-export; the implementation remains under the DSv4 attention namespace.

Architecture limits: testing used one AMD Instinct MI350X with ROCm 7.2. CUDA-only tiny GEMM, DeepGEMM, HPC-Ops, AMX, NPU, XPU, and distributed paths were not available. No model weights were available, so checkpoint loading, full-model accuracy, serving, and performance claims remain unverified. No native sources changed, so a native rebuild was not applicable.

Raw command output and the reviewed diff are retained in `raw/`. Structured claims and exact commands are in `result.json`.
