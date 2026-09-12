# Independent review of PR 2826

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2826 at exact
commit `45edfe6adc673223d69098db3524f1052de7ef15`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38516

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2860

The prepared checkout matched the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, an independent
simulated-sm_75 sentinel showed that all three reported families remained
eligible and attempted/proceeded toward JIT: standalone QKNorm returned true
and called `_jit_qknorm_module`, fused QKNorm+RoPE returned true and called its
loader, and residual-gate returned eligible. The check exited 1 as expected.

At the exact candidate commit, the identical check returned false for all
three paths with no loader calls and exited 0. Source tracing also confirmed
that multimodal diffusion `apply_qk_norm` calls
`can_use_fused_inplace_qknorm`, so the standalone gate covers the concrete
counterexample from the prior independent review rather than an unrelated
smoke test. Candidate tests passed 17/17, including sm_75 rejection, sm_80+
acceptance, and the ROCm exemption.

An independent GPU check on the assigned AMD Instinct MI355X (gfx950 family),
Torch 2.11.0+rocm7.2, confirmed that the ROCm exemption remains enabled and
that the residual-gate HIP JIT executes. Its fp32 result matched an independent
CPU-fp64 expression with max absolute error 0.0.

The candidate changes only Python dispatch and tests; it does not change C++,
CUDA, HIP, FlyDSL, or another native source, so no native rebuild was required.
Import inspection confirmed the three reviewed modules loaded from
`/job/repo/python/sglang/...`; `sgl_kernel` loaded from the prepared installed
0.4.6.post1 package.

Recommendation: accept. The source-level architecture gates fully cover the
three JIT families reported by the issue and the previously omitted standalone
QKNorm path. No remaining source counterexample was found within the original
sm<80 skip contract.

Limitations: this host has no NVIDIA sm_75 GPU and uses ROCm rather than pip
CUDA 13, so the exact glibc `rsqrt`/`rsqrtf` compiler failure and the reported
compile-time latency elimination were not directly rerun. FLUX.2-klein-4B
weights were unavailable, so full denoise first-step latency and end-to-end
model behavior were not measured. These limit performance reproduction, not
the deterministic proof that the candidate returns before each affected JIT
loader on simulated sm_75.
