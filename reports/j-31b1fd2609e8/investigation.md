# Consolidated correction investigation

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2689 at exact commit
`811f07d456adfd5744b1cb703c5a67e989a39fff`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2733

The candidate's focused 12-test architecture suite passed and its gates correctly
short-circuit fused QKNorm+RoPE and residual-gate JIT eligibility on simulated
sm_75 while retaining sm_80+, sm_90, sm_121, and ROCm. Those changes are
preserved.

The review's remaining counterexample was independently reproduced against the
exact candidate. `can_use_fused_inplace_qknorm(128, torch.float16)` returned
true and called `_jit_qknorm_module` once because the standalone QKNorm path had
no architecture check. This loader produces the `sgl_kernel_jit_qknorm_*`
module named in the original report and is reachable from diffusion
`apply_qk_norm`.

The correction adds the same pre-Ampere CUDA / ROCm-aware gate before that
loader. The expanded focused suite has 17 passing tests and asserts that sm_75
cannot call the standalone loader, while sm_80, sm_90, sm_121, and ROCm remain
enabled. On the assigned AMD Instinct MI350X/gfx950, the retained HIP exemption
allowed the residual-gate JIT to execute and its fp32 output matched an
independent CPU-fp64 expression exactly for the fixture.

The full registered QKNorm numerical test is environment-blocked: the prepared
ROCm environment's installed AOT `sgl_kernel` extension does not register
`sgl_kernel::rmsnorm`, causing 560 cases to fail in the unchanged AOT reference
before the changed JIT path. This is not treated as evidence about the fix.

No NVIDIA sm_75 device, pip CUDA 13 nvcc/glibc combination, or
FLUX.2-klein-4B weights were available. Consequently the exact `rsqrt` compiler
diagnostic and reported full-denoise first-step latency remain unverified.
