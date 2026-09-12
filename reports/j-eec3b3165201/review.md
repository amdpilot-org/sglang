# Independent review of PR 2689

Candidate reviewed exactly at `811f07d456adfd5744b1cb703c5a67e989a39fff`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a useful partial fix: it
short-circuits the fused diffusion QKNorm+RoPE and residual-gate JIT eligibility
paths before a build on CUDA targets below sm_80, while leaving ROCm enabled.
However, it does not fully resolve the original issue because the report also
identifies failed `sgl_kernel_jit_qknorm_*` builds. That module is loaded by
`python/sglang/kernels/ops/layernorm/norm.py`, is used by diffusion's
`multimodal_gen/runtime/layers/layernorm.py::apply_qk_norm`, and remains
ungated. An independent simulated-sm_75 check showed
`can_use_fused_inplace_qknorm(128, torch.float16)` still invokes its JIT loader.

The prepared checkout was exactly the requested base. The candidate's 12-test
regression failed 12/12 on the base and passed 12/12 at the candidate commit.
Source imports at the candidate commit resolved to `/job/repo/python/sglang/...`,
not an installed copy. The candidate changes Python only, so no native rebuild
was applicable. On the assigned AMD Instinct MI350X (`gfx950`, ROCm 7.2), the
HIP exemption remained enabled and the residual-gate JIT executed; fp32 output
matched an independent CPU-fp64 expression exactly for the tested fixture.

The reporter's RTX 2080 Ti/sm_75, pip CUDA 13 nvcc/glibc combination and
FLUX.2-klein-4B weights were unavailable. Therefore the actual `rsqrt` compiler
diagnostic and reported 49-second first-step latency could not be reproduced.
The deterministic tiny Llama fixture is not relevant evidence for this
diffusion-kernel/compiler contract and was not substituted for it.

Issue references:

- Upstream issue: https://github.com/sgl-project/sglang/issues/38516
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2672
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2693
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/2689
