# Investigation notes

The base source had no pre-Ampere eligibility gate in either reported path:

- `qknorm_rope_jit._can_use_fused_qknorm_rope` invoked the cached JIT loader as
  part of its capability check.
- `residual_gate_add` attempted its JIT path once per device/dtype and only
  disabled it after a build or runtime exception.

The upstream issue had no comments, and searches for an existing SGLang pull
request mentioning the reported rsqrt/sm75 or pre-Ampere diffusion failure
returned no results at investigation time.

The correction rejects these two reported CUDA JIT paths when the detected JIT
target is below sm_80. ROCm is explicitly exempt because HIP capability values
are not NVIDIA SM versions. No CUDA source, compiler flags, credentials, host
configuration, or native library was changed.

The available GPU was an AMD Instinct MI355X (gfx950), not the reporter's RTX
2080 Ti. Consequently, the actual CUDA 13/glibc `rsqrt` compiler conflict and
the full Flux request latency remain unverified. GPU evidence here only proves
that the new NVIDIA gate leaves the HIP JIT path enabled and executing.
