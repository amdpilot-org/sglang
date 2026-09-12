# Issue 32065 investigation

The recorded base still constructs invalid launch configurations in both
post-quant wrappers. Upstream PR 32066 contains the same narrow fix against the
former source path, but remains open. This checkout has since moved the source
to `python/sglang/kernels/jit/csrc/deepseek_v4/` without those guards.

The retained regression was run against a base-commit copy of the header and
failed twice, then passed twice with the patch. Direct kernel cases were also
attempted on the assigned MI350X/gfx950 GPU. All reached the actual SGLang JIT
compiler, which invoked hipcc for gfx950 and failed before launch because this
CUDA-specific source unconditionally includes `cuda_fp8.h`. The raw compiler
and pytest output is under `raw/`.

Consequently, this change is supported by issue-specific source evidence and a
failing-before/passing-after source regression, but it is not claimed as a
CUDA runtime or end-to-end DP-attention verification.
