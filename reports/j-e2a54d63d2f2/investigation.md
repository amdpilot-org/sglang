# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/29738

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2432

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The base source still imported `deep_gemm` at module scope only when
`ENABLE_JIT_DEEPGEMM` was true, but `tf32_hc_prenorm_gemm` used that global
from the independently gated HC-prenorm path. A focused regression loaded the
actual `entrypoint.py` with the JIT gate false. Before the correction, its two
nonempty cases reproduced `NameError: name 'deep_gemm' is not defined`; the
zero-token fast path passed.

The correction imports `deep_gemm` locally after the zero-token return. This
keeps the empty boundary dependency-free, forwards a nonempty call to an
installed dependency, and exposes `ImportError` rather than a misleading
undefined-global error when the dependency is unavailable.

Related upstream work was inspected before implementation. PRs
https://github.com/sgl-project/sglang/pull/29740 and
https://github.com/sgl-project/sglang/pull/38442 are open and propose the same
local-import direction; neither change was present in the prepared base.

Raw command output is retained under `reports/j-e2a54d63d2f2/raw/`. The host
probe identified one AMD Instinct MI355X (`gfx950`) using ROCm 7.2. This is not
the reported CUDA H100/SM90 environment, and DeepSeek-V4 weights were not
available, so no full-model, CUDA-kernel, semantic, or numerical claim is made.
