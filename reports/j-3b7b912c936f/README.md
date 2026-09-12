# Independent review of PR 2176

Reviewed candidate commit `3f71362934a47e8ad49b2ca1025e500f95f0ea28` against upstream issue https://github.com/sgl-project/sglang/issues/32065 and candidate mirror issue https://github.com/amdpilot-org/sglang/issues/2149.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` still unconditionally launches both post-quant kernels and has no bound on `hidden_dim / 8`. The candidate adds the required `1..1024` block-size check to both launchers, returns for `num_tokens == 0` in both, and independently returns for `topk == 0` in the masked launcher. Tensor matching and parameter construction remain before the early returns.

The candidate's regression failed 2/2 on the base and passed 2/2 at the exact candidate commit. An independent structural check confirmed the order `params -> block check -> warp check -> empty guard -> LaunchKernel` and exactly one launch site in each launcher. The patch also matches the still-open upstream PR 32066's eight source additions after accounting for the source-tree move from `jit_kernel` to `kernels/jit`.

Runtime qualification is architecture-blocked. The assigned device is one AMD Instinct MI350X (`gfx950`) with ROCm 7.2, while the issue is CUDA/H800-specific. Direct base and candidate calls entered the real SGLang JIT build, but hipcc stopped at the unchanged unconditional `#include <cuda_fp8.h>`. Thus neither the original CUDA launch error nor the fixed CUDA behavior and non-empty numerical equivalence could be executed here. This is not a full-model, DP-attention, or multi-node reproduction.

Recommendation: accept. The source correction fully covers both launch-contract defects, but CUDA runtime confirmation remains an explicit limitation.
