# Investigation: FP8 KV cache with Kimi-K3 DSPARK

Upstream issue: https://github.com/sgl-project/sglang/issues/32938

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1975

The upstream issue's follow-up H200 trace identifies the routing error: the
default `--speculative-attention-mode prefill` sends target verification to
FA3 despite an explicit `--decode-attention-backend flashmla`. With FP8 KV,
that path materializes BF16 cache chunks every step. The reported corrected
run adds `--speculative-attention-mode decode`, improving output throughput
from 173.32 to 668.79 tok/s and eliminating copy kernels above 1 ms.

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, Kimi-K3 already has logic
that selects decode-mode verification for supported backends, but the
non-DCP override is gated to SM100/SM103. Therefore the original H200 (SM90)
command returns no override and retains the problematic prefill mode.

The patch extends that existing mechanism only to Hopper and only treats
FlashMLA as suitable for the reported `fp8_e4m3` case. It does not change
backend selection on Hopper. The regression test directly covers the issue
configuration and checks three independent boundaries: FA3 remains on
prefill, BF16 FlashMLA remains on prefill, and non-Hopper gfx950 is unchanged.

The prepared host exposes one AMD Instinct MI350X (`gfx950`) through ROCm
7.2. It cannot execute or profile the CUDA FlashMLA/H200 route. Kimi-K3
weights, its DSPARK draft weights, and the reported 32-GPU topology are also
absent. Consequently this is a configuration-level verified candidate, not a
claim of local full-model or throughput reproduction.
