# Investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/37931

Mirror issue: https://github.com/amdpilot-org/sglang/issues/835

The reported image is an SM12x-only preview at `452239a74f`; the prepared
checkout is main at `358c163250ad3b1f62939b01ce1314a0a31a0365`. Current main
still converted an entire FP8 shared-expert matrix to BF16 before MXFP4
quantization. `block_quant_dequant` expands the block scales to a full FP32
matrix, multiplies into another full FP32 result, and then materializes BF16.
Those weight-sized temporaries explain a conversion-time peak independent of
loader thread count.

The correction converts scale-row-aligned chunks. MXFP4 quantization is local
to each 32-element row group, so chunking rows does not change the packed wire
format. A regression forces multiple chunks and compares exact packed bytes and
scales with the former single-chunk behavior. It also covers grouped weights, a
partial final 128-row FP8 block, and invalid partial 32-column MXFP4 groups.

The assigned hardware is one AMD MI355X (gfx950), not two NVIDIA GB10 systems.
The 156.31 GiB DeepSeek-V4-Flash-Vision-Exp weights and the SM12x-only b12x
runtime were unavailable. Therefore the original distributed serving command
could not be reproduced or declared fixed. The deterministic tiny Llama fixture
does not exercise this model-specific shared-expert conversion and was not used
as a substitute.
