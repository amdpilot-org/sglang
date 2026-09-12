# Multi-layer EAGLE KV sizing correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31588

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2387

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2276 at `0f4eea4aca21cda5a900f6e9b9cffa87b4965ac5`

Independent review: https://github.com/amdpilot-org/sglang/pull/2352

The candidate's two added tests passed before modification, but both measured
each full-capacity draft layer with `full_tokens`. The allocation implementation
instead uses the target pool's virtual byte span, rounded up to a page, plus one
page. Replacing the assertions with that allocation geometry failed on the
prepared base: 72,456,601,600 bytes were required for a 70,276,402,380-byte
budget (`raw/corrected_targeted.log`).

The correction uses the exact solver for full-capacity draft pools regardless
of unified-memory mode. It keeps ordinary draft-SWA layers on the SWA capacity,
and applies the virtual-span rule only to full-attention, full-capacity SWA, and
DFLASH draft storage. Tests also cover MXFP8 data plus scale bytes.

`raw/full_suite_after.log` records 44 passing tests and 7 passing subtests.
`raw/gpu_check.log` records a real FP32 numerical check on the assigned single
AMD gfx950-class GPU. This does not reproduce the unavailable 8x B300 CUDA 13,
TP8, Inkling-NVFP4 startup sequence or validate model-specific execution.
