# DeepSeek-V4-Flash-0731 hidden-size investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33693

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1760

## Finding

The prepared source already contains an issue-specific solution. The published
checkpoint config labels the model FP8, but its routed-expert tensors use packed
FP4 in `I8` storage. For example, hidden size is 4096 while the physical K
dimension of `layers.0.ffn.experts.0.w1.weight` is 2048. The Aug 2026 failure
sent this packed tensor through `fused_experts_impl(..., use_fp8_w8a8=True)`,
whose production shape check reports `AssertionError: Hidden size mismatch`.

Current source avoids that route:

1. `try_detect_fp4_experts` reads a routed-expert safetensors header.
2. `ModelConfig` records `is_fp4_experts=True` for `I8`/`U8`/`F4` storage.
3. The DeepSeek V4 override selects `flashinfer_mxfp4` for a non-A2A H20/SM90
   launch, instead of the FP8 Triton runner that raised the assertion.

`checkpoint_metadata.json` records the public config and shard-header evidence.
Only the safetensors header was range-fetched; no model weights were downloaded.

## Reproduction and validation

`reproduce_hidden_size_route.py` invokes the checked-out production
`fused_experts_impl` on the assigned gfx950 GPU. It intercepts execution at
kernel preparation, after the production shape checks, to isolate routing from
hardware-specific kernels. The 0731 packed shape fails on the legacy FP8 route
and passes on the current FP4 route. A true-FP8 shape passes and malformed FP4
is independently rejected.

`test_checkpoint_detection.py` verifies packed FP4, true FP8, missing metadata,
and remote-path boundaries. It also models the reported NVIDIA H20 as SM90 and
verifies current automatic selection of `flashinfer_mxfp4`; a true-FP8 H20
retains generic routing.

The repository's existing callable-level override test was also attempted on
gfx950. It fails because its SM100 mock does not override the host platform's
`is_hip=True`; this is a test portability issue and is not evidence against the
H20 branch. The report-specific test supplies a complete NVIDIA platform mock.

## Limitations

The assigned device is an AMD Instinct MI350X (gfx950), not the reported NVIDIA
H20 (SM90). The 166,878,536,440-byte checkpoint was not available locally, so a
full model load, CUDA FlashInfer execution, and HTTP serving reproduction were
not performed. The GPU fixture validates the exact production shape gate only;
it does not claim model semantic accuracy, CUDA kernel execution, or a
distributed workload.

Raw command output and downloaded issue/checkpoint metadata are retained under
`/tmp/amdpilot-repo-j-4d0012c4c996/evidence/` in the prepared environment.
