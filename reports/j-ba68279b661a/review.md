# Independent review of amdpilot-org/sglang PR 1939

Upstream issue: https://github.com/sgl-project/sglang/issues/33388

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1972

Candidate reviewed: `aa9b9c3db97e485b64d2c7751e4f55c454920a2f`

Recorded failing base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Recommendation: **accept**. The candidate fully resolves the deterministic
contract in the original issue: tensor slices produced for the CPU pickle path
own proportional storage for `feature`, `precomputed_embeddings`, and directly
split tensor metadata, while accelerator slices remain views.

This is a source fix with regression hardening, not a test-only change. No
native source changed, so a native rebuild was not applicable.

## Independent evidence

The exact 512-image fixture reproduced on the recorded base with 1,048,576
logical bytes and a 543,484,451-byte pickle (518.307x). Each 2,048-byte feature
slice retained the 1,048,576-byte parent storage.

At the exact candidate commit, the same fixture produced a 1,375,278-byte
pickle (1.312x), and each slice owned exactly 2,048 storage bytes. An independent
non-contiguous CPU feature case dropped from 16.500x to 1.322x and likewise
owned storage equal to its logical bytes. A separate simple-split
`precomputed_embeddings` case dropped from 37.445x to 3.527x; the residual ratio
is fixed per-item/object pickle overhead on a deliberately tiny 256-byte item,
not retained parent tensor storage.

The candidate's focused suite passed all 9 tests. The imports used by the
independent cases resolved to the checked-out source files under
`/job/repo/python/sglang`, with Torch from the prepared interpreter.

On the assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`, ROCm/HIP
7.2.26015), the accelerator slice matched an independently constructed expected
tensor and retained the parent's storage pointer, validating the intended
no-copy accelerator boundary.

Raw measurements and the independent reproducer are retained in this folder.

## Limitations

- This directly tests the reported splitting and pickle mechanism, but not a
  full multimodal model, HTTP serving workload, or production-scale request.
- The reporter used NVIDIA H200/CUDA. Available accelerator validation used one
  assigned AMD Instinct MI355X/gfx950 GPU under ROCm 7.2.
- No model weights were needed or used. Therefore no model architecture,
  semantic accuracy, multi-node behavior, or end-to-end serving claim is made.
- The candidate changes Python only. The prepared environment records no native
  target for this checkout, and rebuilding native code was neither applicable
  nor performed.

