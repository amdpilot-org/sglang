# Independent review of amdpilot-org/sglang PR 1959

Upstream issue: https://github.com/sgl-project/sglang/issues/33454

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1991

Candidate: https://github.com/amdpilot-org/sglang/pull/1959 at exact commit
`be7ba1b5424e57410b0af3e04d7bf3305970a897`.

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
The image-prepared checkout matched that commit and was clean before review.

## Finding

The runtime change addresses the source-level contract. On the recorded base, a
six-token DSpark window starting at sequence length 1,048,574 produced positions
1,048,574 through 1,048,579, including four positions outside a 1,048,576-row
RoPE table. At the exact candidate commit, the worker intersects each requested
verify length with both the remaining model context and remaining generation
budget. It forces clipped requests through the compact ragged target path and
repeats the final legal position in the unused fixed-width draft tail.

Independent mixed-row cases measured verify lengths `[6, 3, 2, 1]` and no
position at or above the context bound on both CPU and the assigned AMD GPU.
The candidate regression passed, as did the existing DSpark scheduler and
ragged-layout tests (39 tests plus 18 subtests). No functional counterexample to
the reported positional-overflow contract was found.

The recommendation is nevertheless `request_changes`: the new regression is
located at `test/registered/spec/dspark/test_dspark_context_boundary.py`, but
registered tests must use a recognized kind directory such as `unit` or `e2e`.
The repository registration validator rejects the candidate. The candidate's
committed result claims this validator exited zero, while its retained raw
validator output is empty. This review does not modify the candidate; the test
placement and review evidence should be corrected there.

## Environment and scope

The prepared interpreter imported SGLang source from `/job/repo/python`.
The candidate changes only Python and report/test files; it has no native C++,
CUDA, HIP, or FlyDSL changes, so no native rebuild was applicable. `py_compile`
passed for both changed runtime modules.

The available device was one AMD Instinct MI355X under PyTorch 2.11.0+rocm7.2
and HIP 7.2.26015. The reported environment was two NVIDIA B300 GPUs with CUDA
13 and DeepSeek-V4-Flash-0731. Those weights and that TP2/CUDA architecture were
not available. Consequently this review verifies the actual planner arithmetic,
GPU tensor execution, ragged layout tests, and source routing, but not the
original full-model CUDA kernel crash, TP2 synchronization, or model semantics.

Raw commands and output are retained under `raw/`.
