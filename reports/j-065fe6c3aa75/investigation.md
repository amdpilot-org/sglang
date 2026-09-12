# DSpark context-boundary candidate correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33454

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2077

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1959 at exact commit
`be7ba1b5424e57410b0af3e04d7bf3305970a897`.

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2043.

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Result

The review's concrete counterexample reproduced against the exact candidate:
`scripts/lint/check_registered_tests.py` exited 1 because the new regression
was placed under `test/registered/spec/`, while registered tests require one of
the recognized kind directories. The correction preserves the candidate's
runtime fix and moves the regression to `test/registered/unit/dspark/`.

Before correction, the validator reported the invalid `spec` kind. After the
move, the validator exited 0. The corrected regression and the existing DSpark
scheduler and ragged verification tests passed together: 39 tests and 18
subtests. Both changed Python runtime modules also passed `py_compile`.

The candidate's four focused boundary tests passed at the exact reviewed
commit. Independent review had already found no source-level, CPU, or
single-GPU counterexample to its position-boundary contract, and this correction
found none. Therefore no speculative runtime modification was added.

## Limitations

DeepSeek-V4-Flash-0731 weights, two NVIDIA B300 GPUs, and CUDA 13 were not
available. The original fused RoPE illegal-address failure and TP2 behavior
were not reproduced. The available evidence qualifies the planner arithmetic,
test placement, and existing CPU/single-AMD-GPU checks only. No native rebuild
was applicable because the changes are Python and test/report files.
