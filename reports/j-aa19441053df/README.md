# Investigation report

The prepared base still contained the reported inclusive acceptance condition in
`speculative_sampling.cuh`. The zero-probability bug is reachable through both
the cumulative-probability comparison at a zero coin and the single-token
threshold comparison when that threshold is zero.

The correction guards the complete acceptance condition with positive target
mass and changes CDF selection to the half-open comparison `coin < cumulative`.
The regression matrix covers the two zero-mass threshold settings, an interior
boundary with an empty bucket, a positive first bucket at coin zero, and the
largest float32 coin below one.

Current related work was inspected before editing. These two open upstream PRs
contain the same native predicate correction and closely matching tests:

- https://github.com/sgl-project/sglang/pull/35788
- https://github.com/sgl-project/sglang/pull/35798

The assigned GPU was an AMD Instinct MI350X. The issue-specific native operator
was not registered by the installed ROCm `sgl_kernel`, and DFlash explicitly
reported that non-greedy verification was unavailable on this build/device.
Consequently, native execution and caller-path validation remain blocked. Raw
commands and output are retained in `raw/`.
