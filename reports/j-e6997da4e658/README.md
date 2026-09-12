# DFlash2 dynamic-verification review correction generation 2

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2861 at exact commit `cdd6ac7f5940e15c97cafbee61c0baf3ce4a9603`.

Independent review parent: https://github.com/amdpilot-org/sglang/pull/2947.

The review's concrete STS overflow counterexample was independently reproduced against the exact candidate. A finite positive JSON value of `1e308` passed parsing, overflowed to float32 infinity in `DFlashWorkerV2`, and changed a finite calibrated logit to confidence `0.5`. See `failing-before.log`.

The correction preserves the candidate's dynamic scheduling, trained-head integration, STS calibration, ragged verification, graph bucketing, overlap, and compatibility work. It rejects calibration temperatures that are non-finite in the runtime float32 representation before installing them on the confidence head. The focused regression passes in `passing-after.log`; broader confidence/logits/scheduler/STS coverage is in `focused-tests.log`, and the shared GPU scheduling kernels pass 22 independent numerical-reference subtests in `gpu-kernel-parity.log`.

No compatible DFlash2 checkpoint with a trained survival head or compatible target/draft pair was supplied or cached. Consequently the candidate's assumed real checkpoint schema and end-to-end lossless semantics across sampling, grammar, continuous batching, overlap relay, and live compact graph replay remain unverified. Mixed-quality concurrent serving throughput also remains unmeasured. The tiny Llama transport fixture cannot qualify those DFlash2-specific claims and was not used as evidence.
