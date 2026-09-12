# DFlash terminal capture correction

This correction preserves the valid terminal-boundary work from candidate
https://github.com/amdpilot-org/sglang/pull/2130 at exact commit
`8b5ab59b905622ea414746c75e1f70dc4065c928` and addresses the concrete
counterexamples independently reported by
https://github.com/amdpilot-org/sglang/pull/2191.

The candidate-before regression output is retained in `raw/failing_before.log`.
It demonstrates all three failures using the actual candidate source. The
passing focused suite is retained in `raw/passing_after.log`, and the assigned
GPU arithmetic boundary check is in `raw/gpu_boundary_check.log`.

Qwen3.5-4B weights, Ascend hardware, and a multi-node FlashInfer MNNVL setup
were unavailable. Accordingly, this report makes no full server/model, Ascend,
semantic-accuracy, or distributed-workload claim.
