# RouterGate correction generation 1

This correction preserves the valid work from
https://github.com/amdpilot-org/sglang/pull/2662 at exact commit
`86a6658c7c3de43a5de96b8726423da0994df53b` and addresses the concrete
counterexamples independently reported in
https://github.com/amdpilot-org/sglang/pull/2666.

The candidate AITER failure was reproduced before editing. The same seeded
shapes were then rerun after correction. Raw logs, the executable reproducer,
consumer audit, and import checks are retained in `raw/`; structured claims
and limitations are in `result.json`.
