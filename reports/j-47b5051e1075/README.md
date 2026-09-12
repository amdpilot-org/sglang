# RouterGate correction generation 2

This report reproduces the concrete source counterexamples from
https://github.com/amdpilot-org/sglang/pull/2717 against candidate
https://github.com/amdpilot-org/sglang/pull/2677 at exact commit
`ce113fb8d9d5b8a6fac88e0d2364dfcfc7ff2b6b`.

The correction preserves the candidate's shared bf16-to-fp32 GEMM,
deterministic/small-token policy, neutral helper move, and downcast/bias fixes.
It routes the six specifically reviewed model families through `RouterGate`
while preserving checkpoint parameter names and DeepSeek's special signature.

Raw failing-before and passing-after evidence is in `raw/`. Full-model and
unavailable-backend validation was not performed; see `result.json`.
