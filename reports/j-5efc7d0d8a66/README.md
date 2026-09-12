# Auto-truncation correction evidence

This correction was produced from base `358c163250ad3b1f62939b01ce1314a0a31a0365` after independently reviewing:

- candidate PR: https://github.com/amdpilot-org/sglang/pull/3108
- independent review PR: https://github.com/amdpilot-org/sglang/pull/3180
- exact candidate commit: `5870e01368cead04c7d273ffbd0b3a89b4108f88`

The candidate's feasible reserved-token fixes are preserved. The remaining counterexample is corrected by rejecting a reserved-token budget that exceeds the context length, since truncating prompt and completion tokens cannot reduce that immutable budget.

Evidence files retain the exact candidate failure, candidate compatibility suite, corrected focused suite, and corrected adversarial probe. No compatible EAGLE draft model was available, and the qualified tiny-Llama fixture cannot exercise this architecture-specific reserved-token path, so no unrelated GPU smoke is presented as proof.
