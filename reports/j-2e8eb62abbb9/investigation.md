# Consolidated correction evidence

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3495 at `a47f208ff61d4660c5fdffeebc390fa69e25bf80`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3504

The candidate's valid fixes are retained: plain speculative intermediates are
outside the allocator slot pool, admission remains `floor(K / S)`, explicit
pins reserve `C * S + 1` usable slots, attention-DP division floors, and
ReplaySSM remains distinct.

The remaining defect was in the executable JSX ratio calculators. They formed a
Mamba/KV ratio from desired physical allocation bytes, but the current
configurator consumes that ratio through its own joint solve:

```text
K = floor((budget_in_slot_bytes - (1 + D)) / (1 + D / S))
```

Therefore a safe target `K = C * S + 1` requires:

```text
budget_in_slot_bytes = K * (1 + D / S) + 1 + D
```

The Qwen calculator's CUDA deduplicated conv-window geometry is useful for
auditing physical allocation, but substituting it into this inverse is unsafe
because the configurator itself models `D` full-slot equivalents. The retained
raw evidence reproduces the review's exact `30`, `320`, and `318` candidate
outputs and the corrected `31`, `321`, and `321` outputs.
