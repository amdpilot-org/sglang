# Independent review of PR 3495

Candidate: `a47f208ff61d4660c5fdffeebc390fa69e25bf80`

The candidate is a partial fix. Its skill prose correctly removes plain-spec
`D` from the persistent slot cost, describes request-indexed intermediate
scratch, preserves ReplaySSM distinctions, documents attention-DP flooring,
and changes the explicit pin to `C * S + 1`. The checked-out allocator confirms
that a full `C * S` pool cannot satisfy the allocate-before-free stash.

However, both executable JSX ratio calculators can still recommend an unsafe
plain-spec ratio. They compute the ratio from the physical bytes needed for a
target `K = C * S + 1`. The current configurator does not invert that physical
allocation directly: its ratio path solves

```
K = floor((budget_slots - (1 + D)) / (1 + D / S))
```

before later sizing scratch from the admitted request count. For the original
boundary (`C=6`, `S=5`, `D=6`), the Kimi calculator's modeled budget is 74 slot
equivalents, and the configurator equation returns `K=30`, not the safe `31`.
For the Kimi default-style boundary (`C=64`, `S=5`, `D=8`) it returns `320`,
not `321`. With the Qwen calculator's model-specific physical scratch geometry,
the displayed ratio gives `K=318`, not `321`. Thus the calculators are not
consistent with the actual ratio configurator and can reintroduce the exact
stash boundary the prose warns against.

The candidate regression is text-presence testing and does not exercise this
ratio-to-pool inversion, so it passes despite the counterexample.

No native files changed, so no native rebuild was applicable. Tests used the
prepared ROCm 7.2 / Torch 2.11 interpreter. No GPU or H100 execution was used;
the original GLM-5.3-Flash TP8 + attention-DP8 eight-H100 workload remains
unverified in this environment.

