# Independent review of PR 2092

Upstream issue: https://github.com/sgl-project/sglang/issues/32569

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2125

Candidate: https://github.com/amdpilot-org/sglang/pull/2092 at
`2a22c55e25a978d621fc3627d8ba4dd715edeef4`.

## Verdict

Recommendation: **accept**. The candidate is test-only hardening, not a new
production fix. Its base, `358c163250ad3b1f62939b01ce1314a0a31a0365`, already
contains the production changes that resolve the reported missing ROCm top-k
renormalizer and guard unavailable optional renormalizers. The candidate's
focused tests correctly exercise those fixes and passed independently.

## Failing-before evidence

The recorded base is newer than the defect and does not fail. To reproduce the
original failure in actual repository code, I checked out the parent of the
upstream ROCm top-k fix, `578edb240a6d6f6f2fa4c31497276955d7f73432`. On the
assigned gfx950 GPU, its HIP import path defined only the top-p renormalizer.
Calling `build_dflash_verify_target_probs` with top-k sampling enabled and the
dense path produced:

```
NameError: name 'top_k_renorm_prob' is not defined
```

This is the same missing-callable defect as the reported `NoneType` crash; the
precise exception differs because that historical revision left the HIP name
undefined rather than assigning `None`.

## Candidate validation

At the exact candidate commit, its regression passed (`2 passed`). Independent
GPU cases used 15 rows, vocabulary 1031, temperatures from 0.2 to 3.0, top-k
values 1, 2, 17, vocabulary size, and above vocabulary size, and top-p values
0.01, 0.5, 0.9, 0.999, and 1.0. Both dense and sparse builder paths were tested
with the live ROCm Triton functions and again with both optional function
globals forced to `None`. All four results matched an independent PyTorch
sort/mask/cumulative-mass reference; maximum absolute error was at most
`1.49e-08`.

The imported production module was
`/job/repo/python/sglang/srt/speculative/dflash_utils.py`; both live functions
came from `sglang.kernels.ops.sampling.renorm_triton`. The candidate changes no
native, C++, or FlyDSL source, so no native rebuild was required.

## Scope and limitations

Execution used one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) with Torch
2.11.0+rocm7.2 and HIP 7.2.26015. Kimi-K3 and RadixArk/Kimi-K3-DSpark weights
were unavailable, and this was not the reported eight-MI350X TP8 environment.
Thus the issue-specific GPU sampling/verify boundary is verified, but full
model loading, HTTP transport, semantic output, five-minute serving behavior,
and distributed execution remain unverified. No functional counterexample was
found within the directly testable contract.
