# Retract cache invalidation correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33783

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1931

Candidate: https://github.com/amdpilot-org/sglang/pull/1794 at `7fe1b5433ded03af468fe10969159eae92792666`

Independent review: https://github.com/amdpilot-org/sglang/pull/1893

## Reproduction

The review's independent idle-scheduler test was copied unchanged and run on
the exact candidate applied to the prepared base. It failed because all four
tokens inserted by a completed request remained matchable after
`pause_generation(mode="retract")`. Raw output is in
`evidence/candidate-idle-counterexample.txt`.

Source inspection independently confirmed the other two review observations:

- disaggregated prefill deliberately retains a live mid-chunk request and its
  KV, with an existing TODO documenting stale-weight prefix KV;
- `HiRadixCache.evict()` selects write-back eviction when configured, while
  `HiRadixCache.reset()` resets its cache controller and clears the host pool.

## Correction

The provenance-invalidating operation now occurs at scheduler pause scope,
after active requests drop their cache references. It resets the whole prefix
cache and clears request/token allocators even when there were no active
requests. This covers completed-request prefixes and hierarchical backing
state; it also supersedes the candidate's pressure-eviction implementation.

The existing disaggregated-prefill live-mid-chunk exception remains unchanged:
resetting its live KV before tearing down the sender is documented in source as
unsafe. The correction explicitly skips the reset in that case and retains a
regression asserting this limitation.

## Verification

The unchanged independent counterexample passes after the correction. The
candidate regression, scheduler pause suite, radix lock regression, and the
independent case pass together: 37 tests and 2 subtests passed. Raw output is
in `evidence/correction-focused-tests.txt`.

One assigned gfx950 GPU was visible and completed a simple Torch tensor
operation; details are in `evidence/gpu-inventory.txt`. The issue-specific
ownership and scheduler tests are deterministic CPU tests. The reported Qwen
weights were unavailable, so no full-model weight-swap, semantic-output,
multi-GPU, or multi-node claim is made.
