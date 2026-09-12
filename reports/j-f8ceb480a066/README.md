# Independent review of PR 2276 at 0f4eea4

Recommendation: **request changes**.

The candidate is test-only hardening; it changes no production source. The recorded
base already contains EAGLE draft-layer accounting in
`HybridSWAPoolConfigurator`, and the candidate's full pool-configurator suite
passes. I could not reproduce the original 8x B300 Inkling startup failure on
the prepared base because this job provides one AMD Instinct MI355X (gfx950),
ROCm 7.2, and no Inkling-NVFP4 weights.

The new tests do not prove the allocation contract stated in their prose. The
production unified sizing path computes draft capacity from the target pool's
virtual span and adds a page (`ceil_align(virtual_span, page_size) + page_size`).
Both candidate assertions instead charge draft layers for `full_tokens` only.
For hybrid full/SWA layouts this omits the SWA contribution to the virtual span
and the page padding. The implementation under test accounts for those bytes,
but the test's independently calculated `actual`/`used` values do not represent
the pools that are allocated. The tests should reconstruct the real draft token
capacity (or expose and test the allocator's conservation calculation) before
claiming that target plus all draft pools fit the budget.

This review therefore does not reject the already-present production fix. It
classifies the candidate as useful but incomplete test-only hardening and does
not claim that the original B300/CUDA/TP8 failure is fully resolved.

Raw command output is retained under `raw/`.
