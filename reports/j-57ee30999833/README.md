# Dynamic verification for DFlash2 — validation record

This contribution adapts the complete open candidate from
`https://github.com/sgl-project/sglang/pull/36136` to base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, preserving newer Domino changes
at the three merge overlaps.

The prepared base reproduced the feature gap structurally: `DFlashWorkerV2`
always allocated, drafted, and target-verified `block_size` positions per row,
while only the DSpark worker owned confidence, SPS, and ragged scheduling.

The delivered opt-in `DFLASH_CONFIDENCE` path includes:

- selected-path survival estimates from a trained confidence head, with a
  selector-lattice fallback when the checkpoint does not carry one;
- optional per-position STS calibration;
- SPS cost-table selection of the cross-request optional-token budget;
- contiguous request-local verify prefixes with a target-verified progress
  floor and cutoff acceptance that prevents unverified suffix commitment;
- compact ragged target verification and CUDA Graph token-bucket reuse;
- overlap confidence relay, observability, SPS profiling, and explicit
  compatibility gates for unsupported backend/configuration combinations.

Raw test logs are retained beside this file. The first focused run exposed two
stale fixtures in the open candidate after its selector API changed; the final
fixtures validate the three-value return and full K-by-K lattice metadata.

No end-to-end DFlash2 weights were available. That prevents a claim of a fully
validated fix; the result is therefore `candidate_verified`, not `fixed`.
