# Independent review of PR 2765

Reviewed exact candidate `68287fe4f4bd226849b0c5cf8a494349496d1a90`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
and the original RouterGate issue.

Recommendation: **request changes**. The candidate is a real partial fix and
passes its focused regression tests, but it does not fully resolve the original
issue. Many MoE routers remain on `ReplicatedLinear` or model-local gate
implementations. On the assigned MI355X, deterministic mode also produced
different logits for the same token at batch sizes 1 and 8 for a realistic
4096-by-128 router, contrary to the shared layer's batch-invariant contract.

The review used the prepared interpreter and confirmed imports came from the
candidate checkout. No native files changed, so no native rebuild applied.
Full-model testing was blocked by unavailable model weights; unavailable
architectures and backends are listed in `result.json`.

Raw evidence in this report was copied from a review evidence directory outside
the checkout after returning to the required review branch.
