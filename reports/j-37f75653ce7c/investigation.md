# Investigation: XPU token-ID synchronization

Upstream issue: https://github.com/sgl-project/sglang/issues/31011

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2334

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

At the recorded base, `Sampler._sync_token_ids_across_tp` unconditionally used
an integer `all_reduce(..., ReduceOp.MIN)`. The source and mirror issues were
still open with no comments, and the fetched mirror `main` had no later sampler
change. The repository's existing XPU communicator already uses
`all_gather_into_tensor`, providing an XCCL-supported alternative.

The regression models the issue's reported XCCL behavior (integer MIN silently
acting as SUM). Before the source change, identical rank values `[42, 42]`
became `84`, and the three XPU cases failed. After the change, XPU gathers the
integer token IDs and computes the minimum locally, while a separate non-XPU
case verifies that the original MIN all-reduce remains in use.

The assigned device is an AMD Instinct MI350X (`gfx950`), not an Intel XPU.
The retained GPU check executes the new local integer-min operation on gfx950
and compares it with independent Python minima. It uses a synthetic gather and
therefore does not reproduce or validate XCCL, multi-rank transport, grammar
decoding, or a full serving/model path. Intel XPU hardware and an XPU PyTorch
build are required for final backend confirmation.

