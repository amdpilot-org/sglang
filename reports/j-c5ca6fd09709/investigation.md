# Investigation notes

Upstream issue: https://github.com/sgl-project/sglang/issues/34676

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1540

The prepared base already contains a Mamba-specific admission safeguard in
`PrefillAdder`: new unified-Mamba state slots are gated by Mamba-recoverable
capacity instead of full-attention radix evictability. The added admission
tests show that removing this gate recreates the reported over-admission shape
(57 immediately available tokens, millions of full-attention evictable tokens,
and no schedulable Mamba slot).

That safeguard does not close the general late-allocation hole. A real
`alloc_extend()` miss still raised from `prepare_for_extend()` through
`_get_new_batch_prefill_raw()` and out of the scheduler loop. Upstream PR
https://github.com/sgl-project/sglang/pull/36473 proposed recovery for this
issue but was closed without merging. The implementation here narrows the
catch to allocator functions returning `None`, releases only request-pool slots
allocated by the failed call, restores chunk bookkeeping, requeues the admitted
requests, and marks the round full.

`failing_before_requeue.log` was captured with only the scheduler recovery
hunk temporarily removed. The injected late allocation miss escapes the real
`Scheduler._get_new_batch_prefill_raw()` call. `passing_after.log` contains the
same regression plus allocation cleanup and independent admission boundaries
passing against the delivered source.

The assigned device was detected as one AMD Instinct MI355X (`gfx950`) with
ROCm 7.2. The corrected path is host-side scheduler/allocation-control logic,
so the deterministic tests execute on CPU and no GPU kernel result is claimed.
The original Kimi-K3 TP8/DCP4 NVIDIA B300 workload was not run: this environment
has one AMD GPU and no Kimi-K3 weights. Consequently the exact distributed
traffic reproduction and model-level behavior remain unverified.
