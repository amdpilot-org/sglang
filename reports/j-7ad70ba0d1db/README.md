# Investigation of decode radix cache + HiCache restart failure

Upstream issue: https://github.com/sgl-project/sglang/issues/30322

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2379

## Result

The reported failure was not reproduced in the prepared checkout at base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The current source already contains
an asynchronous local-restore state machine for the condition in the report
(`l1=0, l2=0, l3=64`):

- `HiCacheRestoreGatedKVReceiver.poll()` converts transport success to
  `KVPoll.Transferring` while local restoration is pending.
- `_try_hicache_queue_load_back()` does not attempt L2-to-device load-back until
  the L3 prefetch reports completion, then re-matches and requires the restored
  indices to cover the entire prefix promised to prefill.
- `_process_hicache_local_restores()` marks the request ready only after the
  asynchronous load-back event completes.
- `_commit_hicache_local_restore_to_req()` installs the restored indices before
  the request is released from the transfer queue.

The fix PR referenced from the issue,
https://github.com/sgl-project/sglang/pull/32278, remains open and is not merged.
The prepared source has evolved beyond it and retains the essential readiness
gating without that PR's mutation of the promised prefix lengths.

Three regression cases were added to preserve issue-specific evidence:

1. An L3-only 64-token hit with an incomplete prefetch remains pending and does
   not invoke load-back.
2. The same hit after prefetch completion queues all 64 restored indices and
   retains the restored-node lock receipt.
3. A successful network transfer remains gated until local restore changes from
   `PENDING` to `READY`.

The final focused suite passed 25 tests. Raw output is in
`raw/focused-final-tests.txt`; the issue/PR snapshots and environment inventory
are retained in `raw/`.

## Limitations

The source report requires eight H20 decode ranks, eight prefill CP ranks,
Mooncake transfer, persistent HiCache storage, and a decode-process restart.
This job had one AMD Instinct MI355X (`gfx950`) and no supplied model weights or
multi-node deployment. Consequently, no full serving, Mooncake, NVIDIA H20,
CP=8/DP=8, persistence, or process-restart reproduction was claimed. The added
tests deterministically validate the control-flow invariant and tensor-index
coverage only; they do not validate transport hardware, model semantics, or a
distributed workload. No native component was changed or rebuilt.
