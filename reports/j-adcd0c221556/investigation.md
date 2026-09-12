# Late prefill allocation recovery correction

- Upstream issue: https://github.com/sgl-project/sglang/issues/34676
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1856
- Candidate PR: https://github.com/amdpilot-org/sglang/pull/1728
- Independent review PR: https://github.com/amdpilot-org/sglang/pull/1819
- Exact reviewed candidate: `9ee7239c734292e901b5dbd9792ce261c6ae27b9`
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The review counterexample reproduced against the exact candidate. After an
ordinary request was admitted, an injected `KVCacheOOMError` caused the
candidate to requeue it and set `running_batch.batch_is_full = True`. Because
the running batch was empty and there was no active chunked request, the next
call returned at the initial full-batch guard. The retained request was not
retried: `prepare_for_extend` had one call instead of two.

The consolidated correction preserves the candidate's typed allocation error,
request-pool rollback, requeue behavior, and active chunked-request ownership.
It now retains the full flag only when the running batch is nonempty and can
make decode progress. An empty scheduler clears the flag, allowing a later pass
to retry the requeued request. The inverse boundary test confirms that a
nonempty running batch still receives allocation backpressure.

## Evidence

- `raw/counterexample_before.txt`: the exact candidate fails the two-pass
  regression with `prepare_for_extend.call_count == 1`.
- `raw/focused_after.txt`: the corrected source passes 39 tests and 12 subtests,
  covering typed late allocation failure, request-slot rollback, ordinary
  retry, nonempty-running-batch backpressure, active/new chunk ownership, and
  PrefillAdder boundaries.
- `raw/gpu_inventory.txt`: the assigned device is one AMD Instinct MI355X,
  gfx950.

## Limitations

The reported eight-NVIDIA-B300 TP8/DCP4 Kimi-K3 hybrid-Mamba workload and model
weights were unavailable. The deterministic control-flow regression exercises
the actual scheduler and allocation implementation on CPU; no GPU numerical,
model-semantic, CUDA-specific, or distributed claim is made. The tiny Llama
fixture cannot validate hybrid Mamba allocation and was not substituted as
evidence. No native source changed, so no native rebuild was required.
