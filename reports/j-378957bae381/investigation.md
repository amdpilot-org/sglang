# Investigation evidence

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`,
`Scheduler._get_new_batch_prefill_raw` called `PrefillAdder.add_chunked_req` and
then unconditionally entered the `waiting_queue` admission loop. This matches
the reported within-pass ordering defect: `PrefillAdder.budget_state()` already
reports `OTHER` for an exhausted chunk budget, but the scheduler did not consult
the updated state.

The failing-before run is retained in `failing_before.log` and its numeric exit
status in `failing_before.exit_code`. Both exhausted states (`OTHER` for the
chunk/input-class budget and `NO_TOKEN` for the total-token budget) performed an
extra waiting-queue admission iteration. The independent `CONTINUE` boundary
correctly retained that iteration.

The correction uses a local admission iterable. After a resumed chunk, a
non-`CONTINUE` budget state empties only that iterable. It does not return early:
LoRA accounting, allocator group balancing, final waiting-queue maintenance,
and construction of the batch containing the resumed request still execute.

GPU execution and a model-serving fixture are not relevant evidence for this
CPU scheduler-control-flow defect, so neither was used.
