# Final chunked-prefill abort race investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34112

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1671

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The base already contained deferred cleanup for aborts while a request remained
in `scheduler.chunked_req`, but it did not cover the later overlap window after
the final `EXTEND` launched and before its queued result was committed. In that
window, `abort_request()` leaves a pending finish on the request. The scheduler
could optimistically admit it to decode, and the prefill result processor
appended the sampled token before consuming the abort.

The added regressions failed on the recorded base with an aborted prefill request
present in the scheduled decode set and with `output_ids == [101]` at streaming
time. The correction protects both boundaries: it filters only non-decode
pending-finish requests while a delayed result exists, and it consumes the
pending finish with zero accepted tokens while still advancing packed metadata
cursors for later live requests.

Raw command output is retained in the private runtime directory:

- `/tmp/amdpilot-repo-j-d58113c5097e/failing_before_scheduler_gate.log`
- `/tmp/amdpilot-repo-j-d58113c5097e/failing_before_result_commit.log`
- `/tmp/amdpilot-repo-j-d58113c5097e/passing_after_focused.log`
- `/tmp/amdpilot-repo-j-d58113c5097e/passing_after_extended.log`
- `/tmp/amdpilot-repo-j-d58113c5097e/gpu_inventory.log`
- `/tmp/amdpilot-repo-j-d58113c5097e/gpu_torch_check.log`

Source paths changed:

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`

No native source or library was changed, so no native rebuild applies.

The original Llama-3.2 weights and NVIDIA hardware were unavailable. The
assigned gfx950 was used only for an explicit Torch/ROCm execution check; it is
not presented as a model-level reproduction. The deterministic tests directly
exercise the current scheduler state and result objects at the two race
boundaries.
