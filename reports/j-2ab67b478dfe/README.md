# Queued streaming-session abort correction

This correction preserves candidate PR #2283's explicit queued-abort and
timeout fixes, then covers the remaining queue-limit paths identified by the
independent review in PR #2377.

On the exact candidate source (`08a1faa06f054b939b1a672f3158b90632c713a2`),
the two added regressions failed because an incoming queue-full rejection did
not call `Session.abort_req`, while priority eviction left a queued Mamba
request with `finished_reason=None` and did not release its cache state.

The correction uses one pre-execution streaming cleanup helper for explicit
queued aborts and both queue-limit outcomes. Priority eviction marks
`FINISH_ABORT` before releasing Mamba cache state.

Raw failing-before and passing-after pytest output is retained in `evidence/`.
No model weights, GPU execution, or native rebuild were required for these
scheduler-level lifecycle counterexamples.
