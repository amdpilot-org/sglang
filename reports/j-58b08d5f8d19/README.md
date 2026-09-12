# Kimi K2 parser correction generation 2

This correction preserves candidate PR https://github.com/amdpilot-org/sglang/pull/2581 at exact commit `6791b29b61cb0054a6be3387528f08f31d1872f8` and addresses the concrete counterexamples from review PR https://github.com/amdpilot-org/sglang/pull/2595.

The candidate was checked out directly before edits. `candidate-failing-before.txt` records both reproduced failures: a completed call masked an oversized unclosed second call, including after `finish()`, and singular K2-Thinking section markers were not recognized by the non-streaming API.

After correction, the size limit is checked against each active unclosed call after complete predecessors have been consumed. End-of-stream releases any remaining marker holdback and resets state. `has_tool_call` and `detect_and_parse` accept both plural and singular section starts.

`passing-after.txt` contains the focused regression, integration, and adversarial-harness results. This is deterministic CPU parser validation. No Kimi-K2/K2.5 weights were available, so model semantics, actual model token emission, HTTP behavior, and distributed behavior remain unverified.
