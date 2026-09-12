# PP + PD + DSpark verification notes

The prepared base rejected every DSpark configuration with `pp_size != 1`, so the reported path could not start. This contribution ports the complete current upstream candidate protocol from PR 33863 onto the prepared newer base and fixes its test compatibility with current state-type names, scheduler instrumentation, model configuration, and the extracted context-KV writer.

The implementation keeps hidden states inside the PP prefill pipeline. Each stage projects only the captured features it owns, adds that `[N, hidden_size]` contribution to `dspark_ctx_acc`, and the final stage applies RMSNorm once before draft KV injection. PD continues to transfer target KV and draft KV, not hidden states.

Evidence is under `logs/`. The initial candidate test run is retained as `failing_candidate_before_compat_fix.log`; it demonstrates the compatibility failure found after applying the upstream candidate to this base. The final focused suite and GPU numerical reference are also retained.

Hardware limitation: this job exposed one AMD Instinct MI355X. That is sufficient for independent GPU math validation, but not a real multi-rank pipeline-parallel serving run. No qualifying Kimi-K3 or DeepSeek-V4 DSpark weights were available.
