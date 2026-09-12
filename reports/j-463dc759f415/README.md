# Diffusion lifecycle correction review

This correction preserves candidate PR https://github.com/amdpilot-org/sglang/pull/2906
at `7f6725c124ac107fbc06daa0810757488cb6ea28` and addresses the concrete
transport counterexample reported by independent review PR
https://github.com/amdpilot-org/sglang/pull/2976.

On the exact candidate, a `ConnectionError` raised by
`sync_scheduler_client.forward` escaped `DiffGenerator.release_memory_occupation`
unchanged. The added regression also covers the `TimeoutError` produced by the
real scheduler client's receive-timeout path. Both are now converted to a
lifecycle-specific `RuntimeError`, with the original exception retained as the
cause.

No diffusion checkpoint or FSDP-capable workload was available. Therefore this
work does not claim model-specific sleep/wake/generation/refit correctness,
distributed support, memory behavior, or the issue's requested comparison with
kill-and-relaunch. The prior GPU evidence only exercised controller code already
present in the base and was not repeated as evidence for this transport fix.
