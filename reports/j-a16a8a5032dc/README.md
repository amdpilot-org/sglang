# Independent review of PR 2021

Upstream issue: https://github.com/sgl-project/sglang/issues/33181

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1943

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2494

Candidate: https://github.com/amdpilot-org/sglang/pull/2021 at `a6e013eeacd205cd0f60d4abd624ae8b99deac85`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the original parser-level issue within the tested contract.

The recorded base was the prepared checkout with no revision difference. It reproduced the leak through the real source imports: when a turn starts with `get_weather<|content_invoke_tool_json|>...`, both one-shot and 7-character streamed operation produced visible `get_weather` alongside the valid tool call. On the exact candidate, the same harness produced empty visible content and retained the valid tool call. Explicit-opener and thinking-first controls remained correct.

Independent cases exercised JSON and raw-text tool markers, an empty tool header, chunk sizes of one and irregular sizes, ordinary unframed text, ordinary text before a later model block, and `continue_final_message` immediate output. The focused reasoning/function-call parser suite passed 367 tests and 68 subtests.

The imported module path was `/job/repo/python/sglang/srt/parser/reasoning_parser.py`, confirming validation of the checkout rather than an installed copy. No native code changed, so no native rebuild applies. No GPU, HTTP server, model weights, full model, or distributed workload was used or needed for this deterministic parser defect; those serving/model layers remain outside the evidence claimed here.

One adversarial observation is retained for transparency: a stream ending in an incomplete control-token prefix drops that suffix on `finish()`. The recorded base behaves identically, so this is not introduced by PR 2021 and is not a counterexample to the original complete tool-call input contract.

Raw outputs and the independent harness are retained in this directory.
