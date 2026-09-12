# Event-loop preprocessing investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/30770

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2363

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The base directly invoked synchronous OpenAI conversion from `handle_request`
and synchronously invoked the regular tokenizer fallback from `_tokenize_texts`.
The companion upstream fix is PR 30771 and remains open.

The imported current regression tests failed on the unmodified base for both
direct paths. The ASGI integration also failed because `/ping` could not run
until the blocked chat conversion watchdog fired. Raw before/after output is
retained in `raw/`.

The correction uses one TokenizerManager-owned worker to preserve serialized
access to tokenizer/template state, copies request context into that worker,
and leaves the existing dynamic-batch tokenizer path unchanged. A fallback to
`asyncio.to_thread` retains compatibility with lightweight or alternate manager
implementations that do not own the executor helper.

This is a CPU-side scheduling fix. No GPU execution, model output validation,
or multi-node claim is made.
