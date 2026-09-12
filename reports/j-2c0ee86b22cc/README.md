# Client-disconnect cancellation investigation

The source at base `358c163250ad3b1f62939b01ce1314a0a31a0365` directly awaited
Starlette's `Request.is_disconnected()` in both timeout and active-output
branches of `TokenizerManager._stream_one_response`. A deterministic request
double that raises `asyncio.CancelledError` reproduced the reported escape in
both branches before the fix (`evidence/unit-before.log`).

The correction catches only cancellation originating below the disconnect
poll. If `asyncio.current_task().cancelling()` is nonzero, cancellation still
propagates. Otherwise the poll reports a disconnected client and the existing
branch aborts the scheduler request and terminates that request stack with its
existing `ValueError`.

The focused regression covers both polling branches, a normal connected
result, and explicit cancellation of the handler task. A real server probe on
the assigned single gfx950 also closed a raw TCP connection without reading
the response, then observed a live process, HTTP 200 health, and a successful
follow-up generation. The generated fixture is private under
`/tmp/amdpilot-repo-j-2c0ee86b22cc/tiny-random-llama`; its deterministic weight
SHA256 is `6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`.

This does not reproduce or qualify the unavailable NVIDIA SM120, four-GPU,
DeepSeek-V4.1-Flash, 200k-token deployment. See `result.json` for exact commands
and limitations.
