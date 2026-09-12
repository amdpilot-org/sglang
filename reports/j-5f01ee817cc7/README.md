# OTel trace relay investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38210

Mirror issue: https://github.com/amdpilot-org/sglang/issues/772

The base implementation disabled both synchronous and asynchronous request
trace contexts whenever pickle materialized them in an uninitialized process.
That is destructive for routing processes, which only relay the object to an
already initialized scheduler or detokenizer.

The correction preserves the original serialized state in such relay-only
processes and returns it unchanged on the next pickle hop. Normal initialized
destinations continue through the existing reconstruction path, and genuinely
disabled contexts remain disabled.

An open upstream candidate instead initializes tracing in the routers. A real
multi-tokenizer/multi-detokenizer gfx950 run rejected that approach because the
main router can initialize OpenTelemetry before child process creation. The
set-once provider and exporter-thread state are then inherited across fork,
where the thread is absent and later initialization cannot replace the global
provider safely.

Raw evidence is retained outside the checkout under
`/tmp/amdpilot-repo-j-5f01ee817cc7/evidence/`.
