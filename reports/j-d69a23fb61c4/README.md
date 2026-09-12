# Duplicate Prometheus request-header correction

The independent review counterexamples were reproduced against candidate PR
https://github.com/amdpilot-org/sglang/pull/3300 at exact commit
`a5b60944c1762c4b9196fec907c2a72a58e9143b` before changing source.

`candidate_before.txt` records both mismatches against
`prometheus_client.make_asgi_app`: repeated `Accept` fields failed to negotiate
OpenMetrics and repeated `Accept-Encoding` fields failed to enable gzip.

The correction preserves the candidate's subprocess isolation, timeout,
singleflight, query filtering, and ordinary content negotiation. It changes
the endpoint to comma-join every value returned by Starlette's `getlist`, in
wire order, before passing the two negotiation fields to Prometheus.

`corrected_after.txt` is the same independent probe passing after the change.
`unit_tests_after.txt` records the focused suite, including regression cases
for both repeated fields. The inherited candidate reports retain its earlier
isolation and ordinary exposition-contract evidence.

No GPU or native build was used: this is deterministic CPU-side HTTP/ASGI
header handling. The original multi-node H100 PD deployment, sustained scrape
pressure, bootstrap timeouts, and decode connection-loss logs remain
unavailable and are not claimed as reproduced here.
