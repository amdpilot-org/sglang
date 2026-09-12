# Independent review of PR 2751

Candidate: `5b6b88cb70670aab3ad36601bb168c569832647c`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the original issue as formulated.

## Failing before

The prepared checkout exactly matched the recorded base. Imports resolved to:

```text
sglang /job/repo/python/sglang/__init__.py
trace /job/repo/python/sglang/srt/observability/trace.py
```

With `OTEL_SERVICE_NAME=review-custom`, tracing initialization created:

```json
{"otel_service_name":"review-custom","resource_service_name":"sglang","trace_module":"/job/repo/python/sglang/srt/observability/trace.py"}
```

The base CLI help did not contain `--otlp-service-name`. All five SRT server process entry points passed the hardcoded `"sglang"`; diffusion passed `"sglang-diffusion"`.

## Passing after

At the exact candidate commit, independent Resource probes produced:

```text
explicit=cli-review, environment=env-review -> service.name=cli-review
explicit=None, environment=env-review -> service.name=env-review
explicit=None, environment unset -> service.name=sglang
explicit="", environment=env-review -> service.name=env-review
explicit=None, environment unset, diffusion default -> service.name=sglang-diffusion
```

The candidate's real in-process OTLP gRPC collector received spans with `service.name` values `cli-real`, `env-real`, and `sglang` for explicit, environment-only, and default cases. Its focused tests passed: `55 passed, 7 subtests passed`.

SRT and diffusion parsers both accept the custom option. Inspection found all current SRT process entry points now pass `otlp_service_name`; the async exporter is passed the already resolved value. Diffusion passes the configured value and retains `sglang-diffusion` as its compatibility default.

## Scope and limitations

No native files changed, so no native rebuild was applicable. No GPU/model execution was used because this is architecture-independent OpenTelemetry resource metadata. A model-backed server or distributed topology was not launched; the resource/export boundary and every current initialization call site were tested or inspected directly. An isolated diffusion `from_cli_args` construction encountered unrelated pipeline model validation, while its parser registration, parsed value, wrapper propagation, and default were independently verified.

Raw command output was preserved during revision switching under `/job/review-evidence-j-d534de1fbecc/`.
