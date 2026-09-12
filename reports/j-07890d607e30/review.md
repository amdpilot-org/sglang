# Independent review of PR 2543

Candidate: https://github.com/amdpilot-org/sglang/pull/2543 at `0db576affad6f3fa606de7143b9d3925d1494641`

Upstream issue: https://github.com/sgl-project/sglang/issues/28157

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2471

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3217

## Recommendation

Request changes. The candidate is a substantive partial fix: it moves synchronous multiprocess collection out of the server event loop, bounds it, and rejects overlapping collections. It passes its focused tests and prevents the deterministic starvation mechanism. It should not be accepted as written because replacing Prometheus' ASGI app with unconditional `generate_latest()` regresses existing `/metrics` HTTP semantics.

## Evidence

The prepared branch was exactly the required base `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, delaying the actual `prometheus_client.asgi._bake_output` call by 350 ms caused 331 ms of event-loop lag. This reproduces the issue's shared-event-loop starvation mechanism using the checked-out implementation.

The repository was then detached at the exact candidate commit. Import inspection reported `/job/repo/python/sglang/srt/utils/common.py`, confirming that tests loaded candidate source rather than an installed SGLang copy. The candidate's four focused exporter tests passed. Its supplied deterministic regression also passed: the first scrape returned 200, the overlapping scrape returned 503, and measured event-loop lag was 0.000 seconds. An independent probe with a trivial `/health` route measured 0.001 seconds of lag during a 350 ms metrics-child delay, with health 200 and the overlapping scrape 503.

The same independent probe found three regressions relative to the replaced `prometheus_client.make_asgi_app` path:

- `GET /metrics?name[]=alpha_total` also returned `beta_total`; the standard restricted-registry query was ignored.
- `Accept-Encoding: gzip` produced no `Content-Encoding`; compression negotiation was lost.
- `Accept: application/openmetrics-text` still returned `text/plain; version=0.0.4`; OpenMetrics negotiation was lost.

The probe exited 1 on the first contract assertion. These are production `/metrics` API behaviors, not an unrelated smoke. The candidate tests mock only child lifecycle/output and do not cover them.

## Classification and limitations

This is a partial original-issue fix with useful regression coverage, not test-only hardening. It is not a fully verified original deployment fix: the eight-H100, multi-node, single-tokenizer PD topology, real scrape load, short bootstrap read timeouts, and decode-side connection-loss logs were unavailable. The prepared host is ROCm 7.2/gfx950. GPU execution was irrelevant to this CPU-side ASGI/Prometheus path and was not used. No native source changed, so no native rebuild was required. No model fixture was used because transport/engine startup would not exercise Prometheus collection or the separate bootstrap server contract.

The existing upstream related PR https://github.com/sgl-project/sglang/pull/28163 was inspected; its existence was not treated as proof for this candidate.
