# Independent review of candidate PR 2788

Upstream issue: https://github.com/sgl-project/sglang/issues/31327

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2762

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2820

Candidate: https://github.com/amdpilot-org/sglang/pull/2788 at `6919f2e20ec866a7df61897448ff7f7c86d60904`

## Recommendation

Accept. The candidate fully resolves the original dashboard-only request.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` fails the requested contract because it has only the existing batch-level `Cache Hit Rate` panel. At the exact candidate commit, the new panel uses cached prefill token rate divided by cached-plus-computed prefill token rate, excludes decode, presents a 0–1 ratio as a percentage, and leaves the old panel unchanged.

This conclusion is based on checkout-level reproduction, source inspection, the candidate regression suite, and an independent adversarial script rather than the candidate prose. Raw evidence is retained outside the checkout under `/job/review-evidence-j-d2b1192d045d/`.

## Scope and limitations

The relevant source paths are `examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` and `python/sglang/srt/observability/metrics_collector.py`. The latter defines `sglang:realtime_tokens_total` as a counter with `prefill_compute`, `prefill_cache`, and `decode` modes.

No native source or library changed. The prepared environment records `native: null` and `wheel: null`, so a native rebuild is neither available nor applicable. GPU, model, serving, and distributed execution are also inapplicable to this static dashboard contract.

No live Grafana/Prometheus deployment was available. The candidate follows the dashboard's established underscore-form metric query naming; raw `prometheus_client` exposition preserves colon-form names, so the complete scrape pipeline's name presentation was not independently observed. Existing dashboard queries and the related open upstream implementation use the same underscore convention.

The zero-prefill case remains no-data because both operands sum to zero. That is an explicitly disclosed operational edge, not a remaining counterexample to the requested token-weighted definition.
