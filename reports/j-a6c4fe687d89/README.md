# KV cache Prometheus metric validation

This change adds `sglang:kv_cache_usage_perc`, defined as the maximum usage
across the full-attention and sliding-window-attention KV pools. Mamba state
pool pressure is deliberately excluded. The new value retains the pool's full
precision rather than inheriting the two-decimal rounding of the legacy
aggregate `sglang:token_usage` metric.

The prepared base lacked the scheduler field; the failure is retained in
`evidence/failing-before.log`. The focused pool-semantics tests cover plain
attention, both full- and SWA-dominant hybrid attention, Mamba-dominant hybrid
SSM, combined SWA+SSM, and an adversarial unrounded value.

The end-to-end probe used the deterministic tiny random Llama fixture from
`amdpilot-org/sglang` PR 649 at commit
`f1d603677ca76a9ea21124a544e405c5b0cbd315`. Weights were generated under
`/tmp/amdpilot-repo-j-a6c4fe687d89/tiny-random-llama`, outside the checkout.
On the assigned AMD Instinct MI350X (`gfx950`), a real source-checkout server
performed prefill and decode work while the probe scraped `/metrics`. The new
gauge reported `0.01171875`, equal to the independently exported
`sglang:full_token_usage` reference expected for a plain-attention model;
`sglang:token_usage` showed its existing rounded value of `0.01`.

`run_metrics_probe.py` is derived from the qualified subreaper pattern and
retains the launch command, request, response, metrics scrape, server log, and
cleanup metadata under `evidence/server/`.

Limitations: the synthetic fixture validates transport, engine execution, and
plain-attention metric wiring only. Hybrid SWA and Mamba semantics were tested
directly on CPU because no qualified hybrid model weights were provided. No
native code changed, so a native rebuild and compiler/ISA validation were not
applicable. The server required SIGKILL after its owned process group did not
finish within the runner's 30-second SIGTERM grace period; descendants were
reaped and no node-wide state was changed.
