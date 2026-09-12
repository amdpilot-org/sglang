# Decode throughput metric-scope investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/3050

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2515

The discrepancy is reproducible, but the current formulas are not two
measurements of the same quantity. `bench_serving` reports request-level TPOT
over each request's lifetime and whole-run aggregate output throughput. The
engine reports aggregate generated tokens over its most recent scheduler log
window. Ramp-up, prefill, draining, and configured versus actual concurrency
therefore prevent converting a TPOT percentile into the engine rate by simply
multiplying by `--max-concurrency`.

On the assigned gfx950 GPU, a real four-request serving run produced 1499.91
tok/s whole-run output throughput while steady engine windows reached
1918.36--1934.43 tok/s. Mean TPOT was 2.16 ms, median TPOT was 2.14 ms, and
measured average concurrency was 3.72 rather than the configured maximum of 4.
The raw client output, server log, and parsed summary are under
`evidence/before`.

The correction leaves all formulas and the stable engine log field unchanged.
It adds a concise scope note to the serving benchmark output directing users to
whole-run output throughput for an aggregate comparison. `evidence/after`
shows the note on the real serving path.

The deterministic tiny Llama fixture has random weights and validates only the
HTTP/engine execution and metric plumbing. It does not reproduce the reported
Qwen2.5-0.5B architecture, H800/CUDA environment, 2048-token prompts, or
16-request workload.

## Reproduce

Generate the qualified PR649 fixture outside the checkout, then run:

```bash
/tmp/amdpilot-repo-j-270f613ff5eb/venv/bin/python \
  reports/j-270f613ff5eb/run_issue_repro.py \
  --fixture /tmp/j-270f613ff5eb-runtime/tiny-random-llama \
  --output /tmp/j-270f613ff5eb-evidence/reproduction

/tmp/amdpilot-repo-j-270f613ff5eb/venv/bin/python -m pytest -q \
  test/registered/unit/benchmark/test_serving_metric_scope.py
```
