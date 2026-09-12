# Independent review of PR 3049

Reviewed `https://github.com/amdpilot-org/sglang/pull/3049` at exact commit
`df6227fa7cc01d5e6fd84b84f8efa2dba3ad12b8` against upstream issue
`https://github.com/sgl-project/sglang/issues/5979` and candidate mirror issue
`https://github.com/amdpilot-org/sglang/issues/2990`.

Recommendation: **accept**. The candidate fully resolves the original feature
request by exporting `sglang:kv_cache_usage_perc` from the scheduler's existing,
unrounded KV-only utilization calculation. The base computes this value but
discards it, so the metric is absent before the candidate. The candidate
publishes it without changing the legacy rounded/all-pool `token_usage` metric.

Review findings:

- The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` was also the image-prepared checkout; there was no base mismatch.
- Base reproduction confirmed that `PoolStats.get_kv_token_stats()` returned `0.123456`, but `SchedulerStats` had no `kv_cache_usage_perc`; legacy `token_usage` was rounded to `0.12`.
- The candidate's five focused regression tests passed at its exact commit.
- Independent adversarial cases covered empty/full boundary values, full- and SWA-dominant hybrid KV pools, precision beyond two decimals, and Mamba-dominant pressure. The KV metric remained `max(full, swa)` and excluded Mamba while legacy `token_usage` included it.
- All 15 scheduler-component unit tests passed on the candidate.
- Imports resolved to the source checkout under `/job/repo/python`. The prepared AITer native module loaded from `/tmp/amdpilot-repo-j-4fe3ebab7085/cache/aiter/module_aiter_core.so`.
- No C++, FlyDSL, or other native source changed, so no native rebuild was applicable.
- Candidate-retained serving evidence shows a real request and metrics scrape on AMD Instinct MI350X (`gfx950`): `kv_cache_usage_perc=0.01171875` exactly matched the pre-existing independent `full_token_usage=0.01171875` reference for the plain-attention fixture, while legacy rounded `token_usage=0.01`.

Limitations:

- The reviewer did not rerun the approximately 47-second GPU server probe; the candidate's raw request, response, metrics, metadata, and server log were independently inspected and retained here before restoring the review branch.
- The tiny random Llama fixture validates transport, engine execution, and plain-attention metric wiring, not semantic model accuracy.
- No qualified hybrid SWA or Mamba model weights were available for end-to-end GPU validation. Those paths were tested directly at the scheduler accounting boundary.
- The environment is ROCm 7.2 with Torch `2.11.0+rocm7.2` on AMD Instinct MI350X (`gfx950`); CUDA and non-AMD architectures were not tested.

Raw evidence is under `evidence/`.
