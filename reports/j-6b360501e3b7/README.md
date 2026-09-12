# Independent review of PR 2577

Upstream issue: https://github.com/sgl-project/sglang/issues/3050

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2580

Candidate: https://github.com/amdpilot-org/sglang/pull/2577 at
`3516408f3178b64300fb78bc28064bed9c653931`.

Recommendation: **request changes**. The candidate correctly warns that a
per-request TPOT cannot be multiplied by configured maximum concurrency and
compared directly with a recent engine reporting window. The real gfx950 run
reproduced that mismatch on the recorded base and on the candidate. However,
the patch is documentation/test-only hardening: it changes no metric or engine
behavior, and its central statement that “TPOT/ITL” cover the “whole request
lifetime” is false for the implementation it documents.

`calculate_metrics` computes TPOT as `(latency - ttft) / (output_len - 1)`, so
TPOT expressly excludes TTFT rather than covering the whole request lifetime.
ITL is collected per received stream event; it can omit the terminal response
tail and one event may carry multiple tokens. The candidate tests only assert
the presence of selected phrases, so they pass without checking either metric
definition. An independent synthetic case records 100 ms TPOT for a request
whose whole-request-per-output-token value is 490 ms, and records 0.8 seconds
of ITL for a 0.9-second post-TTFT interval.

The candidate's direction to use whole-run output throughput is useful, but it
is a scope-aligned diagnostic rather than an exact equality guarantee: engine
generation throughput is calculated from tokens accumulated in a recent
decode log interval, while client output throughput divides all successful
output tokens by end-to-end benchmark duration. Ramp-up, prefill, drain, and
token-boundary differences remain.

No native source changed (`python/sglang/benchmark/serving.py`, tests, and
reports only), so no native rebuild was applicable. Imports resolved to the
prepared checkout and prepared ROCm interpreter. The GPU fixture validates the
HTTP/engine metric plumbing only; it does not qualify Qwen2.5-0.5B, NVIDIA
H800/CUDA, 2048-token prompts, 256-token outputs, or concurrency 16.

Raw outputs are retained under `evidence/`. The base run measured 474.51 tok/s
whole-run output throughput and steady engine windows of 1954.67–1987.23
tok/s. The candidate run measured 1479.77 tok/s and steady engine windows of
1823.38–1867.75 tok/s. The difference between runs is itself evidence that the
tiny fixture is a transport/accounting check, not a stable performance
benchmark.
