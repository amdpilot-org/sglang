# Independent review of PR 1913

Upstream issue: https://github.com/sgl-project/sglang/issues/33483

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1951

Candidate: https://github.com/amdpilot-org/sglang/pull/1913 at
`a3ff51e96cf96008a8382e5e18ec0af420519a17`.

## Verdict

Request changes. The source change is a working partial diagnostic improvement:
it emits a one-time startup warning when effective per-worker request capacity
exceeds enabled decode-graph coverage. It does not reconcile the independent
defaults, cap admission, enlarge graph capture, or prevent the eager-decode
absorbing state. For the issue values, the candidate still resolves
`max_running_requests` to 4096 while the graph ceiling remains 32.

The submitted regression also does not provide the claimed behavioral
failing-before result. Run unchanged at the recorded base, all three tests fail
during helper setup with `AttributeError` because they directly reference the
new `_warn_cuda_graph_request_capacity` method. Thus the equality and
disabled-graph tests do not pass on base, contrary to the candidate report's
"1 failed, 2 passed" claim. The test should exercise the public resolution path
without requiring the new private method to exist, so only the missing-warning
expectation fails before the change.

## What was independently verified

- The prepared checkout exactly matched recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image-prepared
  revision difference.
- The prepared interpreter imported `sglang` and
  `kv_cache_configurator.py` from `/job/repo/python/sglang/...` on both base and
  candidate.
- The candidate's focused and adjacent suites passed: 44 tests and 7 subtests.
- Independent mocked boundaries passed for issue defaults, equality, one above
  the ceiling, DP-per-worker scaling, KV-cap reduction, disabled graphs, absent
  graph configuration, and warn-once behavior.
- A source-checkout server using the qualified deterministic tiny Llama fixture
  on the assigned AMD MI350X/gfx950 emitted the warning for request capacity 4
  versus captured maximum 2, captured decode graph batches `[1, 2]`, and
  returned HTTP 200 for four transport probes.

## Evidence and limitations

Raw commands and outputs are under `raw/`. The candidate commit's own README
references `raw/gpu-graph-final/server.log`, but that file is absent from the
commit; this review therefore reran the qualified fixture and retained its log.

The available device is AMD MI350X/gfx950 with Torch 2.11.0+rocm7.2, not the
reported NVIDIA L40. Qwen2.5-0.5B weights and the ShareGPT workload were not
available. The tiny random Llama fixture qualifies startup integration, graph
capture, engine execution, and HTTP transport only; it does not reproduce the
reported 4x latency step, its absorbing dynamics, Qwen behavior, or distributed
execution. No native source changed, so no native rebuild was required.
