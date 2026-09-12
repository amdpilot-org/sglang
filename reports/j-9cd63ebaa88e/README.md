# Independent review of PR 3365

Reviewed `https://github.com/amdpilot-org/sglang/pull/3365` at exact commit
`3fa63e29ff7226e4097aaea27ec426a1eb52a85e` against upstream issue
`https://github.com/sgl-project/sglang/issues/28157`.

## Recommendation

Accept. The candidate isolates synchronous Prometheus multiprocess collection in
a child process, bounds it with an eight-second timeout, and rejects overlapping
scrapes without queueing. The recorded base was shown to delay the event loop by
0.331 seconds when collection took 0.35 seconds; the exact candidate kept event
loop lag to 0.001 seconds and returned 503 to an overlapping scrape. The earlier
candidate's repeated `Accept` and `Accept-Encoding` failures were independently
reproduced at `a5b60944c1762c4b9196fec907c2a72a58e9143b`, and both pass at the
reviewed commit. No remaining source-level counterexample was found.

## Evidence

- Base `358c163250ad3b1f62939b01ce1314a0a31a0365`: a deterministic slow collector
  returned 200 while delaying an event-loop timer by 0.331 seconds.
- Parent candidate `a5b60944c1762c4b9196fec907c2a72a58e9143b`: duplicate `Accept` selected
  Prometheus text instead of OpenMetrics, and duplicate `Accept-Encoding` did
  not select gzip; the probe exited 1.
- Reviewed candidate `3fa63e29ff7226e4097aaea27ec426a1eb52a85e`: five focused unit tests passed;
  duplicate-header, filtering, gzip, OpenMetrics, timeout/cancellation, and
  singleflight cases passed.
- Independent weighted and repeated header cases matched `make_asgi_app`, and
  both `/metrics` and `/metrics/` returned the metric.
- The loaded source was `/job/repo/python/sglang/srt/utils/common.py`.

Raw command output and standalone probes were preserved outside the revision-
switching checkout under `/job/review-evidence-j-9cd63ebaa88e/`.

## Limitations

The reported eight-H100, multi-node, single-tokenizer PD deployment, model
weights, sustained production scrape pressure, and decode-side connection-loss
logs were unavailable. The host instead exposes ROCm 7.2 and gfx950-class AMD
hardware. GPU execution would not strengthen this CPU/ASGI/Prometheus contract,
so no GPU was used. No native source changed and the prepared environment has no
task native build target, so no native rebuild was applicable. The conclusion
is based on direct deterministic reproduction of the blocking mechanism and
candidate behavior, not a full production-topology replay.
