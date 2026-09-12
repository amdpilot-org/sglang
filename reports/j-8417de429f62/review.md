# Independent review of PR 2811 at `6defd730`

Upstream issue: https://github.com/sgl-project/sglang/issues/32728

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2740

Review deliverable issue: https://github.com/amdpilot-org/sglang/issues/2842

Recommendation: **accept**. The candidate fully resolves the original requested contract within the replay buffer's documented retention limit.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` was also the image-prepared checkout, so there was no base drift. A temporary review-only native test sent sequence 1 followed by sequence 3 through the real base Rust ZMQ subscriber. It received both as `(1, 3)`, reproducing the stale-view failure: the base advanced past missing sequence 2 and had no replay endpoint or client.

The exact candidate commit was then checked out detached. Its native gap regression repaired the same live gap through ROUTER/DEALER replay and delivered `(1, 2, 3)`. Separate tests accepted the complete Python-compatible multipart replay sequence and rejected a truncated replay ring. An independent temporary adversarial test supplied replay order `(2, 4, 3)` for requested range `[2,5)`; the candidate rejected it at `expected seq 3, got 4`, so it did not partially apply non-contiguous state.

Source review confirmed the implementation covers all requested pieces: optional replay endpoint advertisement in `/server_info`, wildcard-host resolution and per-DP-rank port selection in both introspection paths, a bounded DEALER client using the existing Python format, per-worker gap detection and ordered repair, conservative withholding on replay failure, backward compatibility when replay metadata is absent, and replay counters in the metrics response.

The candidate changes native Rust code. Rust 1.90.0 was installed in `/tmp/amdpilot-repo-j-8417de429f62`, and a fresh x86-64 `sgl-router` binary was built under that private target directory. All 548 Rust library tests, six selected KV component tests, 29 `/server_info` tests plus 12 subtests, and three descriptor contract tests passed.

No GPU run was performed because this feature has no GPU numerical path; the relevant proof is native socket execution and rebuilding the Rust router. No production serving process or deliberately saturated HWM workload was run. The deterministic native tests inject the identical sequence-gap condition. If the replay ring has already evicted the earliest missing sequence, recovery is inherently impossible; the candidate exposes that limitation honestly by rejecting the suffix and withholding the newer live event.

Raw logs and fetched issue/PR metadata are retained outside the checkout at `/job/review-evidence-j-8417de429f62/`.
