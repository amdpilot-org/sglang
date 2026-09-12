# Independent review of PR 2245

Upstream issue: https://github.com/sgl-project/sglang/issues/31766

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2206

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2281

Candidate: https://github.com/amdpilot-org/sglang/pull/2245 at
`368fafdf463b9f208433d3254d7f3e1b42180a74`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a substantive partial fix: its LRU cache and
paired socket/monitor cleanup bound descriptor growth when the configured cache
size is safely below the process descriptor budget. It is not merely test
hardening. However, the default cache size is a fixed 1024 and is not constrained
by `RLIMIT_NOFILE` or an available-FD budget. With the candidate unchanged and a
soft descriptor limit of 128, the real `CommonKVManager._connect` path still
raises `zmq.error.ZMQError: Too many open files` after 32 endpoints, before LRU
eviction begins. This is the original failure class.

The same adversarial process survives 1,000 unique endpoints when the new cache
setting is manually reduced to 8. Thus the implementation works when correctly
tuned, and on the reporter's stated 1,048,576-FD limit its default 1024-entry cap
would leave ample headroom. The candidate nevertheless does not fully satisfy
the general contract that traffic-driven endpoint growth must not exhaust the
process FD limit without operator tuning.

## Evidence

On the recorded base, a real PyZMQ probe through the checked-out
`CommonKVManager._connect` created 200 unique endpoints and retained 200 PUSH
sockets plus 200 monitor sockets. `/proc/self/fd` grew from 9 to 613 (peak 614).

At the exact candidate commit, imports resolved to
`/job/repo/python/sglang/srt/disaggregation/common/conn.py`. The same probe with
a cache size of 8 ended with 8 PUSH and 8 monitor entries; descriptors grew from
9 to 45 (peak 48). The candidate's five focused regressions passed, as did the
complete registered disaggregation unit directory and environment tests (376
tests and 47 subtests).

The independent low-limit boundary used a fresh subprocess, set only its own
soft `RLIMIT_NOFILE` to 128, and exercised the real `_connect` implementation:

- Candidate default (1024): 32 endpoints, then `ZMQError: Too many open files`.
- Explicit cache size 8: 1,000 endpoints, no error, 8 cached endpoints and 38
  open descriptors at completion.

Raw outputs and the standalone probes are preserved outside the checkout in
`/job/review-evidence-j-7a23ab1ce074/`.

## Scope and limitations

No native source changed, `repository-environment.json` has `native: null`, and
no native rebuild was applicable. The assigned device was one AMD Instinct
MI350X (`gfx950`, Torch 2.11.0+rocm7.2); no GPU kernel was relevant or executed
for this host-side PyZMQ lifecycle issue. The reporter's 16x NVIDIA H800 model,
weights, and distributed serving workload were unavailable. This review proves
the descriptor lifecycle behavior directly but does not claim a full-model or
multi-node reproduction.
