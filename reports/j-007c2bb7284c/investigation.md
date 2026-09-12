# Independent review of the fd-aware ZMQ endpoint cache

Upstream issue: https://github.com/sgl-project/sglang/issues/31766

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2467

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2429 at exact commit
`52a17bf276b24ea1f62f93c2c43f4ffca182c217`.

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Request changes. The candidate fixes sequential endpoint churn under a low fd
limit, but it does not fully resolve the original traffic-spike failure. Its
active-sender protection deliberately permits the cache to exceed its derived
limit whenever all cached endpoints are busy. Enough simultaneously busy,
distinct endpoints therefore reach the same real PyZMQ monitor-creation
`ZMQError: Too many open files` reported in the original issue.

## Evidence

The prepared base was tested first with the repository implementation imported
from `/job/repo/python`. With soft `RLIMIT_NOFILE=128`, real
`CommonKVManager._connect` failed after 38 completed unique endpoints while
creating the next monitor socket. This reproduced the original host-side
failure without model weights or a mocked ZMQ context.

The checkout was then detached at the exact candidate commit. Runtime source
inspection confirmed that both `sglang.srt.environ` and `CommonKVManager` were
loaded from `/job/repo/python`, not from an installed SGLang wheel. The
candidate's regression suite passed all eight tests. Its sequential low-limit
scenario also passed independently: the unset default derived a cache limit of
21 and completed 1,000 unique endpoints, retaining 21 entries with 76 open fds.

An adversarial case tied directly to the candidate's concurrency policy held
each endpoint's send lock, representing distinct endpoints with active sends,
before opening the next endpoint. `_evict_idle_socket()` could not evict any
entry, so `_connect()` followed its documented temporary-overflow path. With
the same soft limit of 128, real PyZMQ failed after 38 completed endpoints at
`sock.get_monitor_socket(zmq.EVENT_DISCONNECTED)` with `Too many open files`.
The cache had grown to 38 despite its derived limit of 21. This is not an
unrelated smoke: it exercises the same `CommonKVManager._connect` call and
monitor creation named by the original traceback, under the traffic-spike
condition described by the report.

The candidate is therefore a partial source fix with useful regression
hardening, not a full original-issue fix. It bounds idle/sequential churn and
closes paired sockets correctly, but it lacks a hard descriptor-safe policy
when all cache entries are active.

Raw logs and the standalone probe are retained outside the checkout under
`/job/review-evidence-j-007c2bb7284c/` so revision switching did not overwrite
the evidence.

## Architecture and environment limitations

The reporter's 16x NVIDIA H800, full model, multi-node serving workload, and
weights were unavailable. The prepared host is x86_64 with PyTorch
`2.11.0+rocm7.2`, HIP 7.2, and PyZMQ 27.2.0. No GPU kernel participates in the
fd lifecycle tested here, so the assigned single AMD GPU was not executed and
no claim is made about the full distributed workload. The concrete failure was
reproduced through the actual host-side manager and real PyZMQ sockets.

No native or C++ paths changed (`repository-environment.json` also records
`native: null`), so no native rebuild was applicable. The exact candidate
changed only Python, tests, and report files.
