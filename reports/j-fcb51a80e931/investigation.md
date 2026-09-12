# RLIMIT-aware ZMQ endpoint cache correction

Upstream issue: https://github.com/sgl-project/sglang/issues/31766

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2362

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2245 at
`368fafdf463b9f208433d3254d7f3e1b42180a74`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2328

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding and correction

The independent review's counterexample reproduced against the exact candidate.
A subprocess with its own soft `RLIMIT_NOFILE` reduced to 128 and the candidate's
unset/default cache setting failed in real `CommonKVManager._connect` while
creating the monitor PAIR socket. The exception was
`zmq.error.ZMQError: Too many open files`, before the fixed 1024-entry cache
limit could trigger eviction.

The candidate's valid LRU, paired PUSH/monitor cleanup, send locking, and atomic
monitor creation fixes are preserved. The correction changes only the unset
cache default: it is capped using the process's soft `RLIMIT_NOFILE`, current
`/proc/self/fd` occupancy, a 32-descriptor reserve, and a conservative budget of
four descriptors per cached endpoint. An explicit
`SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE` remains an operator override.

The same subprocess regression completes 1,000 unique endpoints after the
change, deriving a limit of 21 in the imported test process, retaining 21
endpoints, and finishing with 76 open FDs. Deterministic boundary tests verify
that 8 already-open descriptors
and a limit of 128 derive a cache size of 22, while the reporter's high
1,048,576 limit retains the existing maximum default of 1024.

## Validation

- Before: focused low-limit regression failed on candidate commit
  `368fafdf463b9f208433d3254d7f3e1b42180a74` at
  `sock.get_monitor_socket(...)` with `Too many open files`.
- After: `python -m pytest -q
  test/registered/unit/disaggregation/test_common_kv_manager_socket_cache.py`
  passed 8 tests.
- After: `python -m pytest -q test/registered/unit/disaggregation
  test/registered/unit/test_environ.py` passed 379 tests and 47 subtests.
- Pre-commit checks passed after formatter output was applied.

Raw test output is retained in `/job/evidence-j-fcb51a80e931/`.

## Limitations

The reporter's 16x NVIDIA H800 full-model and multi-node workload was not
available. No model weights or alternate architecture were used as evidence for
this source correction. This is a host-side PyZMQ descriptor-lifecycle defect,
so the assigned single `gfx950` GPU was not executed. No native source changed
and no native rebuild was applicable. The derived default is sampled when the
manager is initialized; an explicitly oversized operator override can still
exceed the process's descriptor budget.
