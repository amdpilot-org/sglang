# Consolidated correction for prefill ZMQ descriptor exhaustion

Upstream issue: https://github.com/sgl-project/sglang/issues/31766

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2540

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2429 at exact commit
`52a17bf276b24ea1f62f93c2c43f4ffca182c217`.

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2522.

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Reproduction before correction

The prepared checkout was detached at the exact candidate commit and imported
`CommonKVManager` from `/job/repo/python`. With soft `RLIMIT_NOFILE=128`, its
unset default produced a cache size of 21. A real-PyZMQ probe connected distinct
endpoints and acquired each endpoint's send lock before connecting the next.
Because `_evict_idle_socket()` skipped every entry, `_connect()` followed the
candidate's documented overflow path. It completed 38 endpoints, grew the cache
to 38 entries, and failed with `ZMQError: Too many open files`. The traceback
included the real PUSH/monitor socket creation path. This independently
confirmed the concrete review counterexample rather than relying on its report.

The exact candidate's own focused suite also produced 7 passes and one failure
on this prepared host: its 1,000-endpoint low-RLIMIT subprocess reached
`sock.get_monitor_socket()` and raised the same error. Independent sequential
probing showed why: libzmq retires descriptors asynchronously, so rapid churn
could have only 20 cached entries but 125 open descriptors before monitor
creation failed. This was retained as separate evidence and did not replace the
active-sender counterexample.

Raw pre-correction outputs are under
`/job/evidence-j-69e29e9c7bf8/candidate-busy.log` and
`candidate-tests.log`.

## Correction

The candidate's valid fixes are preserved: bounded LRU ordering, paired
PUSH/monitor cleanup, per-endpoint send locks, protection against closing an
active socket, atomic monitor creation, and an fd-aware unset default.

The overflow behavior is replaced with condition-variable backpressure. When
all entries are active, a new distinct endpoint releases the global cache lock
while waiting; existing sends and cached lookups can finish, and the waiting
connection evicts an idle entry before creating another socket. The cache can
therefore never exceed its configured bound.

Socket creation now retries `EMFILE` for at most one second with a 10 ms
throttle. This covers the independently observed libzmq asynchronous-close
window without masking persistent process-wide descriptor exhaustion or
turning unrelated exhaustion into an unbounded wait.

The real-PyZMQ regression fills the derived low-RLIMIT cache with locked
endpoints, verifies the next connection remains blocked with the cache still at
the hard bound and the process below 128 descriptors, releases one sender, and
verifies the connection then completes while the cache remains bounded.

## Validation

The corrected focused suite passed 9 tests. The broader disaggregation and
environment suite passed 380 tests and 47 subtests. Pre-commit passed every
applicable hook for the changed source, environment definition, test, and
report files.

## Limitations

The reporter's 16x NVIDIA H800 full-model, multi-node workload and weights were
unavailable. The concrete defect is host-side `CommonKVManager`/PyZMQ descriptor
management, so no GPU kernel was executed and no claim is made about the full
distributed workload. No native source changed, and the prepared environment
records no native build, so no native rebuild applied. An explicitly oversized
`SGLANG_DISAGGREGATION_ZMQ_SOCKET_CACHE_SIZE` can still exceed the process's
descriptor budget; explicit configuration remains an intentional override.
