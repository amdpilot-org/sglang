# Independent review of candidate PR 1979

Upstream issue: https://github.com/sgl-project/sglang/issues/33088

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1952

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2025

Candidate: https://github.com/amdpilot-org/sglang/pull/1979 at exact commit
`611cdb905edbdba7ccb82fcbad2200e514c5eb8b`.

## Recommendation

Request changes. The patch is a real, narrowly tested reduction in query rate,
but it does not fully resolve the original contract. An unreachable bootstrap is
still queried indefinitely, independently by every decode DP-rank process. The
fixed 10 ms gate has no bounded retry count, exponential backoff, terminal failure,
or node-wide coordination. Requests can therefore remain deadlocked with zero
successful completions while continuing to generate connections and repeated
errors.

The candidate's `outcome: fixed` claim is too strong. This is a partial mitigation,
not a full original-issue fix or merely test-only hardening.

## Evidence

The prepared checkout exactly matched the recorded base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The prepared interpreter imported
`sglang.srt.disaggregation.decode` and `sglang.srt.disaggregation.common.conn`
from `/job/repo/python/sglang/...`, so both base and candidate tests exercised the
checked-out source. The candidate changes only Python and tests; no native source
changed, so a native rebuild was not applicable.

On the base, the candidate's focused tests fail: the unresolved lookup is submitted
on all three scheduler cycles instead of the two allowed by the new interval, and
the throttle helper is absent. At the exact candidate commit, the complete modified
test file passes (11 tests).

An independent adversarial check then repeatedly exercised the candidate's actual
`_can_query_prefill_dp_ranks` gate. For one permanently unresolved bootstrap it
allowed 3,374 attempts per rank over 60 simulated seconds. Eight separate queue
instances, representing the independent state in eight DP-rank processes, allowed
680 aggregate attempts over one simulated second. The object has no DP-rank query
retry-cap state. Thus the candidate lowers the scheduler-cycle flood but preserves
unbounded retries and per-rank multiplication.

The recorded base already has a thread-local `requests.Session` per bootstrap
address, a one-connection HTTP adapter pool, and batched room queries. The candidate
does not change those transport details and does not prove that the reported
ephemeral-port exhaustion or zero-success serving failure is eliminated.

## Environment limitations

The assigned device was one AMD Instinct MI355X reporting
`gfx950:sramecc+:xnack-` through Torch 2.11.0+rocm7.2 / HIP 7.2.26015. GPU
inventory and source import validation succeeded, but GPU execution was not used as
evidence because the changed path is CPU-side scheduling/HTTP logic.

The Qwen3.5-397B-A17B-FP8 weights, eight GPUs per node, a second node, and the
reported TP8/EP8/DP8 MoRI topology were unavailable. Therefore neither the full
distributed model reproduction nor the exact `[Errno 99]` exhaustion could be
performed. The deterministic queue tests establish the retry behavior only; they
cannot qualify model semantics or distributed serving success.

## Raw artifacts

- `raw/base-candidate-regression.txt`: candidate regression run against the base.
- `raw/candidate-full-regression.txt`: complete modified test file at the candidate.
- `raw/candidate-adversarial.txt`: independent indefinite/per-rank retry cases.
- `raw/candidate-code.diff`: exact candidate code and regression diff.
