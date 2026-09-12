# Decode bootstrap DP-rank retry correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33088

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2112

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1979

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2076

The exact candidate commit `611cdb905edbdba7ccb82fcbad2200e514c5eb8b`
was applied to the recorded base and tested before correction. Its complete modified
test file passed (11 tests), confirming that the candidate's 10 ms throttle worked
as designed. An independent exercise of the actual candidate helper then allowed
all 100 attempts spaced 11 ms apart, exposed no retry-count or maximum-retry
state, and allowed eight initial attempts from eight independent queue instances.
This reproduces the review's boundedness and per-process-state counterexamples.

The correction preserves batching, session reuse, asynchronous prefetch, and the
candidate's per-address scheduling gate. It changes the fixed interval into
bounded exponential backoff (100 ms initial, doubling to a 2 s ceiling), limits
an unresolved address to eight attempts, aborts affected receivers at the limit,
and removes those requests from the pending lookup queue. A room arriving after
an in-flight prefetch is deferred to the next eligible batch instead of causing a
second synchronous query in the same scheduler cycle.

The focused regression verifies three failed attempts followed by receiver abort
and pending-queue removal. The full registered disaggregation unit suite passes:
361 tests and 45 subtests.

## Limitations

The retry budget is process-local. Eight decode processes can therefore each make
an initial attempt and, for a permanently unreachable bootstrap, make at most 64
attempts in total before their affected requests fail. A once-per-node query would
require a new cross-process RPC/cache owner and must also preserve distinct rooms
owned by different DP processes; no existing safe node-wide owner was found, so
this change does not claim node-wide deduplication.

Only one AMD Instinct MI355X (`gfx950`) was available. The Qwen3.5-397B-A17B-FP8
weights, a second node, eight GPUs, and the TP8/EP8/DP8 MoRI topology were not
available. The exact two-node `[Errno 99]` exhaustion and successful model serving
were not reproduced. GPU execution and a native rebuild are not relevant evidence
for this CPU-side Python scheduling change and are not claimed.
