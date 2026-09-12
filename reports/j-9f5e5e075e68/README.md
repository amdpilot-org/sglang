# Independent review of PR 843 at `f0cb5a958e600a68ea16a35b98b877f781e2046a`

Upstream issue: https://github.com/sgl-project/sglang/issues/38167

Mirror issue: https://github.com/amdpilot-org/sglang/issues/885

Recommendation: **request changes**. The candidate is a useful partial hardening, but it does not fully resolve the original service-availability contract.

The behavioral probe in `raw/base-event-loop.txt` shows the recorded base returning the reported `device not ready` error and then dispatching a second request. On the exact candidate, `raw/candidate-event-loop.txt` shows the first request receives an explicit restart-required error and the monolithic scheduler loop stops before a second dispatch. The candidate's focused tests also pass (`19 passed`).

The remaining counterexample is at the serving boundary. `run_scheduler_process` treats the stopped event loop as a normal return and exits the worker without notifying or terminating the HTTP process. The HTTP scheduler client defaults to no receive timeout; `raw/default-timeout-evidence.txt` records `_resolve_timeout_ms(...) == None` and ZMQ `RCVTIMEO == -1`. Thus a request submitted after the fatal request can wait indefinitely on the now-absent scheduler. This avoids executing more GPU work on poisoned state, but it does not make the server fail fast or provide the later request a clear restart-required signal.

The candidate also does not establish that the first `device not ready` error is memory exhaustion or provide an admission-time clean insufficient-memory response. That diagnosis cannot be qualified here: the assigned device is an AMD Instinct MI350X with 258 GB under ROCm 7.2, rather than the reported RTX 5070 12 GB under WSL2/CUDA 13.0, and MiniMax-H3 weights plus the reporter's LoRA are unavailable.

No native files changed, so no native rebuild was applicable. Import evidence confirms the candidate scheduler was loaded from `/job/repo/python/...`, not from an installed wheel. The candidate PR body also cites mirror issue 794, while the reviewed task's mirror is issue 885.
