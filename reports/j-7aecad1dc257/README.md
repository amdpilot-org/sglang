# Independent review of PR 1265 at `8b21de7`

Upstream issue: https://github.com/sgl-project/sglang/issues/36475

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1300

Candidate: https://github.com/amdpilot-org/sglang/pull/1265 at exact commit
`8b21de71aa771d3a43d8b440937b4b551d4464d3`.

Recommendation: **accept**, with the exact Qwen/CUDA scheduler-crash symptom still
unverified in this ROCm environment.

The source change is narrow and addresses the reproduced defect. A rejected
streaming-session overlap never acquired the session-wide `_inflight` slot, but
the base cleanup path released that slot unconditionally. The candidate records
ownership on admitted requests and makes abort cleanup conditional on ownership.
This prevents a rejected request from admitting a third concurrent turn and
mutating the disconnected owner's shared token arrays.

On the recorded base, the candidate's focused tests produced 3 failures and 13
passes. At the exact candidate commit, all 16 passed. Three independent tests
also passed for delayed rejected-request cleanup after a later owner is admitted,
duplicate abort of an old owner, and reverse-order cleanup of eight overlaps.

A source-checkout server was exercised on the assigned AMD Instinct MI355X
(`gfx950`) with the qualified deterministic tiny Llama fixture. The candidate's
immediate retry took the issue's explicitly permitted visible-failure branch:
`BadRequestError: Streaming session already has an active request.` The next
retry rolled back to the correct 72 prompt tokens and `/health` remained 200.
The checked-in runner returned 1 because its predicate requires both immediate
retries to be busy; that assertion rejects the valid busy-then-clean-rollback
sequence and is a harness false negative.

The independent base HTTP attempt did not hit the timing window, though the
lower-level defect failed deterministically and the candidate's retained base
artifact records the one-token overlap response. The tiny fixture did not
reproduce the reported pool-memory invariant crash. Qwen2.5 weights and the
reporter's NVIDIA architecture were unavailable, so this report does not claim
an exact model/architecture reproduction. No native source changed; imports
were confirmed from `/job/repo/python/sglang`, and no native rebuild was needed.

Raw review output is retained under `/job/review-j-7aecad1dc257/evidence`.
Structured commands, results, limitations, recommendation, and remaining
counterexamples are in `result.json`.
