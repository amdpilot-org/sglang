# Independent review of PR 2383

Candidate: https://github.com/amdpilot-org/sglang/pull/2383 at `e08baaaa5812a00905d6734eebc50133cdc6ff80`

Upstream issue: https://github.com/sgl-project/sglang/issues/31206

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2277

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2417

## Verdict

Recommendation: **accept**, specifically as test-only hardening. The candidate does not change production code and therefore is not itself a full fix for the original v0.2.4 failure. Its regression tests accurately protect behavior already present at recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

`fully_resolves_original` is false. The prepared base and candidate both fail fast when a healthy prefill worker's breaker is open, but the reporter's exact v0.2.4 router, abort-driven breaker opening, real SGLang engines, KV transfer, and distributed 1P1D environment were not reproduced here.

## Findings

- The candidate diff contains only tests and report files. There are no production, C++, HIP, Python, or other native changes to rebuild.
- Candidate production behavior is identical to the recorded base. On the base, PD selection chooses prefill before decode; `pick_worker_by_policy_arc` filters with `Worker::is_available()`, and the policy layer separately filters on health plus `circuit_breaker().can_execute()`.
- The exact candidate suite passed 3/3. It verifies 503 for an open healthy prefill breaker, sequential successful half-open recovery to Closed at `success_threshold=2`, and the symmetric open-decode case.
- Two temporary independent adversarial cases passed: the report's exact `/v1/chat/completions` endpoint returns 503 with the prefill breaker open, and a failed half-open prefill probe returns a server error, reopens the breaker, and causes the next pair selection to return 503.
- The candidate's tests use deterministic HTTP mocks. They do not recreate client cancellation or prove that the historical v0.2.4 failure path no longer exists in that release. The candidate correctly labels its outcome `not_reproduced`, but its mutation is only a sensitivity check, not a reproduction of the original base behavior.

## Environment and architecture

The host exposes one `gfx950` / AMD Instinct MI350X agent with ROCm 7.2 and Torch 2.11.0+rocm7.2. GPU execution is not relevant to this Rust HTTP control-plane test and was not used. The prescribed interpreter imports `sglang` from `/job/repo/python/sglang/__init__.py`; the tested gateway was compiled from `/job/repo/sgl-model-gateway` with Rust 1.90 into a private target directory. No native library or wheel was changed, imported, or rebuilt.

## Retained evidence

- `exact-candidate-tests.txt`: exact candidate, 3 passed.
- `adversarial-tests.txt`: candidate plus two temporary review-only cases, 5 passed after correcting the review assertion to accept the implementation's status-faithful 500 response.
- `fmt-check.txt`: formatting check passed; stable rustfmt emitted existing warnings for nightly-only options.
- `environment-and-paths.txt`: source import path, machine, and gfx950 evidence.
- `candidate-files.txt` and `candidate-stat.txt`: exact diff scope.

The temporary adversarial tests were removed before returning to the prepared review branch. No candidate source was modified.
