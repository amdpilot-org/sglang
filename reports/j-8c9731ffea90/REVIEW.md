# Independent review of PR 2899 at `4caa276`

Upstream issue: https://github.com/sgl-project/sglang/issues/22889

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2932

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2849

Candidate: https://github.com/amdpilot-org/sglang/pull/2899 at exact commit `4caa276b9291e874d88cfae66295ec3cac2e1a65`.

## Verdict

Recommend accepting the candidate as a verified incremental synchronization fix. It does **not** fully resolve the original Python 3.14t feature request and must not close or be described as completing that issue.

The candidate fixes a concrete race in `DecodeStagingHandler`: on the recorded base, teardown can remove and free a staging allocation while scatter is paused after launch but before its completion event is published. An independent forced interleaving failed on base `358c163` and passed at the exact candidate commit. The submitted focused selection and full affected unit file also pass.

## Classification

- Full original-issue fix: no.
- Partial fix: yes; one Phase 4 shared-state lifecycle is serialized.
- Test-only hardening: no; production code adds a handler-owned `RLock` around registration, arrivals, scatter publication, completion polling, and teardown.
- Unverified claim: cp314t support itself remains unverified. The candidate report correctly disclaims complete support.

No native source changed. The imported candidate module resolved directly to `/job/repo/python/sglang/srt/disaggregation/common/staging_handler.py`, so the tested Python source was not taken from an installed wheel. A native rebuild was therefore not applicable.

## Environment and architecture limits

The mandated interpreter is CPython 3.12.3 with SOABI `cpython-312-x86_64-linux-gnu` and `Py_GIL_DISABLED` unset. No Python 3.14/3.14t executable was found. Torch is `2.11.0+rocm7.2`, HIP is 7.2, and one AMD GPU is visible. The original contract also targets CUDA, which this node cannot qualify.

No GPU numerical test is claimed: the change is host synchronization and the regression deliberately mocks the scatter/event boundary. A serving smoke would not prove free-threaded correctness or a distributed staging workload, so none was substituted for the missing cp314t/CUDA coverage.

## Evidence summary

- Base independent regression: failed as expected because teardown completed inside the launch/event-publication window.
- Candidate independent regression: passed; teardown waited and allocation 17 was freed exactly once.
- Candidate focused tests: 4 passed, 35 deselected.
- Candidate full affected file: 39 passed, 12 subtests passed.
- Candidate compile check: passed.
- Candidate diff whitespace check: source was clean, but the committed candidate test log contains one trailing-whitespace line.

Raw logs, the independent regression, source snapshots, the current upstream issue snapshot, and environment probes are retained at `/job/review-evidence-j-8c9731ffea90/` outside the checkout so revision switches did not overwrite them.

## Still open from the original contract

The candidate provides no clean cp314t installation/import result, dependency resolution, sglang-kernel free-threaded build, native-extension audit, actual free-threaded distributed staging run, CI/wheels, or performance evidence. Most Python shared-state areas named by the RFC also remain untouched. The exact remaining counterexamples and commands are recorded in `result.json`.
