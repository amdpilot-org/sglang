# Investigation report: gateway unknown-model registration under IGW

Upstream issue: https://github.com/sgl-project/sglang/issues/37554

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2679

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

Outcome: `environment_blocked`.

The checked-out implementation still contains every source-level link in the reported failure chain:

1. `src/core/steps/worker/local/mod.rs` declares `discover_metadata` with `FailureAction::ContinueNextStep`.
2. `src/core/steps/worker/local/create_worker.rs` selects `config.model_id`, then discovered `served_model_name`, then discovered `model_path`, and finally `UNKNOWN_MODEL_ID`; it does not distinguish IGW mode.
3. `src/service_discovery.rs::handle_pod_event` inserts a healthy pod into `tracked_pods` before submitting `AddWorker`. Later applied events are treated as duplicates. The pod is removed from that set only when queue submission itself fails, not when the asynchronous registration workflow fails.
4. The HTTP and PD routers use model-indexed worker selection under IGW, so an `unknown` worker is not a candidate for a named request.
5. `main.rs` automatically enables IGW when service discovery is enabled.

## Related fixes checked

Upstream draft PR https://github.com/sgl-project/sglang/pull/37555 changes the real `CreateLocalWorkerStep` to reject a worker with no model identity under IGW and adds two mode-boundary unit tests. Its issue discussion later corrects the PR's original recovery assumption: on current main, rejecting the worker alone leaves the pod tracked and unregistered.

Upstream PR https://github.com/sgl-project/sglang/pull/32322 adds periodic full-LIST reconciliation and logic to forget/resubmit a tracked pod that is absent from the worker registry while no `AddWorker` is in flight. Its exact head commit, `a3d51c0b0442762f302462a98d1f411b62c053d0`, is not an ancestor of this checkout, and the corresponding resync symbols are absent. Both PRs were open when inspected.

Therefore applying #37555 alone here would replace a permanently registered unroutable worker with a permanently tracked, unregistered pod. That is observably different but does not repair fleet convergence, so it was not committed as a fix.

## Execution blocker

The prepared environment documents only the Python interpreter. `cargo` is not on `PATH`, no executable named `cargo` was found under `/opt`, `/usr/local`, `/root`, `/tmp`, or `/job`, and no prebuilt `sgl-model-gateway`/`smg` executable was found under the job or private runtime directories.

A temporary regression invoking the real `CreateLocalWorkerStep` was added locally and the focused command was attempted with cargo caches and target output under `/tmp/amdpilot-repo-j-1d91e012c283`. The command exited 127 with `cargo: command not found`; the temporary source edit was then removed. Consequently this report does not claim an actual runtime reproduction, a passing candidate, or a native rebuild.

## Required follow-up

Re-run in an image containing the pinned Rust toolchain (or a prepared gateway executable). Validate the full pair of behaviors together:

- IGW refuses registration when all identity sources are absent, while non-IGW retains the `UNKNOWN_MODEL_ID` compatibility fallback and explicit `config.model_id` remains accepted.
- After the initial IGW refusal, service discovery actually resubmits the same still-live pod after metadata becomes available, without accumulating duplicate in-flight `AddWorker` jobs.

The second assertion is essential; a startup smoke or the rejection test alone does not qualify the original issue as fixed.
