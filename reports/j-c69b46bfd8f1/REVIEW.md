# Independent review of PR 3071 at `8c48a94e349fe8bd2d700a8a3d9d1287706dc1a5`

Recommendation: **request changes**. The candidate is a meaningful partial fix, but it does not fully resolve the original issue.

Upstream issue: https://github.com/sgl-project/sglang/issues/38075

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3005

Candidate: https://github.com/amdpilot-org/sglang/pull/3071

Earlier candidate: https://github.com/amdpilot-org/sglang/pull/2873 at `19b9f743bf6d251f666f43cc6c165a7d375b3efd`

Earlier independent review: https://github.com/amdpilot-org/sglang/pull/2970

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3099

## What is verified

- The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` was rebuilt. The candidate's live native contract probe starts the base server and then fails at the absent `GetOperationalStateRequest`; the base proto also lacks media fields.
- The exact candidate commit was checked out detached. Its 18 Rust tests pass. Its Python selection reports 4 passed and 5 skipped.
- The candidate Rust library was rebuilt with Rust 1.92.0 and explicitly loaded from `/tmp/amdpilot-repo-j-c69b46bfd8f1/candidate-cargo-target/release/libsglang_grpc_core.so`; the installed package supplied no `_grpc` extension.
- The rebuilt-native probe observes lifecycle states and confirms image, inline audio, video, and prefill/decode bootstrap fields reach the Python request dictionary.
- The candidate's standard-health and media transport work is therefore more than test-only hardening. It is functional source/native behavior.

## Blocking counterexample

`RuntimeHandle` represents weight-update activity with a single boolean. With two overlapping `UpdateWeightsFromDisk` calls, both set the boolean. When the first returns, its `finally` block clears the boolean even though the second remains in progress. The independent adversarial test records:

```text
both_running False {"phase": "UPDATING_WEIGHTS", ... "ready_to_serve": false, "weight_update_in_progress": true}
one_still_running True {"phase": "SERVING", "accepting_new_requests": true, "ready_to_serve": true, "weight_update_in_progress": false}
```

This violates the core sidecar/RL contract: the worker advertises readiness during an active weight update. A counter or serialized state transition is required, with regression coverage for overlapping calls and failure paths.

## Scope distinction

The candidate test called “aggregated and disaggregated” combines media with `DisaggregatedParams.bootstrap_*`. Those fields belong to prefill/decode KV disaggregation. SGLang's encoder-disaggregated path is separately implemented under `python/sglang/srt/disaggregation/encoder/` and exercised by dedicated EPD tests. No encoder-disaggregated multimodal server was launched here or by the candidate evidence. Thus transport compatibility is demonstrated, but the original encoder-disaggregated coverage requirement remains unverified.

No multimodal weights were available, so media preprocessing and numerical inference were not qualified. The visible GPU was one AMD Instinct MI355X with ROCm 7.2. A deterministic text-only tiny Llama fixture would only validate transport/engine execution and was not used as evidence for multimodal or distributed correctness.

Raw commands and outputs are retained beside this report. Review evidence was first stored outside the checkout and copied back only after returning to `amdpilot/j-c69b46bfd8f1`.
