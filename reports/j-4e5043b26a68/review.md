# Independent review of PR 1565

Upstream issue: https://github.com/sgl-project/sglang/issues/34737

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1603

Candidate: https://github.com/amdpilot-org/sglang/pull/1565 at `1065fe89c32aa4918bac5d64a38c54a1f931076b`

## Recommendation

Request changes. The candidate is a real partial production fix, and its focused regression fails on the recorded base and passes on the exact candidate. However, it does not fully satisfy the original cleanup contract.

`CommonKVReceiver._setup_bootstrap_infos()` builds a receiver's `bootstrap_infos` by concatenating cached groups for every target CP rank. `register_wm_subscriber()` keys the registry by that complete concatenated tuple. The candidate's `_handle_node_failure()` instead passes each `connection_pool` group separately to `snapshot_wm_subscribers()`. For a multi-CP receiver, none of those keys equals the concatenated registry key.

The independent adversarial case constructs two cached CP groups for the failed address and registers the same combined receiver shape produced by `_setup_bootstrap_infos()`. On the exact candidate it records:

```text
multi_cp_remaining 1
stale_broadcast_calls 1
single_cp_remaining 0
```

Thus failure cleanup works for the candidate's single-group fixture but still leaves a stale subscriber and a later stale watermark broadcast for a valid multi-group receiver.

## Evidence

- Base `358c163250ad3b1f62939b01ce1314a0a31a0365`: candidate lifecycle tests produced 4 failures, including the original stale entry and receiver-refresh failures.
- Exact candidate: focused lifecycle suite passed 4 tests.
- Exact candidate: relevant NIXL and wire suites passed 90 tests and 14 subtests.
- Exact candidate: imports resolved to the changed source under `/job/repo/python/sglang/...`, not an installed copy.
- The candidate changes only Python source and tests. No native rebuild applies.

Raw commands and outputs are retained under `reports/j-4e5043b26a68/raw/`; structured claims are in `result.json`.

## Architecture limitations

The environment exposed one AMD Instinct MI350X with Torch 2.11.0+rocm7.2. The original heterogeneous-TP topology needs at least three GPUs, so this review does not claim a full serving, model, or distributed node-failure reproduction. The deterministic tests exercise the actual shared staging lifecycle implementation used by NIXL and Mooncake. No GPU execution, model weights, semantic validation, or native compilation was involved.
