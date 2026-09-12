# Independent review of amdpilot-org/sglang PR 1799

- Upstream issue: https://github.com/sgl-project/sglang/issues/33698
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1837
- Candidate: https://github.com/amdpilot-org/sglang/pull/1799 at exact commit `e0bd61c3b3f11e3edfb1c5a74b956c8f62a62020`
- Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**
- Fully resolves original issue: **true**

## Finding

The base implementation has one shared `model_update_result`, `model_update_tmp`, and `model_update_expected_workers` state tuple. Paused disk updates bypass `model_update_lock.writer_lock`, so overlapping callers replace this state and an untagged scheduler response can complete the wrong caller. The candidate adds a dedicated operation lock around dispatch, response aggregation, and successful-result finalization. It also shields and drains the dispatched operation before propagating cancellation. Unpaused operations acquire the existing writer lock before the operation lock, retaining writer preference.

The original issue was reproduced on the exact recorded base with the candidate's unchanged scheduler-boundary regression: 5 tests failed and 3 passed. The defining failure was `the first update lost its completion`; cancellation also allowed a successor to dispatch before the first untagged response was drained. On the exact candidate, all 8 focused tests passed.

Independent adversarial tests also passed for:

- three overlapping paused updates receiving FIFO, operation-owned responses;
- cancellation of a caller queued behind an active update without dispatch or response theft;
- draining a failed response for an active cancelled caller before dispatching its successor;
- two-worker aggregation across two concurrent operations without cross-operation responses.

The candidate therefore constitutes a full original-contract fix, not merely test hardening. No remaining counterexample was found within the scheduler-boundary contract.

## Source and native path verification

Both the base and exact candidate imported `sglang` from `/job/repo/python/sglang/__init__.py` and `tokenizer_manager` from `/job/repo/python/sglang/srt/managers/tokenizer_manager.py`. The candidate changes Python control-plane source only; it changes no C++, HIP, CUDA, FlyDSL, or other native source. A native rebuild is therefore not applicable.

## Environment and limitations

The prepared interpreter was `/tmp/amdpilot-repo-j-275f05e568eb/venv/bin/python`, with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. One AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) was visible. No GPU kernels were executed because scheduler completion ownership and cancellation are CPU-side asyncio behavior; an unrelated GPU smoke would not strengthen this conclusion.

No model weights, HTTP server, full model reload, semantic-accuracy workload, or multi-node workload was run. The scheduler-boundary fixture directly exercises the defect but does not qualify those broader paths. Multi-worker validation used deterministic manager-level aggregation with two simulated worker responses.

`git diff --check` over the entire candidate commit reports trailing whitespace in committed raw report logs. The changed source and test behavior remain valid, but PR 1799's prose claiming the whole patch passed `git diff --check` is not literally reproducible. This is non-functional report-artifact hygiene and does not leave a counterexample to the original contract.

## Evidence

Raw commands and outputs are retained under `reports/j-275f05e568eb/raw/`. The independent test source is `raw/independent_adversarial_test.py`.
