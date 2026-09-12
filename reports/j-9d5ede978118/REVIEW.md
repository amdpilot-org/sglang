# Independent review of PR 873

Upstream issue: https://github.com/sgl-project/sglang/issues/38099

Mirror issue: https://github.com/amdpilot-org/sglang/issues/903

Candidate: https://github.com/amdpilot-org/sglang/pull/873 at `8073bd76d62c50adcf3650ee5bc5dab3e641994a`

## Recommendation

Accept. The candidate fully resolves the two reproducible send-worker defects stated by the original issue: reuse of a shared staged source before asynchronous completion, and replay of already-completed destinations following staging deferral. This conclusion does not extend to the issue's explicitly unproven hypothesis that these defects caused the production text corruption.

## Evidence

The prepared checkout was clean and exactly at the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image/base discrepancy. I preserved the candidate regression outside the checkout and ran the same file on both revisions.

On the base, all three focused tests failed. The observed values directly matched the issue: destination A read source B, the submission order included A twice before B, and mixed-transfer handles remained live at the next dequeue. On the exact candidate, all three passed.

I then made the schedule more adversarial without changing the assertions: transfers stayed `PROC` for four polls instead of one, and B deferred twice instead of once. All three cases still passed. The neighboring NIXL/disaggregation selection also passed: 103 tests and 14 subtests.

The implementation waits for each staged KV handle before another destination can gather into the worker-private buffer. Before retrying a deferred work item, it drains all handles submitted for the partial fanout and records completed destination session identities on that work item. Queue sharding keeps a room's chunks on one worker/private buffer, so the completion state is updated before that same worker can dequeue the requeued item.

## Source and environment qualification

Imports resolved to the candidate checkout:

- `/job/repo/python/sglang/srt/disaggregation/nixl/conn.py`
- `/job/repo/python/sglang/srt/disaggregation/common/utils.py`

The candidate changes Python only. No C++/HIP/FlyDSL/native source changed, so a native rebuild was not applicable.

The prepared interpreter reports Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, one AMD Instinct MI355X (gfx950), and NIXL imported from `/opt/venv/lib/python3.12/site-packages/nixl/__init__.py`. The focused tests are CPU control-flow simulations; they did not execute a real NIXL transfer on the GPU.

## Limitations

Only one GPU was assigned. A real reproduction needs distinct prefill/decode endpoints with heterogeneous TP, so no multi-GPU or multi-node NIXL run was possible. No model weights were loaded. Accordingly, this review verifies the original issue's deterministic worker contract and fix, but not semantic model output, the production architecture, or causation of the reported corrupted response.

Raw outputs are retained under `reports/j-9d5ede978118/raw/`.
