# Independent review of PR 1776

Candidate: https://github.com/amdpilot-org/sglang/pull/1776 at `176f90767c89458d462c29d12583309886648522`

Upstream issue: https://github.com/sgl-project/sglang/issues/34737

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1826

Recommendation: accept. The candidate fully resolves the original issue's deterministic control-plane contract.

## Evidence

The prepared checkout was clean at the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. With `PYTHONPATH=/job/repo/python`, Python reported both imported source files under `/job/repo/python/sglang/srt/disaggregation/common/`, excluding an installed-wheel false positive.

An independent base harness used the real `DecodeStagingHandler.register_wm_subscriber`, `CommonKVManager._handle_node_failure`, and `_free_and_send_watermark` methods. It observed:

```text
refresh_receiver_is_new False
multi_cp_remaining_after_failure 1
stale_broadcast_calls 1
```

Thus the recorded base retains the first request-scoped receiver and broadcasts to a failed subscriber whose key concatenates two connection-pool CP groups.

At the exact candidate, its five focused regressions passed. The two relevant unit files also passed completely: 91 tests and 14 subtests. An independent adversarial harness then registered a failed two-CP subscriber plus a healthy subscriber. Failure cleanup left only the healthy subscriber, and the subsequent watermark broadcast targeted only that healthy session. It also verified that same-key registration refreshes the receiver and that a new generation registered after a cleanup snapshot survives unregister with the stale token.

The correction matches subscriber keys by intersection with all bootstrap infos cached for the failed node. That addresses the concrete counterexample left by PR 1565, where comparing each CP group to the concatenated subscriber key missed cleanup. Generation tokens prevent the snapshot/unregister interval from deleting a newly registered receiver, and the lock protects registry snapshots and mutation.

## Scope and limitations

The candidate changes Python source, Python tests, and its prior report only. It changes no C++, GPU kernel, or other native source, so a native rebuild was not applicable. Execution used the prepared Python environment with Torch 2.11.0+rocm7.2 and imported the candidate checkout directly.

Only one gfx950 GPU was assigned; the issue's two-prefill-plus-decode heterogeneous-TP topology requires at least three GPUs and multiple processes. No full distributed serving or CUDA reproduction is claimed. No weights were needed, and no model-semantic or distributed-performance claim is made. Within the original deterministic subscriber-lifecycle contract, no remaining counterexample was found.
