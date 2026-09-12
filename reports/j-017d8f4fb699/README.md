# Independent review of PR 1890

Candidate: https://github.com/amdpilot-org/sglang/pull/1890 at exact commit `e17404202ff719a952d605431afe34514bb62643`

Upstream issue: https://github.com/sgl-project/sglang/issues/38167

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1900

## Recommendation

`request_changes`. The candidate is a partial fix, not a full resolution of the original issue.

The recorded base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) reproduced the previously reported transport and fatal-state gaps: default scheduler RPC configuration resolved to `None`, the real ZMQ socket retained `RCVTIMEO=-1`, and `Scheduler` had neither `_handle_execution_error` nor `_fatal_state_result`.

At the exact candidate commit, its focused suite passed (`216 passed, 16 warnings, 62 subtests`). Its ordinary disaggregated encoder loop now latches the reported fatal error and rejects later encoder work, and the default RPC timeout is one hour.

However, an independent denoiser-prefetch probe raised `RuntimeError("CUDA driver error: device not ready")` on request `r1`. The loop did not call `_handle_execution_error`, left `_fatal_error_message` as `None`, and dispatched request `r2`. Output:

```text
candidate_prefetch_dispatches=['r1', 'r2'] fatal_state=None
```

This is a concrete remaining poison-after-failure path in `_disagg_prefetch_event_loop`, used by disaggregated denoiser/decoder receiver roles. The related multi-rank non-rank-0 loop likewise was not changed by the candidate. Therefore the candidate does not establish the original contract that no later GPU work is accepted after the device enters the reported unrecoverable state.

The candidate also intentionally does not reject 1344x768 at admission and does not turn ordinary OOM into a clean insufficient-memory response. That restraint is reasonable without the reporter's hardware and weights, but these portions of the original issue remain unresolved/unverified.

## Environment and commands

Prepared interpreter: `/tmp/amdpilot-repo-j-017d8f4fb699/venv/bin/python`.

Focused candidate tests:

```bash
PYTHONPATH=python /tmp/amdpilot-repo-j-017d8f4fb699/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_scheduler_device_failure.py \
  python/sglang/multimodal_gen/test/unit/test_scheduler_client.py \
  python/sglang/multimodal_gen/test/unit/test_server_args.py
```

The independent probe directly drove `Scheduler._disagg_prefetch_event_loop` with two `transfer_compute` queue items. The first mocked denoiser compute raised the exact reported fatal marker and the second recorded whether it was dispatched.

Source imports were confirmed at `/job/repo/python/sglang/multimodal_gen/runtime/managers/scheduler.py` and `/job/repo/python/sglang/multimodal_gen/runtime/disaggregation/scheduler_mixin.py` while the exact candidate was checked out. No C++/CUDA/HIP/native files changed, so no native rebuild was applicable.

One assigned GPU was used for an environment/numerical check: Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, AMD Instinct MI355X, and `(tensor([1,2,3]) ** 2).sum()` returned `14.0` on device.

## Limitations

The prepared system is AMD MI355X/ROCm 7.2, not RTX 5070 12 GB/WSL2/CUDA 13.0. MiniMax-H3 weights, FL2VA BF16 configuration, and the turbo LoRA were unavailable. Consequently the exact 1344x768 failure, its memory-causality classification, and a subsequent real 480p request could not be reproduced. The tiny Llama fixture is not architecture-qualified for MiniMax-H3 and would not validate this diffusion/model-memory failure, so it was not substituted as proof.
