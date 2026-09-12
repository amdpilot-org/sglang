# Investigation report: Mooncake failed-session recovery

Upstream issue: https://github.com/sgl-project/sglang/issues/37022

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1038

## Finding

The prepared source at `358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the targeted recovery for the reported persistent `remote mooncake session ... is not alive` failure mode. A transfer failure still blacklists the remote session immediately, but an opt-in prefill-side probe loop periodically calls Mooncake's `send_probe`. A successful probe removes the session from both `failed_sessions` and `session_failures`, allowing later requests to use the recovered decode instance without re-registration or restart.

This implementation came from merged upstream PR https://github.com/sgl-project/sglang/pull/25287, which addressed the same sticky-blacklist mechanism described in the related issue https://github.com/sgl-project/sglang/issues/13054. It is intentionally disabled unless `SGLANG_ENABLE_FAILED_SESSION_PROBE=1`; the interval defaults to 30 seconds and is configurable with `SGLANG_FAILED_SESSION_PROBE_INTERVAL_S`.

No production-path correction was justified on top of the prepared source. This PR adds missing regression coverage for:

- successful probe un-blacklisting a session;
- unsuccessful probe retaining the blacklist and failure count;
- a probe exception for one peer not preventing another peer's recovery;
- the SGLang transfer-engine wrapper forwarding `send_probe` to the native Mooncake engine.

The prepared native Mooncake extension exposes `TransferEngine.send_probe`, so the checked-in wrapper and recovery loop are connected to an available backend API. Raw environment/API evidence is in `raw/environment_and_probe_api.log`; focused test output is in `raw/mooncake_recovery_after.log`.

## Reproduction and limitations

Run:

```bash
/tmp/amdpilot-repo-j-3dee4b6f4277/venv/bin/python -m pytest -q \
  test/registered/unit/disaggregation/test_mooncake_transfer_batching.py
```

The result was 9 passed plus 3 passing subtests. These deterministic tests validate the session-state recovery behavior and wrapper delegation, not a full serving deployment.

The reported 5-prefill/3-decode GLM5.2 workload, NVIDIA/XCCL environment, and multi-node RoCEv2 RDMA topology were unavailable. The assigned device is one AMD Instinct MI350X (`gfx950`). No model weights or equivalent multi-node RDMA fabric were available, so the original production failure was not reproduced end to end and no claim is made about its underlying network trigger, GLM5.2 semantics, or distributed reliability. GPU execution was not relevant to this control-plane state-machine regression and was not used as substitute evidence.

No native SGLang or FlyDSL library was changed or rebuilt.
