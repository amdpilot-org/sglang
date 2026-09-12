# Independent review of amdpilot-org/sglang PR 2873

Candidate: https://github.com/amdpilot-org/sglang/pull/2873 at exact commit `19b9f743bf6d251f666f43cc6c165a7d375b3efd`

Upstream issue: https://github.com/sgl-project/sglang/issues/38075

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2807

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2905

## Verdict

Recommendation: **request changes**. The candidate is a verified partial fix for the standard gRPC health-checking portion, but it does not fully resolve the original umbrella issue.

The exact candidate's rebuilt native library correctly:

- implements standard `grpc.health.v1.Health/Check` and `Watch` alongside the native service;
- maps a false `RuntimeHandle.health_check()` result to `NOT_SERVING`;
- avoids duplicate unchanged Watch events across multiple one-second polls;
- returns `SERVICE_UNKNOWN` for an unregistered Watch without terminating the stream; and
- passes its focused Rust and live grpcio regression coverage.

The candidate does not add native gRPC image, audio, or video inputs; aggregated or encoder-disaggregated multimodal coverage; RL-specific operations; explicit drain/accepting/readiness state; or tested pause/drain, weight-update, ready-to-serve, and resume lifecycle semantics. Those are original-issue requirements, not optional follow-up polish.

## Reproduction and source-path evidence

The checkout began on the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. There was no importable bundled `_grpc` module, so both revisions were built from source with Rust 1.92.0 into distinct private target directories. Live Python probes loaded the absolute `.so` paths with `importlib`, and the candidate pytest run printed the preloaded path.

Recorded base release artifact:

`/tmp/amdpilot-repo-j-f2399b3900a4/base-target/release/libsglang_grpc_core.so`

Base live result:

```text
check_error=UNIMPLEMENTED
watch_open_error=UNIMPLEMENTED
unknown_open_error=UNIMPLEMENTED
```

Exact candidate release artifact:

`/tmp/amdpilot-repo-j-f2399b3900a4/candidate-target/release/libsglang_grpc_core.so`

Independent candidate result with health held at false:

```text
check=2
watch_first=2
watch_second_error=DEADLINE_EXCEEDED
watch_wait_seconds=2.4
unknown_first=3
unknown_second_error=DEADLINE_EXCEEDED
unknown_wait_seconds=2.4
```

Wire status 2 is `NOT_SERVING`; wire status 3 is `SERVICE_UNKNOWN`. Waiting through more than two polling intervals without a second Watch message demonstrates unchanged-event suppression. The unknown stream's deadline, rather than `NOT_FOUND`, demonstrates that it remained open.

Candidate Rust tests: 16 passed. Candidate Python regression/bridge tests against the explicit rebuilt release library: 7 passed and 2 subtests passed.

## Architecture and environment limits

The host is x86_64. Torch reports `2.11.0+rocm7.2`, HIP `7.2.26015`, and one visible AMD Instinct MI350X. No GPU execution was used: the candidate is a CPU control-plane-only health change with no numerical output. A GPU smoke would not substantiate the missing multimodal architectures, encoder-disaggregated deployment, or RL lifecycle contract. No model weights or distributed workload were used.

Raw logs and the independent probe are preserved outside the checkout in `/job/review-evidence-j-f2399b3900a4` while the committed report remains on `amdpilot/j-f2399b3900a4`.
