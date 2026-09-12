# Independent review of PR 2709

Reviewed exact commit `2d936043ec8b8908941ba527ddfc82310fa08c9a` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the complete contract in https://github.com/sgl-project/sglang/issues/38075.

Recommendation: **request changes**. The patch is a real native implementation, not test-only hardening, but it is only a partial issue fix and its standard `Watch` behavior has protocol counterexamples.

## Evidence

- A release native rebuild of the recorded base was loaded directly from `/tmp/amdpilot-repo-j-f18e41a65082/base-target/release/libsglang_grpc_core.so`. A grpcio standard-health `Check` returned `UNIMPLEMENTED`, reproducing the reported gap.
- A release native rebuild of the exact candidate was loaded from `/tmp/amdpilot-repo-j-f18e41a65082/candidate-target/release/libsglang_grpc_core.so`.
- Candidate Rust tests passed: 14/14. Candidate Python health and bridge tests passed: 6 tests plus 2 subtests. Clippy passed with warnings denied.
- Independent live checks passed for registered aggregate/named `Check`, initial `NOT_SERVING`, recovery to `SERVING`, exception mapping to `NOT_SERVING`, subsequent recovery, and stopping probes after shutdown.
- Independent adversarial evidence found an unchanged `NOT_SERVING` event emitted again after one poll interval. The poller calls `set_service_status` unconditionally, and tonic's watch sender notifies receivers even when the value is equal.
- Independent unknown-service `Watch` returned `NOT_FOUND` and terminated. The `health.proto` bundled by the new `tonic-health` dependency states that `Watch` must return `SERVICE_UNKNOWN` and not terminate, allowing later registration to be observed.

Raw build/test/probe logs and the exact review script were retained outside the revision-switching checkout at `/job/review-evidence-j-f18e41a65082/`.

## Scope classification

This is a partial fix for the standard-health portion. It does not fully resolve the original umbrella issue: native multimodal image/audio/video transport, aggregated and encoder-disaggregated multimodal coverage, and RL pause/drain/update/resume lifecycle semantics remain absent.

## Environment

The host is x86_64 Linux with an AMD Instinct MI350X (`gfx950`), Torch `2.11.0+rocm7.2`, and HIP `7.2.26015`. No GPU was used because the candidate is solely a CPU control-plane protocol change. Rust 1.92.0 and clippy were installed into the private job runtime because the prepared image had no Rust compiler. No model or distributed architecture was validated.
