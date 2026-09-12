# Independent review of PR 2761

Reviewed candidate: `da7442bdcecb38e4fa3310e1834cb800b0d0ddab`

Upstream issue: https://github.com/sgl-project/sglang/issues/36858

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2683

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2798

## Verdict

Recommendation: **accept**. The candidate implements the requested organizational refactor, preserves the tested CLI/default/routing contract, builds a fresh Rust gateway binary, and improves the startup/shutdown/logging structure. No counterexample was found in the executable compatibility checks or the candidate library suite.

The recorded base checkout exactly matched `358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, configuration remains divided between `RouterConfig` and `ServerConfig`, with substantial startup wiring in `main.rs` and `server.rs`; this reproduces the architectural condition described by the feature request. The candidate introduces `GatewayConfig`, moves CLI conversion into the config module, reduces `main`, decomposes router application startup, and adds owned shutdown coordination.

## Independent evidence

- Base library suite: 392 passed, 0 failed.
- Candidate library suite: 344 passed, 0 failed, including the candidate's CLI equivalence/default/adversarial and prompt-shutdown tests.
- Fresh candidate debug binary built with repository-pinned rustc 1.90.0. No prebuilt wheel or stale native library was used.
- Extracted long-option sets from direct `--help`: no base option was removed. Candidate adds `--version` and `--version-verbose`. For `launch --help`, no base option was removed and the candidate exposes `--prefill` plus `--version-verbose`.
- Independent SIGINT run of the rebuilt candidate exited promptly at the 3-second signal trigger (`3063 ms`) without a shutdown timeout warning.
- The same minimal SIGINT run on the recorded base also exited promptly (`3055 ms`). Thus the candidate's claimed long shutdown regression is not a failing-before condition on the recorded campaign base; it was a regression in an intermediate imported proposal. This does not invalidate the original feature refactor, but that before/after claim should not be read as an original-base defect reproduction.

The reduction from 392 to 344 library tests is explained by moving/removing in-tree mesh and WASM implementation tests while delegating those implementations to the already-declared `smg-mesh` and `smg-wasm` crates. The broad library suite still passes, but Kubernetes discovery, live mesh peers, production WASM components, TLS deployment, and a model worker were not available for end-to-end validation.

## Environment

Host architecture was `x86_64`; the prepared Python stack reported Torch 2.11.0 + ROCm 7.2. This Rust control-plane change does not alter GPU kernels or FlyDSL/C++ and requires no GPU numerical reference. No GPU was used. The image initially lacked the candidate report's `/job/.cargo/bin/cargo`; rustup and the pinned 1.90.0 toolchain were installed privately for this review. Build outputs and caches stayed under `/tmp/amdpilot-repo-j-6887cc79907b`.

Raw logs and option diffs are retained in `raw/`.
