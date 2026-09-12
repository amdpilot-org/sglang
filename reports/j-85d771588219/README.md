# Rust-egress piggyback load correction

Candidate: https://github.com/amdpilot-org/sglang/pull/2993 at `d0666577d91c31b154d6492cdb4eaacef2fddb1c`

Independent review: https://github.com/amdpilot-org/sglang/pull/3080

The candidate's Python egress fixes are preserved. The remaining Rust path now appends the scheduler `LoadSnapshot` as a trailing, backward-compatible batch-header column, records the newest report per DP rank in the Rust frontend dispatcher, and serves the cached reports from the Rust `/v1/loads` JSON endpoint. Stale out-of-order reports are rejected.

The candidate counterexample was reproduced before editing by inspecting the exact `push_generation` body: it contained no `payload.load_snapshot` reference and the check exited 1. The added Python wire regression would therefore fail on that commit because its four-column header has no snapshot; it passes after the correction.

The Rust crate was tested in full and its release PyO3 extension was rebuilt into `python/sglang/srt/rust_extensions/_server.cpython-312-x86_64-linux-gnu.so` with private Cargo/Rustup/target directories under `/tmp/amdpilot-repo-j-85d771588219`.

No model weights or live serving process were required to establish the dropped-field transport bug. No GPU execution was performed. Rust `/v1/loads` currently reports accelerator topology fields as `null`, because that metadata is not present in the Rust launch handoff, and supports JSON rather than Prometheus output; these are explicit remaining parity limitations rather than evidence against the transport correction.
