# Rust `/v1/loads` correction generation 2

This correction preserves the fixes from candidate PR https://github.com/amdpilot-org/sglang/pull/3181 at exact commit `34fe255b6c2bae81eb73d7546d6526033bbbeda1` and resolves the concrete counterexamples reported by independent review PR https://github.com/amdpilot-org/sglang/pull/3229.

The Rust cache now receives every successful scheduler watch publication and is seeded at startup after the load inquirer is initialized, so it does not depend on a generated response. Generation frames remain a second, timestamp-arbitrated update source. The Rust endpoint also emits the established Prometheus format and receives accelerator metadata through its typed launch configuration.

Failing-before and passing-after commands, outcomes, and limitations are recorded in `result.json`. Raw command logs are retained under `/job/reports/j-fcde7ccf0fa0/evidence/`; the rebuilt native library is `/tmp/amdpilot-repo-j-fcde7ccf0fa0/cargo-target/release/libsglang_server.so`.
