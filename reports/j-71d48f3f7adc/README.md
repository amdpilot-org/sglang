# Investigation report: issue 35156

Upstream issue: https://github.com/sgl-project/sglang/issues/35156

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1416

The prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` already contains the proposed fix at both reported sites in `sgl-model-gateway/src/policies/cache_aware.rs`: `BasicWorkerBuilder::new(format!(...))` receives the owned `String` without a needless borrow.

Repository history attributes the two-line correction to commit `413df1f8db4fd159c5d07e960f9ec5547edeb5c1`. To retain a failing-before regression, only those two historical `&` tokens were temporarily restored. With Rust/Clippy 1.90, the documented CI command then failed with exactly two `needless_borrows_for_generic_args` errors at lines 682 and 741. Restoring the prepared-base source made the same full command pass. Both affected cache-aware unit tests also ran independently and passed.

No product source change is justified because duplicating the already-present fix would be a no-op. This report and the raw evidence are the only committed changes.

## Evidence

- `raw/source-history.log`: current call sites and the historical fixing diff.
- `raw/toolchain.log`: pinned Rust, Cargo, and Clippy versions.
- `raw/clippy-failing-before.log`: exact failure after restoring the two historical borrows.
- `raw/clippy-passing-after.log`: successful full lint on the prepared base.
- `raw/test-imbalanced-tie-break.log`: first affected test, one test passed.
- `raw/test-cold-start.log`: second affected test, one test passed.

GPU execution and native rebuilding were not relevant to this Rust-only lint defect and were not performed.
