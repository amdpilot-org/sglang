# Independent review of amdpilot-org/sglang PR 2048

Recommendation: **accept**. The exact candidate commit `48a1e2bee69055ab37c8339e100ea88a7e37d598` fully resolves the original issue on the reviewed router path.

Upstream issue: https://github.com/sgl-project/sglang/issues/32752

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1986

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2084

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2048

## Findings

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate's route regression reproduces the defect: both successful legacy-affinity PD requests return HTTP 200, but the rendered metrics contain only the `sgl_router_decode_affinity_total` HELP/TYPE declarations and no outcome sample. Plain mode and the empty-decode-pool failure correctly produce no sample.

At the exact candidate commit, the production chat route carries the legacy affinity classification through the current decode policy/admission pipeline and records it only when that proposal is the final selected decode worker. The candidate route regressions pass for same-host selection, breaker fallback, plain mode, and no decode workers. Registry tests also cover load imbalance, malformed URLs, missing same-host peers, and all-open breakers.

The reviewer-added temporary cases sent two successful no-same-host PD requests and observed exactly `sgl_router_decode_affinity_total{outcome="fallback_no_same_host"} 2`. A successful request through the default power-of-two decode policy emitted no affinity sample, avoiding a misleading label for a non-affinity decision. Those temporary tests were removed before returning to the review branch; their patch is retained outside the checkout at `/job/review-evidence-j-d45c5e2aaaed/reviewer-tests.patch`.

No source/native import ambiguity exists here: all changed executable code is in the Rust crate at `/job/repo/experimental/sgl-router`, and Cargo compiled that checkout into the private target directory `/tmp/amdpilot-repo-j-d45c5e2aaaed/cargo-target`. The candidate changes no Python module, C/C++ extension, FlyDSL code, or native shared library, so a native rebuild is not applicable.

## Limitations

The review used deterministic in-process HTTP mock workers. It validates routing execution, branch classification, and Prometheus accounting, but not model semantics or a distributed deployment. No GPU execution was performed because the affected path is CPU-only Rust routing/metrics code; therefore no gfx950-specific or model-weight claim is made. The host architecture was x86_64 Linux.

The prepared `PATH` referenced an unreadable Cargo installation under `/root/.cargo` for the assigned user. To execute rather than trust the candidate's logs, the repository-pinned Rust 1.90 toolchain and rustfmt component were installed in the job-private runtime directory. Full raw logs and the reviewed diff are retained under `reports/j-d45c5e2aaaed/raw/`.
