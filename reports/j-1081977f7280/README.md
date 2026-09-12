# Independent review of PR 3271

Reviewed https://github.com/amdpilot-org/sglang/pull/3271 at exact commit `1fadfed0a286bddaf57ddbab0463f5613ec52f28` against https://github.com/sgl-project/sglang/issues/11186 and the three concrete counterexamples inherited from https://github.com/amdpilot-org/sglang/pull/3229.

Recommendation: **accept**. The candidate is a full original-issue fix, not merely test hardening. On the recorded base, all six focused piggyback/watch regressions failed. At the exact candidate commit, 14 focused Python regressions, 266 Rust tests, and two independent adversarial tests passed. The changed Rust extension was rebuilt with pinned Rust 1.92 and imported directly from the private build target; the source imports resolved to the candidate checkout.

The independent adversarial tests verified that a successful idle/watch snapshot reaches the Rust cache without a generation frame, and that a later best-effort collection exception does not replace or delete the last successful update. Direct Rust handler coverage verified Prometheus output and non-null accelerator metadata. Source inspection confirmed each DP leader embeds its own Rust listener and feeds its own watch snapshot, while generation frames remain a timestamp-arbitrated secondary source.

No GPU kernels, model weights, semantic model behavior, multi-node transport, or numerical claims are involved in this control-plane change. One assigned AMD Instinct MI355X (`gfx950`) was visible, but GPU execution was intentionally not used as proof. A live multi-rank server was not launched; multi-rank behavior is supported by topology inspection and newest-per-rank unit coverage rather than an end-to-end distributed run.

Raw review evidence is retained outside the checkout at `/job/review-evidence-j-1081977f7280/`. The rebuilt native library is `/tmp/amdpilot-repo-j-1081977f7280/cargo-target/release/libsglang_server.so`.
