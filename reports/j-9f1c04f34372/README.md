# Independent review of PR 3181

Upstream issue: https://github.com/sgl-project/sglang/issues/11186

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3189

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/3181 at exact commit `34fe255b6c2bae81eb73d7546d6526033bbbeda1`.

This PR contains only the review report, not a duplicate of the candidate patch.

Recommendation: **request changes**. The candidate corrects the concrete Rust-generation-egress counterexample: scheduler snapshots are appended to the positional frame, decoded by the rebuilt Rust server, retained newest-per-rank, and exposed by the Rust JSON `/v1/loads` handler. Its Python regression suite passed, all 265 Rust tests passed, and an optimized Rust 1.92 native library built and imported from the private build path.

It does not fully resolve the original issue's complete contract. In Rust-server mode the new HTTP cache has no watch/SHM/ZMQ reader or fallback; it is populated only by generation egress. Consequently an idle rank, a rank before its first token response, or a scheduler whose snapshot collection temporarily fails is absent from Rust `/v1/loads`, whereas the original request explicitly requires retaining watch mode and always storing the latest information for external query. The Rust endpoint also rejects `format=prometheus`, which the established Python `/v1/loads` API supports, and returns null accelerator metadata.

No GPU numerical claim was needed for this transport/cache review. One AMD Instinct MI355X was visible through Torch 2.11.0+rocm7.2 / HIP 7.2, but no model-serving run was performed. Multi-rank and multi-node behavior remains unverified.

Raw logs were retained outside the checkout at `/job/review-evidence-j-9f1c04f34372/`.
