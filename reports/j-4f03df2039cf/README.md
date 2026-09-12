# Correction generation 1: pre-parse multimodal admission

This change preserves candidate PR https://github.com/amdpilot-org/sglang/pull/2784 at
`e3da2b295dcd70b2c65158c3458006ea03f2b9b4` and corrects the concrete HTTP-body
counterexample reported by review PR https://github.com/amdpilot-org/sglang/pull/2881.

The configured limiter now takes a provisional one-item lease before an admitted
generation endpoint consumes its ASGI body. Tokenizer-side admission resizes that
same lease to the parsed image/video/audio count and retains the candidate's
weighted lifetime accounting. At capacity, excess requests receive HTTP 503
without invoking their body receive callback.

The exact Kimi-K2.6 200-request, 5,600-image workload remains unverified because
the weights and equivalent 1.5 TiB host were unavailable. The Rust frontend and
EPD language-only mode still reject this option because their separate ownership
paths were not available for qualified validation; they are not claimed fixed.

Raw commands and outputs are retained in `evidence/`; structured claims are in
`result.json`.
