# Independent review of PR 2030

Upstream issue: https://github.com/sgl-project/sglang/issues/33033

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1954

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2066

Candidate: https://github.com/amdpilot-org/sglang/pull/2030 at `98f960f5c5b6761da1f3bb75703478559443d4ac`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original source-level contract. It removes the quadratic dense mask, retains only compact per-image intervals, applies the non-causal exception in both active Triton extend kernels, and restricts that exception to sliding-attention layers. No issue-specific counterexample remained after independent boundary testing.

## Evidence

The prepared checkout was exactly the requested recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`. Running the candidate's metadata regression against that base failed both tests because `image_span_*` metadata was absent; inspection confirmed that `prepare_attn_masks` allocated an `extend_seq_len × (prefix_len + extend_seq_len)` tensor per request and installed it as `custom_mask`.

At the exact candidate commit, imports resolved to the modified files under `/job/repo/python/sglang`. The metadata tests passed, and both the two-stage and unified Triton extend kernels matched a separately computed dense PyTorch reference on the assigned gfx950 GPU. An independent adversarial variant also passed with a finite sliding window, a one-token image span, a span beginning exactly at the prefix boundary, and spans crossing Triton query blocks.

The full existing Triton attention file produced 11 passes and three ambient/general gfx950 failures. They are unrelated to image spans and match the candidate author's disclosed results. Raw logs are retained in `raw/`.

## Architecture and environment limits

The assigned device was one AMD Instinct MI355X (`gfx950`) using ROCm 7.2 and PyTorch 2.11.0. The reported NVIDIA H20/DGX performance runs and their exact speedups were not reproduced. Gemma4 E4B weights were unavailable, so this review does not claim a full-model output or TTFT reproduction. The candidate changes Python and Triton JIT source only; there is no changed native C++ library to rebuild.
