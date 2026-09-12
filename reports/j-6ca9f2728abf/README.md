# Independent review of PR 1202

Reviewed candidate commit `87a8434d94c3f73626fdb66d00c73268f058e1d3`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and
the two startup failures in https://github.com/sgl-project/sglang/issues/36536.

Recommendation: **accept**. The candidate fully resolves the reported
`LlamaForCausalLM` contract by choosing the issue's explicitly acceptable
behavior: reject `--enable-multi-layer-eagle` early with a clear `ValueError`.
It is a source fix with regression coverage, not test-only hardening.

On the unmodified recorded base, a deterministic two-layer random Llama loaded
on the assigned gfx950 GPU and reproduced both reported failures: the auto-fill
case reached the `(8, 5)` worker assertion, while the explicit `(5, 1, 6)` case
reached the unexpected `draft_model_idx` constructor keyword. At the exact
candidate commit, the same two launch commands exited during argument
resolution with the new clear `ValueError`, before `Load weight begin`.

The candidate's three focused tests passed. The complete server-argument module
had 227 passes and two unrelated existing ROCm failures because prefill context
parallelism is deliberately rejected on HIP. Source imports were confirmed from
`/job/repo/python/sglang`; `LlamaForCausalLM.__init__` still has no
`draft_model_idx` parameter. The diff contains no native files, so no native
rebuild was applicable.

Environment limitations: this review used one AMD Instinct MI350X (`gfx950`),
ROCm 7.2, and Torch 2.11.0, rather than the reporter's NVIDIA/CUDA system. The
private random fixture validates startup, configuration, model loading, and the
constructor path only; it does not validate TinyLlama semantics. No embedded-MTP
checkpoint was available, so successful generation on a supported multi-layer
architecture was not run; the independent MiMoV2 configuration boundary passed.

Raw logs remain outside the revision-switching checkout at
`/job/review-evidence-j-6ca9f2728abf/`.
