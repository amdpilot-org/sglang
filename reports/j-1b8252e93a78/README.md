# Independent review of PR 2013

Upstream issue: https://github.com/sgl-project/sglang/issues/33656

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2046

Candidate: https://github.com/amdpilot-org/sglang/pull/2013 at exact commit
`8a0f2af113ca421283e8cc9fe1099cc349fb419d`.

## Recommendation

Accept the candidate as a narrow correction to follower backup completion
accounting. It is not a full fix for the original issue.

The candidate changes the empty set of follower-local writes from failure to
vacuous success. Its regression failed on the recorded base with
`completed_tokens=0` instead of `128`, and all three cases passed at the exact
candidate commit. Independent cases also confirmed that replicated DSV4/SWA
transfers can be skipped locally while malformed or incomplete rank-local
storage acknowledgements remain unsuccessful.

This evidence validates only the completion-accounting defect. It does not
exercise a successful DSV4 FULL+SWA restore, the reported `512` versus `8448`
SWA position mismatch, kv-canary, multi-rank Mooncake identity, logits, or
sampling probabilities. The candidate itself accurately states these limits;
its prose was not treated as proof of the original contract.

## Environment and source paths

- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Python: `/tmp/amdpilot-repo-j-1b8252e93a78/venv/bin/python`
- Imported SGLang: `/job/repo/python/sglang/__init__.py`
- Imported changed module:
  `/job/repo/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py`
- Accelerator: one AMD Instinct MI350X, `gfx950`, ROCm 7.2
- Reported production architecture: eight NVIDIA H20 GPUs with CUDA 13.0.1
- No DeepSeek-V4 checkpoint was available.
- No native files changed, so a native rebuild was not applicable.
- The Rust-backed linker subset could not initialize because `cargo` was not
  executable/available; its Python/non-Rust portions produced 56 passes before
  seven environment failures.

Raw logs and the independent test fixture were retained outside the checkout
at `/job/review-evidence-j-1b8252e93a78/` while revisions were switched.

