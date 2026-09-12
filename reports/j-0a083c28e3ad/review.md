# Independent review of amdpilot-org/sglang PR 1508

Upstream issue: https://github.com/sgl-project/sglang/issues/35148

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1546

Candidate reviewed: `ac23f039ebd7e7dcbc44a11226265c82b1e95f18`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**.

## Findings

1. **Blocking: the exact HTTP-worker reproduction remains unfixed.** The issue's compose file passes `--worker-urls http://worker:8000`. `CliArgs::determine_connection_mode` maps that URL to `ConnectionMode::Http`, and `RouterFactory` selects `routers/http/router.rs`. That router forwards the worker response body and contains no reasoning-parser invocation. The candidate only changes files under `routers/grpc/`, so its new parser initialization is unreachable in the issue's supplied configuration. The candidate therefore improves the gRPC backend path but does not fully resolve the original issue, which explicitly reports both HTTP and gRPC gateway modes.

2. **The candidate's regression tests do not exercise a gateway response processor.** They instantiate `reasoning-parser` 1.3.0 directly, manually call `mark_reasoning_started()`, and prove the upgraded dependency can split an issue-shaped string. They do not prove that HTTP routing invokes the parser or that a real tokenizer/request causes either processor to initialize the state correctly.

3. **Independent boundary counterexample: parser state is inferred from the thinking toggle without requiring a prefilled think token.** `ThinkingToggle` and `think_in_prefill` are separate `llm-tokenizer` properties. In both candidate paths, an enabled toggle calls `mark_reasoning_started()` even when `think_in_prefill == false`; non-streaming does not consult `think_in_prefill` at all. An independent test initialized the Qwen3 parser as the candidate does for `DefaultOn`, then parsed a tag-free `plain answer`. It was classified entirely as reasoning instead of normal content. The state transition should be tied to evidence that `<think>` was actually placed in the prefill (or equivalent rendered-prompt evidence), not only that thinking is enabled.

## Verification

- On the exact base, an issue-shaped regression using `Compute 17 * 23. </think>\n\n391` failed: `reasoning_text` was empty and the complete string remained normal content.
- At the exact candidate SHA, both candidate `qwen3_prefill_reasoning` tests passed.
- The full candidate library suite passed: 396 tests, 0 failures.
- The independent non-prefill boundary suite produced one pass and one failure; the tag-free response was incorrectly returned as reasoning after candidate-style initialization.
- `cargo fmt --all -- --check` passed after installing the repository-pinned Rust 1.90 `rustfmt` component; stable rustfmt emitted only the repository's expected nightly-option warnings.
- Cargo resolved `reasoning-parser v1.3.0` from the private Cargo cache and rebuilt the Rust gateway test library into `/tmp/amdpilot-repo-j-0a083c28e3ad/target-candidate/`. No Python wheel or stale installed gateway library was used.

## Environment limitations

The prepared host uses PyTorch 2.11.0 with ROCm 7.2 and an assigned gfx950 GPU. The reporter used CUDA on an NVIDIA L40S, and Qwen/Qwen3.8-27B-FP8 weights were not available. No full-model or NVIDIA reproduction was attempted or claimed. GPU execution is not needed for the deterministic post-generation parser failures above, but model-specific generation and semantic behavior remain unverified.

Raw logs and fetched issue/PR metadata were preserved outside the checkout at `/job/review-evidence-j-0a083c28e3ad/` while revisions were switched.
