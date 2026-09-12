# Investigation: compact target-verify capture on hybrid Triton attention

Upstream issue: https://github.com/sgl-project/sglang/issues/36481

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1171

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The current source already has the backend capability information needed for a safe fallback:

- `TritonAttnBackend` inherits `supports_ragged_verify_graph = False`.
- `HybridLinearAttnBackend.supports_ragged_verify_graph` requires both its full-attention and linear-attention children to support ragged graphs.
- `DecodeCudaGraphRunner._can_run_ragged_verify_graph` rejects replay when that capability is false.

The missing boundary was startup capture. `DecodeCudaGraphRunner.capture` ran warmup and every requested compact target-verify capture shape without checking the same capability. Thus a hybrid Triton+GDN model could enter the unsupported `extend_attention_fwd` capture path before replay admission ever had a chance to select eager execution.

The correction checks the capability at the start of `capture`. Unsupported compact target-verify graphs are not captured and the existing replay admission remains false, so target verification uses eager execution. Supported ragged backends and all non-compact graph captures are unchanged.

## Evidence

The focused regression was run once against the base behavior with the production guard temporarily removed. It failed because `capture()` called its poisoned `warmup()` even though the backend declared ragged graphs unsupported. Raw output is retained at `/tmp/amdpilot-repo-j-3beed6345a9c/failing-before.txt`.

With the guard restored, the complete capability suite passed: 6 tests and 3 subtests. It covers the failing unsupported-compact case and independent supported-compact and non-compact boundaries. Raw output is retained at `/tmp/amdpilot-repo-j-3beed6345a9c/passing-after.txt`.

The assigned device was identified by `rocm-smi` as AMD Instinct MI350X, gfx950. The source report is B300/CUDA 13.2 and no Qwen3.5/Qwen3.6 or DSpark weights were available. Therefore the original illegal memory access and full serving path were not reproduced here; the verified result is the issue-specific admission-control correction, not a claim that Triton ragged graph execution itself works.
