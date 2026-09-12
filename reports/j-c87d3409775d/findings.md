# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/37111

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1013

## Result

The original silent-corruption A/B is environment-blocked here. The report requires two NVIDIA GB10/SM121 devices, TP2 over RoCEv2, and `RadixArk/Qwen3.8-Flash-Next-NVFP4`. This job has one AMD Instinct MI350X (gfx950) and no model weights. A tiny Llama fixture would test transport and engine execution, not Qwen4-Exp QSA/NEXTN semantics, so it was not used as a substitute.

Current `main` already contains a much newer speculative QSA CUDA-graph implementation from merged PR #37500. It builds per-token target-verify/draft-extend graph layouts, refreshes replay metadata on device, pads dynamic draft-extend tails with inert rows, and rejects unsupported branching or over-wide speculative windows. Merged PR #38851 also fixes a separate punctuation-loop mechanism in paged sparse decode by zero-filling scratch tails and using safe offsets. These are relevant evidence, but neither replaces an SM121 TP2 model-level reproduction of issue #37111.

## Reproduced current-head defect

The current QSA+NEXTN draft-extend path called `_speculative_max_row_length` with a real `EagleDraftExtendInput`. That input defines `num_tokens_per_req`, not `draft_token_num`; the base revision raised:

```text
AttributeError: 'EagleDraftExtendInput' object has no attribute 'draft_token_num'
```

This matches the unmerged related correction in upstream PR #37110. The patch uses `num_tokens_per_req`, the canonical row width populated by `EagleVerifyInput` and carried directly by `EagleDraftExtendInput`. It preserves verify behavior and allows draft-extend metadata construction.

## Validation scope

- Failing-before reproduction retained in `evidence/failing-before.txt`.
- Five focused regression/boundary tests pass after the change.
- The complete QSA unit file passes: 41 passed, 1 skipped. The skipped test requires SM121.
- Three tests executed GPU kernels on the assigned gfx950 and compared graph metadata/expansion with independent host or Torch references.

This PR does not claim a full model, SM121, CUDA, TP2, RoCEv2, or multi-node reproduction, and does not label the original semantic-corruption issue solved.
