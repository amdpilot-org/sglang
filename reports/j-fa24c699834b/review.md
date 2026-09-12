# Independent review of PR 2774

Candidate: https://github.com/amdpilot-org/sglang/pull/2774 at `8cf9aac4789b675f221af1728ffa391749866b67`

Upstream issue: https://github.com/sgl-project/sglang/issues/32101

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2805

## Verdict

Recommendation: **unverified**. The candidate is a plausible partial implementation and adds useful portable policy hardening, but it does not establish a full fix for the original issue. `fully_resolves_original` is therefore `false`.

The exact prepared base was `358c163250ad3b1f62939b01ce1314a0a31a0365`; it matched the recorded base with no discrepancy. The candidate was reviewed at the exact requested commit, and the checkout was returned to `amdpilot/j-fa24c699834b` before this report was committed.

## What the candidate demonstrably changes

The bridge recognizes `gemma4` and `gemma4_text`, rejects unsafe radix/overlap/chunked-prefill combinations, avoids replacing Gemma 4 attention modules, allocates model-owned cache lists with `model.make_cache()`, decodes requests individually, drops request cache references at completion, and documents a conservative 2,048-token invocation. The portable policy tests pass, as does an independent exhaustive configuration matrix.

This is Python-only work. No native C++, Metal kernel, FlyDSL, or other compiled source changed, so there was no native artifact to rebuild.

## Why this is not verified as the original fix

The prepared node is Linux x86_64 with one AMD ROCm GPU. Both `mlx` and `mlx_lm` are unavailable. Importing the MLX runner on the recorded base fails immediately with `ModuleNotFoundError: No module named 'mlx'`. On the candidate, all 43 tests in the MLX runner regression file skip. Only three pure-Python policy tests execute.

Consequently, the review could not execute any acceptance criterion that depends on the real architecture or Metal runtime: a tiny public `gemma4.Model` and text-backbone test, greedy parity beyond the sliding window, scheduler-driven concurrent request isolation/release, real E2B startup and HTTP serving, or token-for-token comparison with unmodified `mlx-lm`. The candidate's cache tests use manually constructed doubles and are themselves behind the MLX skip gate. Its PR prose correctly calls these paths unverified; that prose is not proof.

The supplied tiny Llama HTTP fixture was not used as substitute evidence because it validates transport and engine execution only and cannot qualify Gemma 4 model structure, YOCO cache ownership, Metal behavior, or semantic accuracy. Likewise, an AMD GPU smoke would be unrelated to this Apple Silicon feature.

## Evidence

Raw issue/PR snapshots, the exact candidate diff, upstream `mlx-lm` Gemma 4 source inspected during review, command logs, exit records, and environment output are preserved outside the revision-switching checkout at `/job/review-evidence/j-fa24c699834b`. Structured commands and outcomes are also recorded in `result.json`.

No executable semantic counterexample was observed because the required runtime is unavailable. The unresolved adversarial cases are listed in `remaining_counterexamples`; any of them can still falsify the candidate's core correctness claim on Apple Silicon.
