# Correction review for fully-NaN request abort

This correction is based on candidate PR https://github.com/amdpilot-org/sglang/pull/1970 at exact commit `41e7ba79d5a83b5e3ddb4f43b733ed0820a04351` and independent review PR https://github.com/amdpilot-org/sglang/pull/2068.

The candidate's request mask, 503 finish reason, no-token-commit behavior, overlap copy, and radix insertion guard are preserved. Two concrete review counterexamples were reproduced and corrected:

1. On the assigned gfx950 GPU, with abort enabled and sanitization disabled, the candidate detected the full-NaN row but left all 163,840 values NaN. After correction the same row is finite before sampling.
2. The candidate prefill branch directly invoked `release_kv_cache`. A failing unit regression showed that the established cleanup helper was not called. Prefill now shares the decode cleanup path, including multimodal release, disaggregated-offload finalization, HiSparse notification, backend preparation, and `is_insert=False` cache release.

The cleanup boundary regression verifies `is_insert=False`; this prevents both radix insertion and the hierarchical-cache write-through publication initiated by insertion. No hierarchical storage backend or model weights were available for an end-to-end host-cache exercise.

Speculative decoding remains explicitly unsupported by the candidate's assertion. It was disabled in the reported production configuration, and no justified algorithm-specific row mapping was established here.

Raw failing-before and passing-after outputs are in `raw/`.
