# Independent review of PR 565 at `fb6a8ee6`

Recommendation: **request changes**. This is a partial original-issue fix, not merely test hardening: the source change fixes all three exact examples, but it does not fully implement the stated circular-write contract.

The prepared base reproduced the original failures with an independent GPU oracle. On the exact candidate, the reported oversized linear case matched, the reported circular case matched, and width 5 was rejected. The candidate's focused tests also passed (17 passed).

The remaining counterexample is `width=4, state_len=5, seqlen=12, cache_seqlens=[4]`. The output matches PyTorch within `4.77e-07`, but the mutated state differs by up to `3.9746375`. When more than one input token maps to a slot, the candidate's vectorized store does not preserve sequential last-write-wins semantics.

This review ran on AMD MI350X `gfx950` with ROCm 7.2, not the reporter's NVIDIA/CUDA system. No native files changed, so a native rebuild was not applicable. See `result.json` and `raw/` for commands, paths, and numerical evidence.
