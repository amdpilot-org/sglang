# Independent review of amdpilot-org/sglang PR 807

- Upstream issue: https://github.com/sgl-project/sglang/issues/38202
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/852
- Candidate commit: `8736f3389903d2c6bb3cc3e268bb8ef536f32128`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**

## Finding

The candidate fully resolves the original source-level accounting defect. On the recorded base, `_resolve_dflash_draft_cell_size` passed the raw tensor-parallel width (`tp_size`) to `get_num_kv_heads`, while every relevant KV-pool builder derives local KV heads from `attn_tp_size`. With `tp_size=16` and `attn_tp_size=1`, the independent reproduction observed a 2,560-byte/token reservation instead of 40,960 bytes/token for its bfloat16 geometry. The assertion failed because the resolver passed 16 rather than 1.

The exact candidate changes that argument to `get_parallel().attn_tp_size`. Its regression passed, as did independent full-DP, partial-DP, no-DP, low-KV-head clamping, fp8, pool-capacity, and GPU storage-byte cases. A synthetic combined budget changed from the incorrect 31-token capacity to the corrected 19-token capacity. On the assigned MI355X/gfx950, actual fp8 tensor storage was 21,504 bytes for the attention-local geometry versus 1,344 bytes for the incorrectly TP-sharded geometry, exactly a 16x ratio.

The candidate is one commit directly atop the recorded base. Python imported `sglang` and the resolver from `/job/repo/python/sglang/...`, not an installed copy. The change is Python-only; no native source or native build product changed, so a native rebuild was not applicable.

## Classification and limitations

This is a full fix for the original accounting contract, not merely test hardening: it corrects the exact topology width used to reserve the draft pool, and the resulting combined per-token budget reduces `max_total_num_tokens` before both pools allocate.

The reported Kimi-K3/DSPARK deployment was not reproduced end-to-end. The environment exposes one AMD Instinct MI355X (`gfx950`) rather than the reported B300 multi-GPU deployment, and the Kimi-K3/DSPARK weights and 16-way topology were unavailable. Therefore this review verifies the defective calculation, allocator topology contract, downstream capacity consequence, and GPU byte geometry, but does not claim a full-model serving startup or OOM reproduction.

Raw review evidence is retained outside the revision-switching checkout at `/job/review-evidence-j-ae384682feee/`.
