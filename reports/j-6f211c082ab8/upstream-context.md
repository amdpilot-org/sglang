# Read-only upstream context

## Issue 38745

Title: "`DUAL_STREAM_TOKEN_THRESHOLD` makes the DSA indexer's dual-stream gate
unsatisfiable on ROCm — the leftover #14337's TODO named"

The issue reports:

```python
DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0

enable_dual_stream = (
    self.alt_stream is not None
    and get_is_capture_mode()
    and q_lora.shape[0] > 0
    and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
)
```

On HIP the last two clauses are `tokens > 0 && tokens <= 0`, so no input can
enable dual stream. The issue also notes that `SGLANG_ROCM_USE_MULTI_STREAM`
allocates and threads an alternative stream even though this gate cannot use it.

When read on 2026-09-10, the issue was open and had no comments.

## PR 14337

PR 14337, "remove unecessary dual stream token threshold from the rest of models
(qwen moe, kimi linear, etc.)", was merged on 2025-12-07. It removed thresholds
from `bailing_moe.py`, `kimi_linear.py`, `llada2.py`, and `qwen2_moe.py`.

Its body explicitly left the DSA/NSA work unresolved:

> TODO there are still some in GDN and NSA backend, I'm not sure if those are
> needed either. it can be done seperately

## PR 9405

PR 9405, "Use dual stream for DS MoE whenever cuda graph is used (instead of with
token threshold)", was merged on 2025-11-22. A collaborator comment says it was
updated "to follow trtllm and use dual stream whenever cuda graph is enabled."

That PR changed `deepseek_v2.py` MoE behavior, not the DSA/IndexerKPool gate.

## PR 28846

PR 28846, "Combined qwen3.5_fp4 + glm5 optimization work for MI355X (gfx950)",
was closed without merging on 2026-06-21. Its body says:

> `dsa_indexer.py` — documents the HIP indexer dual-stream gate
> (`DUAL_STREAM_TOKEN_THRESHOLD`), kept disabled on HIP: tested neutral at TP=4
> and risks the TP=8 HW-queue regression.

Its exact diff hunk was:

```python
# HIP keeps this 0 (disabled): enabling indexer dual-stream overlap was tested
# neutral at TP=4 CONC=32/64 (overlapped Q/K projections are tiny vs the
# random-KV paged-MQA logits cost that dominates) and risks the TP=8 alt-stream
# HW-queue-oversubscription regression. CUDA keeps the upstream 1024.
DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0
```

This is the strongest available evidence that the zero value is intentional on
HIP. The PR was not merged, so its comment is not authoritative upstream policy,
but it is a direct maintainer-side rationale from a related optimization effort.

## Current upstream state

Read-only upstream main was fetched as
`3700c4ee26a1df3fd27e10a4a83d40d991d87d6c`. It still contains:

```python
DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0
```

and the same `q_lora.shape[0] > 0` plus
`q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD` predicate in both DSA and
KPool. No already-merged fix was found.
