# Gemma 4 ROCm vision synchronization investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35673

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1348

The reported watchdog stack ends in `resolve_precomputed_max_seqlen`. In the
prepared implementation, Gemma 4 selects `triton_attn` on ROCm and calls that
backend with `cu_seqlens=None`. The backend creates the dense cumulative-length
tensor on the GPU and, when no `max_seqlen` is supplied, computes its maximum
and calls `.item()`. Gemma 4 already has the exact dense maximum as the host
integer `seq_len`, so repeating that device-to-host synchronization in every
vision encoder block is unnecessary.

The correction forwards `seq_len` as `max_seqlen`. It does not alter attention
math, packed-input behavior, tensor-parallel sharding, or SWA allocation.

Evidence is retained under `reports/j-ad639cc3195c/raw/`. The new contract tests
were run once with the source correction removed and both failed, then passed
after it was restored. A real single-gfx950 run exercised
`VisionTritonAttention` and compared its output with PyTorch SDPA for one-item
and batched dense inputs.

This is a verified candidate rather than a claimed full reproduction. The job
had one GPU and no Gemma-4-31B-it weights, while the report requires eight GPUs,
the full multimodal model, hybrid SWA allocation, and serving readiness.
