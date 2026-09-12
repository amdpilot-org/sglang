# Investigation: DFlash2 concurrent-request corruption

Upstream issue: https://github.com/sgl-project/sglang/issues/36548

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1147

## Result

The reported Qwen3.8-27B DFlash2 failure could not be reproduced in this job.
The report requires the Qwen3.8-27B NVFP4 target and DFlash2 draft checkpoints
on an NVIDIA target-verify path. The prepared environment provides one AMD
Instinct MI355X (`gfx950`) and does not provide those weights. A tiny Llama
serving fixture would exercise transport and ordinary engine execution, but it
would not exercise Qwen3.8's hybrid GDN recurrent state or the DFlash2 draft
model, so it was deliberately not substituted for the issue reproduction.

No source correction is proposed. The public issue follow-up localizes the
first divergence to the GDN selected-step SSM transition despite bitwise-equal
inputs and persistent pre-state; it also reproduces at batch size one with all
drafts rejected. That evidence does not support the title's initial hypothesis
of request/context ownership corruption. Concurrency raises the observed error
rate, but is not required for the underlying numerical divergence.

## Current related implementation

The prepared base already contains two relevant safeguards:

1. `test_decode_bookkeeping_ownership.py` records the only permitted owners of
   per-request decode clocks, KV watermarks, verify counters, and SWA eviction.
   Its two tests pass, including the guard that speculative draft workers do
   not duplicate scheduler bookkeeping.
2. The GDN ReplaySSM fold implementation has an independent snapshot baseline.
   On the assigned `gfx950`, all four tests pass for FP32 and BF16 state,
   non-contiguous request slots, null slots, accept lengths 1/3/4, untouched
   slots, and 256 chained commits. This validates that kernel on AMD only; it
   does not validate Qwen3.8-27B, NVFP4, CUDA, or HTTP concurrency.

ReplaySSM is not currently a solution for the reported configuration. In
`python/sglang/srt/mem_cache/kv_cache_configurator.py`,
`_build_hybrid_req_pool` explicitly raises for DFLASH/DSPARK plus
`--enable-linear-replayssm-spec` when the model is non-KDA. Qwen3.8's GDN path
therefore remains on the recurrent target-verify implementation implicated by
the issue evidence.

The CUDA/CuTeDSL ring-verify test was also attempted, but collection fails in
the prepared ROCm environment because the CUDA Python bindings are absent
(`ModuleNotFoundError: No module named 'cuda'`). This is an expected platform
boundary and is retained as evidence, not treated as a product failure.

## Reproduction and evidence

Raw command output is retained in `raw/`:

- `gdn_replayssm_spec_fold.log`: gfx950 execution, 4 passed.
- `decode_bookkeeping_ownership.log`: source ownership regression, 2 passed.
- `gdn_cutedsl_ring_verify.log`: CUDA-only test collection blocker.
- `replayssm_memory_pool_excerpt.txt`: source excerpt of the relevant memory
  pool implementation.

The exact source issue and comments were inspected with:

```bash
gh issue view 36548 --repo sgl-project/sglang --comments \
  --json title,body,state,comments,updatedAt,url
```

No full-model, HTTP concurrency, CUDA, RTX Pro 6000, semantic-accuracy, or
multi-node claim is made.
