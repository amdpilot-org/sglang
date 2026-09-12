# Correction generation 2 validation

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3095 at exact commit `d8a3b79cb118a1d6248317c8cf0e6c68d866ae30`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3183

The exact candidate's useful single-rank scheduler behavior, deterministic result assembly, explicit opt-out, multi-rank fallback, and SGLang-owned construction lock are preserved. The review's native thread-safety claim was independently reproduced against that candidate using Transformers 5.12.1's real `local_torch_dtype` context: requested dtypes were crossed and the process default was corrupted.

The correction serializes the complete native `from_pretrained` boundary with `model_construction_lock`. The pinned native APIs do not expose model construction separately from checkpoint I/O, so a narrower generally correct lock is not available. Customized loaders still benefit from the outer parallel scheduler.

Failing before:

```text
{'observed': {'a': torch.float64, 'b': torch.float32}, 'original': torch.float32, 'final': torch.float16}
exit_code=1
```

Passing after:

```text
{'observed': {'a': torch.float16, 'b': torch.float64}, 'original': torch.float32, 'final': torch.float32}
exit_code=0
```

Focused tests passed 32/32; extended compatibility passed 68 tests plus 9 subtests; the synthetic GPU/NumPy numerical regression passed on one AMD Instinct MI350X. See `result.json` and `raw/` for exact commands and output.

This does not establish the original Qwen-Image performance claim. Real weights were unavailable. Wake/refit, pinned checkpoint mappings, multi-rank parallel loading, and automatic host-memory/storage-pressure bounds remain unimplemented or unverified.
