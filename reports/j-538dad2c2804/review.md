# Independent review of amdpilot-org/sglang PR 2141

Candidate reviewed: `cb61a727f731f4b00935590d85372031d8625a46`

Recorded failing base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **unverified**. The patch is source-consistent and passes independent model-free contract checks, but this environment cannot execute MLX and the candidate's own added tests all skip here. Therefore `fully_resolves_original` is false for this review.

## Findings

1. The original missing-field failure is reproduced on the recorded base. The inherited insertion path raises for missing `mamba_max_states_per_path`; the inherited match finalizer independently raises for missing `mamba_checkpoint_grid`.
2. The base already defines `MlxAuxiliaryStateReqToTokenPool.mamba_allocator = self.mamba_pool`. The candidate does not add that alias, so the allocator portion of the original issue had already been addressed by related changes in the prepared base.
3. At the exact candidate commit, initialization supplies the same chunk size, checkpoint grid, and path cap used by `MambaComponent`. Independent checks also verify synchronous CoW into an already-owned slot, allocation and copy into a new slot, and the disabled-CoW boundary.
4. The candidate adds no native code. Its source import resolves to `/job/repo/python/sglang/...`; no native rebuild applies.
5. The three new regression methods are in `test_attention_patching.py` under a file-wide `mlx` availability guard. On this Linux host the complete file reports 41 skips. They directly construct the component rather than reproducing insertion/match/CoW/eviction through a real `UnifiedRadixCache`, so the exact original model-free contract is not retained as a runnable cross-platform regression.

## Architecture limitation

The review host is x86_64 Linux with PyTorch 2.11.0+rocm7.2 and an AMD Instinct MI350X. `mlx` is absent. ROCm GPU execution would not validate Apple MLX behavior, and no Apple Silicon, hybrid-SSM weights, or MLX serving environment was available. Consequently the first-request serving path, native snapshot semantics, prefix reuse rather than recomputation, eviction, and scheduler rollback remain unverified.

Raw logs, the exact candidate commit summary, and environment evidence are under `reports/j-538dad2c2804/raw/`.
