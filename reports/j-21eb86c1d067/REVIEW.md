# DeepSeek-V4 HiSparse accounting correction

Candidate: https://github.com/amdpilot-org/sglang/pull/930 at `a3fa16067a307d43108209a4235e91e0b7a3611c`

Independent review: https://github.com/amdpilot-org/sglang/pull/1010

The review counterexample was reproduced on the exact candidate with actual gfx950 allocations. Standard FP8, FP4, and unified layouts matched their independently enumerated owned tensors, while HiSparse owned an additional one-entry GPU `uint64` pointer table: 3,594,952 owned bytes versus 3,594,944 reported bytes.

The correction preserves the candidate's broader accounting and makes `HiSparseC4DevicePool.allocated_tensors()` include `data_ptrs`. Because the base constructor finalizes before that subclass tensor exists, the subclass refreshes `mem_usage` after allocating it. One- and three-layer unit boundaries verify the pointer table contributes 8 bytes per layer. The retained GPU fixture verifies all four layouts have zero byte delta after the change.

Full DeepSeek-V4 weights and the reported 8x H100 environment were unavailable, so serving-path and multi-rank behavior remain unverified.
